"""compile_hailo: Export all-MiniLM-L6-v2 to ONNX and compile for Hailo-8L NPU.

Steps
-----
1. Load all-MiniLM-L6-v2 via sentence-transformers
2. Export to ONNX with dummy input (batch=1, seq=128)
3. hailo optimize minilm.onnx --hw-arch hailo8l
4. hailo compile minilm.onnx --hw-arch hailo8l -o adapters/all-minilm-l6-v2.hef
5. Verify: HEF loads via hailo_platform

On ANY failure the script writes pi_npu_enabled=false to configs/base.json and
exits 1. The bio server CPU fallback path takes over immediately — no manual
intervention required. Week 4 is never blocked by NPU compilation.

Usage (on Raspberry Pi 5 with Hailo-8L AI HAT+):
    python scripts/compile_hailo.py

Force CPU-only (skip compile, just set flag):
    python scripts/compile_hailo.py --force-cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
HW_ARCH = "hailo8l"
ONNX_PATH = Path("models/minilm.onnx")
HEF_PATH = Path("adapters/all-minilm-l6-v2.hef")
CONFIG_PATH = Path("configs/base.json")

# batch=1, seq_len=128 — matches the Pi bio server single-bio inference shape
DUMMY_BATCH = 1
DUMMY_SEQ_LEN = 128


# ── Step 1: ONNX export ───────────────────────────────────────────────────────


def export_onnx(output_path: Path) -> None:
    """Load all-MiniLM-L6-v2 and export to ONNX."""
    logger.info("Loading %s via sentence-transformers...", MODEL_NAME)
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers and torch are required.\n"
            "Run: pip install sentence-transformers torch"
        ) from exc

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    transformer = model[0].auto_model

    dummy_ids = torch.zeros((DUMMY_BATCH, DUMMY_SEQ_LEN), dtype=torch.long)
    dummy_mask = torch.ones((DUMMY_BATCH, DUMMY_SEQ_LEN), dtype=torch.long)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting ONNX → %s", output_path)

    torch.onnx.export(
        transformer,
        (dummy_ids, dummy_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    logger.info("ONNX export complete: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)


# ── Step 2: hailo optimize ────────────────────────────────────────────────────


def run_hailo_optimize(onnx_path: Path) -> None:
    """Run hailo optimize on the ONNX model."""
    cmd = ["hailo", "optimize", str(onnx_path), "--hw-arch", HW_ARCH]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"hailo optimize failed (exit {result.returncode}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    logger.info("hailo optimize complete.")


# ── Step 3: hailo compile ─────────────────────────────────────────────────────


def run_hailo_compile(onnx_path: Path, hef_path: Path) -> None:
    """Compile ONNX to HEF via hailo compile."""
    hef_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "hailo",
        "compile",
        "--hw-arch",
        HW_ARCH,
        str(onnx_path),
        "-o",
        str(hef_path),
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"hailo compile failed (exit {result.returncode}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    logger.info("hailo compile complete → %s (%.1f MB)", hef_path, hef_path.stat().st_size / 1e6)


# ── Step 4: verify HEF ────────────────────────────────────────────────────────


def verify_hef(hef_path: Path) -> None:
    """Verify the compiled HEF loads without error via hailo_platform."""
    logger.info("Verifying HEF: %s", hef_path)
    try:
        from hailo_platform import HEF  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "hailo_platform is not installed. Install the Hailo SDK on the Pi "
            "to verify the HEF. On non-Pi machines use --skip-verify."
        ) from exc
    HEF(str(hef_path))
    logger.info("HEF verification passed — model loads cleanly.")


# ── Config flag ───────────────────────────────────────────────────────────────


def set_npu_enabled(config_path: Path, enabled: bool) -> None:
    """Write pi_npu_enabled to configs/base.json."""
    if not config_path.exists():
        logger.warning("Config not found at %s — skipping pi_npu_enabled update.", config_path)
        return
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    config["pi_npu_enabled"] = enabled
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    logger.info("Wrote pi_npu_enabled=%s → %s", enabled, config_path)


# ── Entry point ───────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compile all-MiniLM-L6-v2 for the Hailo-8L NPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--onnx", default=str(ONNX_PATH), help="ONNX output path")
    p.add_argument("--hef", default=str(HEF_PATH), help="HEF output path")
    p.add_argument("--config", default=str(CONFIG_PATH), help="Pipeline config JSON")
    p.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip ONNX export — use existing file (saves ~30s if already exported)",
    )
    p.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip HEF verification (use on non-Pi machines that lack hailo_platform)",
    )
    p.add_argument(
        "--force-cpu",
        action="store_true",
        help="Skip compilation entirely and write pi_npu_enabled=false to config",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    onnx_path = Path(args.onnx)
    hef_path = Path(args.hef)
    config_path = Path(args.config)

    if args.force_cpu:
        logger.warning("--force-cpu: skipping NPU compilation, setting pi_npu_enabled=false.")
        set_npu_enabled(config_path, enabled=False)
        logger.info(
            "CPU fallback active. Bio server will run at ~20 ms/bio on Pi CPU — correct but slower."
        )
        sys.exit(0)

    try:
        if args.skip_export:
            if not onnx_path.exists():
                raise FileNotFoundError(
                    f"--skip-export set but ONNX not found: {onnx_path}. "
                    "Remove --skip-export to regenerate it."
                )
            logger.info("Skipping ONNX export — using existing %s", onnx_path)
        else:
            export_onnx(onnx_path)

        run_hailo_optimize(onnx_path)
        run_hailo_compile(onnx_path, hef_path)

        if not args.skip_verify:
            verify_hef(hef_path)
        else:
            logger.info("Skipping HEF verification (--skip-verify).")

        set_npu_enabled(config_path, enabled=True)
        logger.info(
            "NPU compilation succeeded. pi_npu_enabled=true written to %s. "
            "Start the bio server with: uvicorn scripts.pi_bio_server:app --host 0.0.0.0 --port 9000",
            config_path,
        )
        sys.exit(0)

    except Exception as exc:
        logger.error("NPU compilation FAILED: %s", exc)
        logger.warning(
            "Activating CPU fallback: writing pi_npu_enabled=false to %s. "
            "Bio server will run at ~20 ms/bio on Pi CPU — slower but fully correct. "
            "Week 3/4 work is not blocked.",
            config_path,
        )
        set_npu_enabled(config_path, enabled=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
