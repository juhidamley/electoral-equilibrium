#!/usr/bin/env python3
"""generate_synthetic.py — generate synthetic shock training data via Gemini 2.5 Pro.

Constructs a single prompt from the voter panel + real shock records + stratum IDs,
calls Gemini to generate 500–1,000 plausible hypothetical shock scenarios with
per-stratum delta bin estimates, then computes three diagnostics:

  MMD   : RBF-kernel MMD between synthetic and real Δμ distributions.
           λ = 1/(2σ²) using the median heuristic.
           mmd_weight = 0.5 × exp(−λ × MMD²)
  PCA   : Project synthetic batch onto top-2 PCs of real data. Flag if
           synthetic variance explained < 70%.
  PCD   : Frobenius norm of (real − synthetic) pairwise-correlation matrices,
           normalised by number of off-diagonal elements.

Output:
  data/finetune/synthetic.jsonl       — synthetic fine-tune records
  data/finetune/synthetic_diagnostics.json

Per CLAUDE.md: Gemini is permitted for synthetic data generation only.
Never use Gemini for the cleaning model (use Qwen/Mistral via mlx_lm instead).

Usage:
    export GEMINI_API_KEY=...
    python scripts/generate_synthetic.py
    python scripts/generate_synthetic.py --n-scenarios 300 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

_SHOCKS_PATH    = REPO_ROOT / "configs" / "shocks.json"
_PANEL_DIR      = REPO_ROOT / "data" / "panel"
_FINETUNE_DIR   = REPO_ROOT / "data" / "finetune"
_OUTPUT_JSONL   = _FINETUNE_DIR / "synthetic.jsonl"
_OUTPUT_DIAG    = _FINETUNE_DIR / "synthetic_diagnostics.json"

CANONICAL_RACES     = ["african_american", "latino", "asian", "white", "other_race"]
CANONICAL_RELIGIONS = ["evangelical", "catholic", "protestant", "secular", "jewish", "muslim", "other_rel"]
CANONICAL_GENDERS   = ["women", "men", "other_gender"]
ALL_BLOCS           = CANONICAL_RACES + CANONICAL_RELIGIONS + CANONICAL_GENDERS

DELTA_BINS = [
    "strong_neg", "mod_neg", "mild_neg", "slight_neg", "neutral",
    "slight_pos", "mild_pos", "mod_pos", "strong_pos",
]
BIN_MIDPOINTS = {
    "strong_neg": -0.120, "mod_neg": -0.070, "mild_neg": -0.035,
    "slight_neg": -0.012, "neutral": 0.000,  "slight_pos": +0.012,
    "mild_pos": +0.035,   "mod_pos": +0.070, "strong_pos": +0.120,
}


# ── Data loading ──────────────────────────────────────────────────────────────


def _load_real_shocks() -> list[dict]:
    if not _SHOCKS_PATH.exists():
        raise FileNotFoundError(f"shocks.json not found: {_SHOCKS_PATH}")
    return json.loads(_SHOCKS_PATH.read_text(encoding="utf-8"))


def _load_panel_csv() -> str:
    """Return a compact CSV string of voter panel data for the Gemini prompt."""
    import csv as _csv
    lines: list[str] = []
    for fname in ("panel_race.parquet", "panel_religion.parquet", "panel_gender.parquet"):
        path = _PANEL_DIR / fname
        if not path.exists():
            continue
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(path)
            df = table.to_pydict()
            cols = list(df.keys())
            rows = list(zip(*[df[c] for c in cols]))
            lines.append(f"# {fname}")
            lines.append(",".join(cols))
            for row in rows[:50]:  # cap to keep prompt manageable
                lines.append(",".join(str(v) for v in row))
        except Exception as exc:
            logger.warning("Could not load %s: %s", fname, exc)
    return "\n".join(lines) if lines else "(panel data not available)"


def _real_delta_matrix(shocks: list[dict]) -> np.ndarray | None:
    """Build matrix of real Δμ midpoints: shape (n_shocks, n_blocs).

    Uses delta bins stored in shocks.json if present (populated after scoring).
    Falls back to None if no real delta data is available.
    """
    rows: list[list[float]] = []
    for shock in shocks:
        deltas = shock.get("deltas") or {}
        row = [BIN_MIDPOINTS.get(deltas.get(b, "neutral"), 0.0) for b in ALL_BLOCS]
        if any(v != 0.0 for v in row):  # skip shocks with no data
            rows.append(row)
    if not rows:
        return None
    return np.array(rows, dtype=float)


# ── Prompt construction ───────────────────────────────────────────────────────


def _build_prompt(shocks: list[dict], panel_csv: str, n_scenarios: int) -> str:
    shock_examples = json.dumps(
        [
            {
                "id": s["id"],
                "description": s.get("description") or s.get("name") or s["id"],
                "date": s.get("date", ""),
            }
            for s in shocks[:30]
        ],
        indent=2,
    )

    blocs_str = json.dumps(
        {"race": CANONICAL_RACES, "religion": CANONICAL_RELIGIONS, "gender": CANONICAL_GENDERS},
        indent=2,
    )

    return f"""You are an expert in US electoral politics, voter demographics, and political shocks.

TASK: Generate {n_scenarios} synthetic hypothetical political shock events with per-stratum demographic sentiment delta estimates for training a political science model.

VOTER PANEL DATA (sample):
{panel_csv}

CANONICAL DEMOGRAPHIC STRATA:
{blocs_str}

REAL SHOCK EXAMPLES (for style reference):
{shock_examples}

DELTA BINS (9 tokens, ordered negative to positive):
{DELTA_BINS}

INSTRUCTIONS:
1. Generate {n_scenarios} diverse hypothetical shock events covering a wide range of political domains: economic shocks, social events, judicial decisions, foreign policy, natural disasters, cultural moments, electoral events.
2. For each shock, provide delta bin estimates for EVERY bloc in ALL THREE strata (race, religion, gender).
3. Estimates must be demographically grounded — e.g., a pro-immigration shock would likely be mild_pos for latino, slight_neg for white evangelical. Be realistic and internally consistent.
4. Avoid duplicating real shocks above. Create genuinely novel hypotheticals.
5. Cover a range of magnitudes: most should be slight_neg/slight_pos/neutral; strong bins should be rare.

OUTPUT FORMAT (strict JSON array, no markdown, no explanation):
[
  {{
    "id": "synthetic_<short_slug>",
    "description": "One-sentence description of the hypothetical shock",
    "party": "democrat",
    "delta_bins": {{
      "african_american": "<bin>",
      "latino": "<bin>",
      "asian": "<bin>",
      "white": "<bin>",
      "other_race": "<bin>",
      "evangelical": "<bin>",
      "catholic": "<bin>",
      "protestant": "<bin>",
      "secular": "<bin>",
      "jewish": "<bin>",
      "muslim": "<bin>",
      "other_rel": "<bin>",
      "women": "<bin>",
      "men": "<bin>",
      "other_gender": "<bin>"
    }}
  }},
  ...
]

Generate exactly {n_scenarios} objects. Output ONLY the JSON array."""


# ── Gemini call ───────────────────────────────────────────────────────────────


_BATCH_SIZE = 150  # scenarios per Gemini call; ~28k output tokens, safely within limits


def _call_gemini(prompt: str, api_key: str, model: str = "gemini-2.5-pro") -> str:
    try:
        import google.genai as genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai not installed. Run: pip install google-genai"
        ) from exc

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            temperature=0.9,
            max_output_tokens=65536,
        ),
    )
    return response.text


def _parse_response(raw: str) -> list[dict]:
    """Extract JSON array from Gemini response, with robust handling of:
    - Markdown code fences (```json ... ```)
    - Preamble/postamble prose before/after the array
    - Truncated responses (output-token limit hit mid-stream): falls back to
      extracting every complete {...} object individually so partial batches
      are not lost.
    """
    text = raw.strip()

    # Strip markdown code fences regardless of where they appear
    if "```" in text:
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()

    # Fast path: well-formed complete array
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass  # fall through to object-by-object extraction

    # Fallback: extract every complete top-level {...} object.
    # Works even when the outer array is truncated (no closing ']').
    objects: list[dict] = []
    depth = 0
    obj_start: int | None = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                fragment = text[obj_start : i + 1]
                try:
                    obj = json.loads(fragment)
                    if isinstance(obj, dict):
                        objects.append(obj)
                except json.JSONDecodeError:
                    pass
                obj_start = None

    if objects:
        logger.warning(
            "Used object-by-object fallback parser — response was likely truncated. "
            "Recovered %d complete scenario objects.", len(objects)
        )
        return objects

    raise ValueError("No JSON array or recoverable objects found in Gemini response")


# ── Diagnostics ───────────────────────────────────────────────────────────────


def _rbf_mmd(X: np.ndarray, Y: np.ndarray) -> tuple[float, float]:
    """Compute RBF-kernel MMD between two sample matrices.

    λ = 1/(2σ²) where σ² = median pairwise distance over real data (X).
    Returns (mmd_value, lambda_value).
    """
    from sklearn.metrics import pairwise_distances

    # Median heuristic on real data X
    dists_XX = pairwise_distances(X, metric="euclidean").ravel()
    dists_XX = dists_XX[dists_XX > 0]
    sigma2 = float(np.median(dists_XX ** 2)) if len(dists_XX) > 0 else 1.0
    lam = 1.0 / (2.0 * sigma2) if sigma2 > 0 else 1.0

    def k(A: np.ndarray, B: np.ndarray) -> float:
        D = pairwise_distances(A, B, metric="euclidean") ** 2
        return float(np.exp(-lam * D).mean())

    mmd2 = k(X, X) - 2 * k(X, Y) + k(Y, Y)
    return float(math.sqrt(max(mmd2, 0.0))), lam


def _pca_alignment(X_real: np.ndarray, Y_synth: np.ndarray) -> dict:
    """Check how much synthetic variance is explained by the top-2 real PCs."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=min(2, X_real.shape[1], X_real.shape[0]))
    pca.fit(X_real)

    Y_proj = pca.transform(Y_synth)
    var_total = float(np.var(Y_synth, axis=0).sum())
    var_explained = float(np.var(Y_proj, axis=0).sum())
    pct = var_explained / var_total if var_total > 0 else 0.0
    return {
        "variance_explained_by_top2_pcs": round(pct, 4),
        "flagged": pct < 0.70,
    }


def _pcd(X_real: np.ndarray, Y_synth: np.ndarray) -> float:
    """Frobenius norm of (real − synthetic) pairwise-correlation matrices,
    normalised by number of off-diagonal elements."""
    n = X_real.shape[1]
    if n < 2:
        return 0.0
    corr_real  = np.corrcoef(X_real.T)
    corr_synth = np.corrcoef(Y_synth.T)
    diff = corr_real - corr_synth
    # Zero the diagonal before Frobenius norm
    np.fill_diagonal(diff, 0.0)
    n_off_diag = n * (n - 1)
    return float(np.linalg.norm(diff, "fro") / n_off_diag) if n_off_diag > 0 else 0.0


def _compute_diagnostics(
    real_matrix: np.ndarray | None,
    synth_matrix: np.ndarray,
) -> dict:
    """Compute MMD, PCA alignment, and PCD between real and synthetic distributions."""
    if real_matrix is None or real_matrix.shape[0] < 3:
        return {
            "mmd": None,
            "mmd_weight": None,
            "lambda": None,
            "pca_alignment": None,
            "pcd": None,
            "note": "Insufficient real shock data for diagnostics (need ≥3 scored shocks)",
        }

    mmd, lam = _rbf_mmd(real_matrix, synth_matrix)
    mmd_weight = 0.5 * math.exp(-lam * mmd ** 2)
    pca = _pca_alignment(real_matrix, synth_matrix)
    pcd = _pcd(real_matrix, synth_matrix)

    return {
        "mmd": round(mmd, 6),
        "mmd_weight": round(mmd_weight, 6),
        "lambda": round(lam, 6),
        "pca_alignment": pca,
        "pcd": round(pcd, 6),
    }


# ── Output writing ────────────────────────────────────────────────────────────


def _to_finetune_record(scenario: dict, mmd_weight: float, now_iso: str) -> dict:
    """Convert a Gemini scenario to a fine-tuning JSONL record."""
    delta_bins = scenario.get("delta_bins") or {}
    return {
        "id": scenario["id"],
        "description": scenario["description"],
        "party": scenario.get("party", "democrat"),
        "delta_bins": {b: delta_bins.get(b, "neutral") for b in ALL_BLOCS},
        "synthetic": True,
        "mmd_weight": round(mmd_weight, 6),
        "weight": 0.5,
        "collected_at": now_iso,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic shock training data via Gemini 2.5 Pro",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-scenarios", type=int, default=500)
    p.add_argument("--model",       default="gemini-2.5-pro")
    p.add_argument("--api-key",     default=None,
                   help="Gemini API key (default: GEMINI_API_KEY env var)")
    p.add_argument("--output",      type=Path, default=_OUTPUT_JSONL)
    p.add_argument("--diagnostics", type=Path, default=_OUTPUT_DIAG)
    p.add_argument("--dry-run",     action="store_true",
                   help="Skip Gemini call; print prompt and exit")
    p.add_argument("--verbose",     action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    import os
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI")
    if not api_key and not args.dry_run:
        logger.error(
            "GEMINI_API_KEY not set. Export it or pass --api-key. Use --dry-run to skip."
        )
        sys.exit(1)

    shocks = _load_real_shocks()
    logger.info("Loaded %d real shocks from %s", len(shocks), _SHOCKS_PATH)

    panel_csv = _load_panel_csv()
    real_matrix = _real_delta_matrix(shocks)
    if real_matrix is not None:
        logger.info("Real delta matrix: %s", real_matrix.shape)
    else:
        logger.warning("No real delta data available — diagnostics will be skipped.")

    if args.dry_run:
        prompt = _build_prompt(shocks, panel_csv, min(_BATCH_SIZE, args.n_scenarios))
        print(f"[dry-run] Prompt length: {len(prompt)} chars")
        print(prompt[:2000])
        return

    # Batch across multiple Gemini calls to stay within output-token limits.
    # _BATCH_SIZE = 150 scenarios × ~750 chars ÷ 4 chars/token ≈ 28k tokens/call.
    all_scenarios: list[dict] = []
    remaining = args.n_scenarios
    batch_num = 0
    seen_ids: set[str] = set()

    while remaining > 0:
        batch_size = min(_BATCH_SIZE, remaining)
        batch_num += 1
        logger.info(
            "Batch %d: requesting %d scenarios from Gemini %s (need %d more total)",
            batch_num, batch_size, args.model, remaining,
        )
        prompt = _build_prompt(shocks, panel_csv, batch_size)
        try:
            raw = _call_gemini(prompt, api_key=api_key, model=args.model)
        except Exception as exc:
            logger.error("Gemini call failed on batch %d: %s", batch_num, exc)
            break
        logger.info("Batch %d response: %d chars", batch_num, len(raw))

        try:
            scenarios = _parse_response(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Failed to parse batch %d response: %s", batch_num, exc)
            logger.debug("Raw response (first 2000 chars):\n%s", raw[:2000])
            break

        # Validate and dedup
        batch_valid = 0
        for s in scenarios:
            sid = s.get("id", "")
            if sid in seen_ids:
                continue
            if s.get("party") not in ("democrat", "republican"):
                logger.warning("Scenario '%s' has invalid party '%s' — skipping", sid, s.get("party"))
                continue
            bins = s.get("delta_bins") or {}
            missing = [b for b in ALL_BLOCS if b not in bins]
            if missing:
                for b in missing:
                    bins[b] = "neutral"
                s["delta_bins"] = bins
            if bins.get("african_american") not in DELTA_BINS:
                logger.warning("Scenario '%s' has invalid bin values — skipping", sid)
                continue
            seen_ids.add(sid)
            all_scenarios.append(s)
            batch_valid += 1

        logger.info("Batch %d: %d/%d valid scenarios (total so far: %d)",
                    batch_num, batch_valid, len(scenarios), len(all_scenarios))
        remaining -= batch_valid

        if batch_valid == 0:
            logger.warning("Batch %d yielded 0 valid scenarios — stopping.", batch_num)
            break

    valid_scenarios = all_scenarios

    logger.info("%d/%d scenarios passed validation", len(valid_scenarios), len(scenarios))

    logger.info("Total valid scenarios collected: %d / %d requested", len(valid_scenarios), args.n_scenarios)
    if not valid_scenarios:
        logger.error("No valid scenarios collected — nothing to write.")
        sys.exit(1)

    # Build synthetic delta matrix for diagnostics
    synth_rows = [
        [BIN_MIDPOINTS.get(s["delta_bins"].get(b, "neutral"), 0.0) for b in ALL_BLOCS]
        for s in valid_scenarios
    ]
    synth_matrix = np.array(synth_rows, dtype=float)

    diagnostics = _compute_diagnostics(real_matrix, synth_matrix)
    mmd_weight = diagnostics.get("mmd_weight") or 0.5
    logger.info(
        "Diagnostics: MMD=%.4f weight=%.4f PCD=%.4f PCA_flagged=%s",
        diagnostics.get("mmd") or 0,
        mmd_weight,
        diagnostics.get("pcd") or 0,
        diagnostics.get("pca_alignment", {}).get("flagged") if diagnostics.get("pca_alignment") else "N/A",
    )

    now_iso = datetime.now(tz=timezone.utc).isoformat()

    # Write JSONL
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for s in valid_scenarios:
            record = _to_finetune_record(s, mmd_weight, now_iso)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records → %s", len(valid_scenarios), args.output)

    # Write diagnostics
    diagnostics["n_scenarios_requested"] = args.n_scenarios
    diagnostics["n_scenarios_written"]   = len(valid_scenarios)
    diagnostics["model"]                 = args.model
    diagnostics["generated_at"]          = now_iso
    args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote diagnostics → %s", args.diagnostics)


if __name__ == "__main__":
    main()
