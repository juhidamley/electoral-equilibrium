"""Constrained-decoding inference for the fine-tuned Mistral 7B adapter.

predict_delta_bins() generates delta bin tokens for all 15 demographic blocs
using the outlines library for structured JSON generation. This guarantees
that every output token is one of the 9 canonical DELTA_BINS values.

Usage (after training):
    from electoral.llm.inference import load_model, predict_delta_bins

    model, tokenizer = load_model(adapter_path="adapters/mistral-7b-electoral")
    bins = predict_delta_bins(
        shock_text="Supreme Court overturns Roe v Wade",
        party="democrat",
        model=model,
        tokenizer=tokenizer,
    )
    # bins = {"african_american": "mod_pos", "latino": "slight_pos", ...}

If the outlines library is unavailable, falls back to greedy decoding with
regex-based extraction of bin tokens from free-form output.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS, DELTA_BINS
from electoral.llm.trainer import format_prompt

log = logging.getLogger(__name__)

_ALL_BLOCS: list[str] = list(CANONICAL_RACES) + list(CANONICAL_RELIGIONS) + list(CANONICAL_GENDERS)

# ── Pydantic schema for constrained JSON generation ──────────────────────────
# Built lazily (pydantic may not be installed in all envs).
_DeltaBinsModel: Any = None


def _get_delta_bins_model() -> Any:
    global _DeltaBinsModel
    if _DeltaBinsModel is not None:
        return _DeltaBinsModel
    try:
        from pydantic import BaseModel
        from typing import Literal

        _BinToken = Literal[tuple(DELTA_BINS)]  # type: ignore[valid-type]

        field_defs = {bloc: (_BinToken, ...) for bloc in _ALL_BLOCS}  # type: ignore[misc]
        _DeltaBinsModel = type("DeltaBinsModel", (BaseModel,), {"__annotations__": field_defs})
        return _DeltaBinsModel
    except ImportError:
        return None


# ── Model loading ─────────────────────────────────────────────────────────────


def load_model(
    adapter_path: str | None = None,
    base_model: str = "mistralai/Mistral-7B-v0.3",
    *,
    device: str | None = None,
) -> tuple[Any, Any]:
    """Load the base model + optional QLoRA adapter.

    Parameters
    ----------
    adapter_path:
        Path to the saved PEFT adapter directory. When None, loads the base model.
    base_model:
        HuggingFace model ID for the base Mistral checkpoint.
    device:
        "cuda", "cpu", or None (auto-detect).

    Returns
    -------
    (model, tokenizer)
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Inference requires transformers + torch. "
            "Install with: pip install transformers torch"
        ) from exc

    if device is None:
        import torch as _torch

        device = "cuda" if _torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path or base_model,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # noqa: F401
        import torch as _torch

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=_torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    except ImportError:
        quant_config = None

    model = AutoModelForCausalLM.from_pretrained(
        adapter_path or base_model,
        quantization_config=quant_config,
        device_map="auto" if device == "cuda" else None,
    )

    if adapter_path and quant_config is None:
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
            log.info("Loaded PEFT adapter from %s", adapter_path)
        except ImportError:
            log.warning("peft not available — loading base model without adapter")

    model.eval()
    return model, tokenizer


# ── Constrained generation ────────────────────────────────────────────────────


def _predict_constrained(
    shock_text: str,
    party: str,
    model: Any,
    tokenizer: Any,
    max_tokens: int = 512,
) -> dict[str, str]:
    """Use outlines for constrained JSON generation."""
    import outlines

    DeltaBinsModel = _get_delta_bins_model()
    if DeltaBinsModel is None:
        raise ImportError("pydantic is required for constrained generation")

    prompt = f"<s>{format_prompt(shock_text, party)}\n"
    generator = outlines.generate.json(model, DeltaBinsModel)
    result = dict(generator(prompt))
    invalid = {k: v for k, v in result.items() if v not in DELTA_BINS}
    if invalid:
        log.warning(
            "_predict_constrained: outlines returned %d invalid bin token(s): %s",
            len(invalid),
            invalid,
        )
    return result


def _predict_greedy(
    shock_text: str,
    party: str,
    model: Any,
    tokenizer: Any,
    max_tokens: int = 512,
) -> dict[str, str]:
    """Fallback: greedy decode + regex extraction of bin tokens."""
    import re
    import torch

    prompt = f"<s>{format_prompt(shock_text, party)}\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, "device") and str(model.device) != "cpu":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    log.debug("Raw generation: %s", generated[:200])

    # Try JSON parse first
    try:
        parsed = json.loads(generated.strip())
        if isinstance(parsed, dict):
            valid = {k: v for k, v in parsed.items() if k in _ALL_BLOCS and v in DELTA_BINS}
            if valid:
                return _fill_missing(valid)
    except json.JSONDecodeError:
        pass

    # Regex fallback: extract "bloc: token" or "bloc": "token" patterns
    bin_pattern = "|".join(DELTA_BINS)
    findings: dict[str, str] = {}
    for bloc in _ALL_BLOCS:
        m = re.search(
            rf'["\']?{re.escape(bloc)}["\']?\s*[:\s]+["\']?({bin_pattern})["\']?',
            generated,
            re.IGNORECASE,
        )
        if m:
            findings[bloc] = m.group(1).lower()

    return _fill_missing(findings)


def _fill_missing(bins: dict[str, str], default: str = "neutral") -> dict[str, str]:
    """Ensure all 15 blocs are present, filling any missing with default."""
    return {b: bins.get(b, default) for b in _ALL_BLOCS}


def predict_delta_bins(
    shock_text: str,
    party: str,
    model: Any,
    tokenizer: Any,
    *,
    use_constrained: bool = True,
    max_tokens: int = 512,
) -> dict[str, str]:
    """Predict delta bins for all 15 demographic blocs.

    Parameters
    ----------
    shock_text:
        Free-text description of the political shock event.
    party:
        "democrat" or "republican".
    model, tokenizer:
        Loaded by load_model().
    use_constrained:
        When True (default), attempt outlines constrained generation first.
        Falls back to greedy decode if outlines is unavailable.
    max_tokens:
        Maximum number of new tokens to generate.

    Returns
    -------
    dict mapping each of the 15 canonical bloc IDs to a DELTA_BINS token.
    All 15 keys are always present (missing ones default to "neutral").
    """
    if party not in ("democrat", "republican"):
        raise ValueError(f"party must be 'democrat' or 'republican', got {party!r}")

    if use_constrained:
        try:
            result = _predict_constrained(shock_text, party, model, tokenizer, max_tokens)
            return _fill_missing(result)
        except ImportError as exc:
            log.warning("outlines not available (%s); falling back to greedy decode", exc)
        except Exception as exc:
            log.warning("constrained generation failed (%s); falling back to greedy decode", exc)

    return _predict_greedy(shock_text, party, model, tokenizer, max_tokens)
