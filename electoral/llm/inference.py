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

from electoral.core.types import (
    BIN_MIDPOINTS,
    CANONICAL_GENDERS,
    CANONICAL_RACES,
    CANONICAL_RELIGIONS,
    DELTA_BINS,
)
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
        from enum import Enum
        from pydantic import create_model

        # str-Enum of the nine canonical bins. Pydantic 2.x handles Enums
        # natively. Build the model with create_model (the supported dynamic
        # API) — the manual type()+__annotations__ hack mis-reads the
        # (Type, default) tuple as the annotation and fails on 2.13.
        _BinToken = Enum("BinToken", {b: b for b in DELTA_BINS}, type=str)

        field_defs = {bloc: (_BinToken, ...) for bloc in _ALL_BLOCS}
        _DeltaBinsModel = create_model("DeltaBinsModel", **field_defs)
        return _DeltaBinsModel
    except ImportError:
        return None


# ── Model loading ─────────────────────────────────────────────────────────────


def load_model(
    adapter_path: str | None = None,
    base_model: str = "mistralai/Mistral-7B-v0.3",
    *,
    device: str | None = None,
    use_quantization: bool = False,
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
    use_quantization:
        When True, load in 4-bit NF4 (BitsAndBytes) for training. Default False
        loads in float16 for inference — quantization causes NoneType errors when
        outlines tries to reorder the KV cache across PEFT adapter layers.

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

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    if use_quantization:
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        except ImportError:
            log.warning("bitsandbytes not available — falling back to float16")
            quant_config = None
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quant_config,
            device_map="auto",
        )
    else:
        # device_map="auto" on MPS (Apple Silicon) places some layers in meta tensors,
        # which causes PEFT's _update_offload to fail with a double-nested key prefix.
        # Load to a single device explicitly to avoid the offload code path.
        if torch.backends.mps.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
            ).to("mps")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto",
            )

    if adapter_path:
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
    seed: int = 42,
) -> dict[str, str]:
    """Use outlines for constrained JSON generation."""
    import outlines

    DeltaBinsModel = _get_delta_bins_model()
    if DeltaBinsModel is None:
        raise ImportError("pydantic is required for constrained generation")

    _example = {
        "description": shock_text,
        "party": party,
        "year": "",
        "source": "",
        "news_roberta_scores": {},
        "social_roberta_scores": {},
    }
    prompt = f"<s>{format_prompt(_example)}\n"
    # outlines needs its own model wrapper (capital-T Transformers); passing the
    # bare HF model makes outlines reach for model.tokenizer, which a
    # MistralForCausalLM does not have. Disable KV cache to avoid the
    # reorder_kv_cache NoneType crash under PEFT adapters.
    model.config.use_cache = False
    wrapped = outlines.models.Transformers(model, tokenizer)
    generator = outlines.generate.json(wrapped, DeltaBinsModel)
    # Seed for reproducibility. Global torch seed covers any version of
    # outlines; the rng kwarg is attempted first for explicit control.
    import torch
    torch.manual_seed(seed)
    try:
        _rng = torch.Generator(device=getattr(model, "device", "cpu"))
        _rng.manual_seed(seed)
        raw = dict(generator(prompt, rng=_rng))
    except TypeError:
        # this outlines version's generator() takes no rng kwarg; global seed applies
        raw = dict(generator(prompt))
    # outlines returns Enum members; unwrap to the canonical str token
    result = {k: (v.value if hasattr(v, "value") else v) for k, v in raw.items()}
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

    _example = {
        "description": shock_text,
        "party": party,
        "year": "",
        "source": "",
        "news_roberta_scores": {},
        "social_roberta_scores": {},
    }
    prompt = f"<s>{format_prompt(_example)}\n"
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

    generated = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
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
    seed: int = 42,
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
            result = _predict_constrained(shock_text, party, model, tokenizer, max_tokens, seed=seed)
            return _fill_missing(result)
        except ImportError as exc:
            log.warning("outlines not available (%s); falling back to greedy decode", exc)
        except Exception as exc:
            log.warning("constrained generation failed (%s); falling back to greedy decode", exc)

    return _predict_greedy(shock_text, party, model, tokenizer, max_tokens)


# ── ShockEstimator ────────────────────────────────────────────────────────────


class ShockEstimator:
    """Stateful estimator: loads model once, exposes estimate() per event.

    Uses outlines constrained decoding with ShockResponseSchema so every
    generated token is guaranteed to be a canonical DELTA_BINS value with the
    correct stratum key names. The same schema class is used as the FastAPI
    response_model in shock_endpoint.py.
    """

    def __init__(
        self,
        adapter_path: str,
        base_model: str = "mistralai/Mistral-7B-v0.3",
    ) -> None:
        self.adapter_path = adapter_path
        self.model, self.tokenizer = load_model(adapter_path, base_model)
        # Disable KV cache reuse — outlines tries to reorder the cache across
        # PEFT model layers, which hits a NoneType error when past_key_values
        # is None on the first forward pass through the adapter.
        self.model.config.use_cache = False
        try:
            import outlines

            self._outlines_model = outlines.models.Transformers(self.model, self.tokenizer)
        except ImportError:
            self._outlines_model = None
            log.warning("outlines not installed — ShockEstimator.estimate() will raise on call")

    def estimate(self, event: dict[str, Any], intensity: float = 1.0) -> Any:
        """Run constrained generation for one shock event.

        Parameters
        ----------
        event:
            Full finetune record dict with keys: shock_id (or shock), cycle
            (or year), party, description, news_roberta_scores,
            social_roberta_scores.
        intensity:
            Scalar multiplier applied to all numeric delta values after
            conversion. 1.0 = full shock; 0.5 = half-strength.

        Returns
        -------
        ShockResponseData — validated frozen dataclass.
        """
        from electoral.artifacts import ShockResponseData, ShockResponseSchema

        if self._outlines_model is None:
            raise ImportError("outlines is required for ShockEstimator.estimate()")

        import outlines

        # (i) Prompt
        prompt = format_prompt(event)

        # (ii)+(iii) Constrained generation — outlines guarantees valid bin tokens
        generator = outlines.generate.json(self._outlines_model, ShockResponseSchema)
        schema_out: ShockResponseSchema = generator(prompt)

        # (iv) Nested Pydantic → plain str dicts
        bins_race: dict[str, str] = schema_out.delta_bins_race.model_dump()
        bins_religion: dict[str, str] = schema_out.delta_bins_religion.model_dump()
        bins_gender: dict[str, str] = schema_out.delta_bins_gender.model_dump()

        # (iv) Bin tokens → numeric midpoints
        deltas_race = {k: BIN_MIDPOINTS[v] for k, v in bins_race.items()}
        deltas_religion = {k: BIN_MIDPOINTS[v] for k, v in bins_religion.items()}
        deltas_gender = {k: BIN_MIDPOINTS[v] for k, v in bins_gender.items()}

        # (v) Scale by intensity, then clip to [-0.15, 0.15]
        deltas_race = {k: v * intensity for k, v in deltas_race.items()}
        deltas_race = {k: max(-0.15, min(0.15, v)) for k, v in deltas_race.items()}
        deltas_religion = {k: v * intensity for k, v in deltas_religion.items()}
        deltas_religion = {k: max(-0.15, min(0.15, v)) for k, v in deltas_religion.items()}
        deltas_gender = {k: v * intensity for k, v in deltas_gender.items()}
        deltas_gender = {k: max(-0.15, min(0.15, v)) for k, v in deltas_gender.items()}

        # (vi) delta_eff: use LLM-predicted value scaled by intensity
        delta_eff = schema_out.delta_eff * intensity

        # (vi) 5×5 identity covariance — placeholder until real estimation
        n = len(CANONICAL_RACES)
        covariance = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

        shock = str(event.get("shock_id") or event.get("shock", ""))
        cycle = int(event.get("cycle") or event.get("year") or 2024)
        party = str(event.get("party", "democrat"))

        # (vi)+(vii) Construct — __post_init__ validates on creation
        return ShockResponseData(
            shock=shock,
            cycle=cycle,
            party=party,
            delta_bins_race=bins_race,
            delta_bins_religion=bins_religion,
            delta_bins_gender=bins_gender,
            deltas_race=deltas_race,
            deltas_religion=deltas_religion,
            deltas_gender=deltas_gender,
            delta_eff=delta_eff,
            covariance=covariance,
            source="llm_unified",
        )
