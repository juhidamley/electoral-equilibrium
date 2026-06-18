"""One-off generator for hand-authored Newsom 2028 synthetic fine-tune rows.

Writes data/finetune/newsom_synthetic.jsonl in the SAME flat schema as the
existing data/finetune/synthetic.jsonl (id / description / party / delta_bins /
synthetic / source / mmd_weight / weight / collected_at), validates every row
against the canonical bloc lists and DELTA_BINS, then appends to synthetic.jsonl.

Provenance: source="hand_authored_prior" — these encode expected Newsom-specific
dynamics and intentionally bypass the MMD/PCA/PCD diagnostics in
generate_synthetic.py. Kept auditable so they are never mistaken for
empirically-derived rows.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from electoral.core.types import (
    CANONICAL_GENDERS,
    CANONICAL_RACES,
    CANONICAL_RELIGIONS,
    DELTA_BINS,
)

SHOCK_ID = "newsom_nomination_2028"
FINETUNE_DIR = Path("data/finetune")
NEWSOM_PATH = FINETUNE_DIR / "newsom_synthetic.jsonl"
SYNTHETIC_PATH = FINETUNE_DIR / "synthetic.jsonl"

# Direction (held constant across all rows; only magnitude varies):
#   hurt: white (working class), latino (CA rightward shift), evangelical
#   neutral→pos: african_american (historic loyalty), secular, women
#   mixed/slightly neg: asian (CA crime/business climate), men
# Each tuple is one record: (description, delta_bins for all 15 blocs).
RECORDS: list[tuple[str, dict[str, str]]] = [
    (
        "Gavin Newsom secures the 2028 Democratic presidential nomination, "
        "energizing coastal progressives but alarming white working-class voters in "
        "the industrial Midwest.",
        {
            "african_american": "slight_pos",
            "latino": "slight_neg",
            "asian": "slight_neg",
            "white": "mod_neg",
            "other_race": "neutral",
            "evangelical": "strong_neg",
            "catholic": "mild_neg",
            "protestant": "slight_neg",
            "secular": "slight_pos",
            "jewish": "neutral",
            "muslim": "slight_neg",
            "other_rel": "neutral",
            "women": "slight_pos",
            "men": "slight_neg",
            "other_gender": "slight_pos",
        },
    ),
    (
        "The Democratic Party nominates Governor Gavin Newsom for president in 2028; "
        "his California record on crime and cost of living becomes a national flashpoint.",
        {
            "african_american": "slight_pos",
            "latino": "mild_neg",
            "asian": "slight_neg",
            "white": "mild_neg",
            "other_race": "slight_neg",
            "evangelical": "strong_neg",
            "catholic": "mild_neg",
            "protestant": "mild_neg",
            "secular": "slight_pos",
            "jewish": "slight_pos",
            "muslim": "neutral",
            "other_rel": "neutral",
            "women": "slight_pos",
            "men": "mild_neg",
            "other_gender": "slight_pos",
        },
    ),
    (
        "Newsom clinches the 2028 nomination after a contested primary, leaning into a "
        "progressive platform that depresses enthusiasm among blue-collar white voters.",
        {
            "african_american": "neutral",
            "latino": "slight_neg",
            "asian": "neutral",
            "white": "mod_neg",
            "other_race": "neutral",
            "evangelical": "mod_neg",
            "catholic": "slight_neg",
            "protestant": "slight_neg",
            "secular": "mild_pos",
            "jewish": "neutral",
            "muslim": "slight_neg",
            "other_rel": "slight_neg",
            "women": "slight_pos",
            "men": "slight_neg",
            "other_gender": "slight_pos",
        },
    ),
    (
        "Gavin Newsom becomes the Democratic standard-bearer for 2028; Latino voters in "
        "California, already drifting rightward during his tenure, remain skeptical.",
        {
            "african_american": "slight_pos",
            "latino": "mild_neg",
            "asian": "slight_neg",
            "white": "mild_neg",
            "other_race": "slight_neg",
            "evangelical": "strong_neg",
            "catholic": "mild_neg",
            "protestant": "slight_neg",
            "secular": "slight_pos",
            "jewish": "neutral",
            "muslim": "slight_neg",
            "other_rel": "neutral",
            "women": "neutral",
            "men": "mild_neg",
            "other_gender": "slight_pos",
        },
    ),
    (
        "With Newsom atop the 2028 Democratic ticket, evangelical leaders mobilize sharply "
        "against him while secular and younger voters rally to his side.",
        {
            "african_american": "slight_pos",
            "latino": "slight_neg",
            "asian": "neutral",
            "white": "mild_neg",
            "other_race": "neutral",
            "evangelical": "strong_neg",
            "catholic": "mild_neg",
            "protestant": "mild_neg",
            "secular": "mild_pos",
            "jewish": "slight_pos",
            "muslim": "neutral",
            "other_rel": "neutral",
            "women": "slight_pos",
            "men": "slight_neg",
            "other_gender": "mild_pos",
        },
    ),
    (
        "The 2028 Democratic convention selects Gavin Newsom; his polished image plays well "
        "with college-educated suburbanites but poorly with rural white communities.",
        {
            "african_american": "slight_pos",
            "latino": "slight_neg",
            "asian": "slight_neg",
            "white": "mod_neg",
            "other_race": "neutral",
            "evangelical": "mod_neg",
            "catholic": "slight_neg",
            "protestant": "slight_neg",
            "secular": "slight_pos",
            "jewish": "slight_pos",
            "muslim": "neutral",
            "other_rel": "neutral",
            "women": "slight_pos",
            "men": "slight_neg",
            "other_gender": "slight_pos",
        },
    ),
    (
        "Newsom wins the Democratic nomination in 2028 amid attacks on California's "
        "homelessness and public-safety record, eroding support among white moderates.",
        {
            "african_american": "neutral",
            "latino": "slight_neg",
            "asian": "mild_neg",
            "white": "mod_neg",
            "other_race": "slight_neg",
            "evangelical": "strong_neg",
            "catholic": "mild_neg",
            "protestant": "mild_neg",
            "secular": "slight_pos",
            "jewish": "neutral",
            "muslim": "slight_neg",
            "other_rel": "slight_neg",
            "women": "neutral",
            "men": "mild_neg",
            "other_gender": "slight_pos",
        },
    ),
    (
        "Gavin Newsom is nominated for president by the Democrats in 2028; Asian American "
        "small-business owners voice concern over California's business and tax climate.",
        {
            "african_american": "slight_pos",
            "latino": "slight_neg",
            "asian": "mild_neg",
            "white": "mild_neg",
            "other_race": "neutral",
            "evangelical": "strong_neg",
            "catholic": "slight_neg",
            "protestant": "slight_neg",
            "secular": "slight_pos",
            "jewish": "neutral",
            "muslim": "neutral",
            "other_rel": "neutral",
            "women": "slight_pos",
            "men": "slight_neg",
            "other_gender": "slight_pos",
        },
    ),
    (
        "Democrats rally behind Newsom for the 2028 race; African American voters extend "
        "their historic party loyalty even as white working-class support slips.",
        {
            "african_american": "mild_pos",
            "latino": "slight_neg",
            "asian": "slight_neg",
            "white": "mod_neg",
            "other_race": "neutral",
            "evangelical": "mod_neg",
            "catholic": "slight_neg",
            "protestant": "slight_neg",
            "secular": "slight_pos",
            "jewish": "slight_pos",
            "muslim": "neutral",
            "other_rel": "neutral",
            "women": "slight_pos",
            "men": "slight_neg",
            "other_gender": "slight_pos",
        },
    ),
    (
        "Governor Newsom emerges as the 2028 Democratic nominee, his climate and social "
        "agenda thrilling secular progressives but hardening evangelical opposition.",
        {
            "african_american": "slight_pos",
            "latino": "slight_neg",
            "asian": "neutral",
            "white": "mild_neg",
            "other_race": "neutral",
            "evangelical": "strong_neg",
            "catholic": "mild_neg",
            "protestant": "mild_neg",
            "secular": "mild_pos",
            "jewish": "slight_pos",
            "muslim": "slight_neg",
            "other_rel": "neutral",
            "women": "slight_pos",
            "men": "slight_neg",
            "other_gender": "mild_pos",
        },
    ),
    (
        "Newsom captures the 2028 Democratic nomination; his support among Latino voters "
        "lags expectations as immigration and economy dominate the conversation.",
        {
            "african_american": "slight_pos",
            "latino": "mild_neg",
            "asian": "slight_neg",
            "white": "mild_neg",
            "other_race": "slight_neg",
            "evangelical": "mod_neg",
            "catholic": "mild_neg",
            "protestant": "slight_neg",
            "secular": "slight_pos",
            "jewish": "neutral",
            "muslim": "slight_neg",
            "other_rel": "neutral",
            "women": "neutral",
            "men": "mild_neg",
            "other_gender": "slight_pos",
        },
    ),
    (
        "The party turns to Gavin Newsom in 2028; women voters respond modestly favorably "
        "to his reproductive-rights stance while men trend the other way.",
        {
            "african_american": "slight_pos",
            "latino": "slight_neg",
            "asian": "neutral",
            "white": "mild_neg",
            "other_race": "neutral",
            "evangelical": "strong_neg",
            "catholic": "mild_neg",
            "protestant": "slight_neg",
            "secular": "slight_pos",
            "jewish": "slight_pos",
            "muslim": "neutral",
            "other_rel": "neutral",
            "women": "mild_pos",
            "men": "mild_neg",
            "other_gender": "slight_pos",
        },
    ),
    (
        "Gavin Newsom is confirmed as the 2028 Democratic presidential nominee; "
        "Rust Belt white voters react coolly to a second consecutive coastal candidate.",
        {
            "african_american": "neutral",
            "latino": "slight_neg",
            "asian": "slight_neg",
            "white": "mod_neg",
            "other_race": "neutral",
            "evangelical": "mod_neg",
            "catholic": "slight_neg",
            "protestant": "mild_neg",
            "secular": "slight_pos",
            "jewish": "neutral",
            "muslim": "neutral",
            "other_rel": "slight_neg",
            "women": "slight_pos",
            "men": "slight_neg",
            "other_gender": "slight_pos",
        },
    ),
    (
        "After Newsom's 2028 nomination, conservative Catholic and protestant congregations "
        "voice unease, though the reaction is milder than among evangelicals.",
        {
            "african_american": "slight_pos",
            "latino": "slight_neg",
            "asian": "neutral",
            "white": "mild_neg",
            "other_race": "neutral",
            "evangelical": "strong_neg",
            "catholic": "mild_neg",
            "protestant": "mild_neg",
            "secular": "slight_pos",
            "jewish": "neutral",
            "muslim": "slight_neg",
            "other_rel": "neutral",
            "women": "slight_pos",
            "men": "slight_neg",
            "other_gender": "slight_pos",
        },
    ),
    (
        "Newsom heads the 2028 Democratic ticket; his nomination galvanizes urban secular "
        "voters while white exurban turnout enthusiasm falls.",
        {
            "african_american": "slight_pos",
            "latino": "slight_neg",
            "asian": "neutral",
            "white": "mild_neg",
            "other_race": "neutral",
            "evangelical": "mod_neg",
            "catholic": "slight_neg",
            "protestant": "slight_neg",
            "secular": "mild_pos",
            "jewish": "slight_pos",
            "muslim": "neutral",
            "other_rel": "neutral",
            "women": "slight_pos",
            "men": "slight_neg",
            "other_gender": "mild_pos",
        },
    ),
    (
        "The Democrats nominate Gavin Newsom in 2028; his California crime record becomes a "
        "central Republican attack line aimed at Latino and Asian swing voters.",
        {
            "african_american": "slight_pos",
            "latino": "mild_neg",
            "asian": "mild_neg",
            "white": "mod_neg",
            "other_race": "slight_neg",
            "evangelical": "strong_neg",
            "catholic": "mild_neg",
            "protestant": "slight_neg",
            "secular": "slight_pos",
            "jewish": "neutral",
            "muslim": "slight_neg",
            "other_rel": "neutral",
            "women": "neutral",
            "men": "mild_neg",
            "other_gender": "slight_pos",
        },
    ),
    (
        "Gavin Newsom wins the 2028 Democratic nod; affluent suburban women and secular "
        "professionals warm to him while blue-collar men drift away.",
        {
            "african_american": "slight_pos",
            "latino": "slight_neg",
            "asian": "slight_neg",
            "white": "mild_neg",
            "other_race": "neutral",
            "evangelical": "mod_neg",
            "catholic": "slight_neg",
            "protestant": "slight_neg",
            "secular": "mild_pos",
            "jewish": "slight_pos",
            "muslim": "neutral",
            "other_rel": "neutral",
            "women": "mild_pos",
            "men": "mild_neg",
            "other_gender": "slight_pos",
        },
    ),
    (
        "With Newsom as the 2028 nominee, evangelical turnout against him surges while "
        "African American loyalty to the Democratic ticket holds firm.",
        {
            "african_american": "mild_pos",
            "latino": "slight_neg",
            "asian": "neutral",
            "white": "mod_neg",
            "other_race": "neutral",
            "evangelical": "strong_neg",
            "catholic": "mild_neg",
            "protestant": "mild_neg",
            "secular": "slight_pos",
            "jewish": "neutral",
            "muslim": "slight_neg",
            "other_rel": "neutral",
            "women": "slight_pos",
            "men": "slight_neg",
            "other_gender": "slight_pos",
        },
    ),
    (
        "Newsom secures the 2028 Democratic nomination on a strong fundraising base; the "
        "white working class remains his most resistant constituency.",
        {
            "african_american": "slight_pos",
            "latino": "slight_neg",
            "asian": "slight_neg",
            "white": "mod_neg",
            "other_race": "slight_neg",
            "evangelical": "mod_neg",
            "catholic": "mild_neg",
            "protestant": "slight_neg",
            "secular": "slight_pos",
            "jewish": "slight_pos",
            "muslim": "neutral",
            "other_rel": "neutral",
            "women": "slight_pos",
            "men": "mild_neg",
            "other_gender": "slight_pos",
        },
    ),
    (
        "Gavin Newsom is chosen as the Democratic presidential nominee for 2028; reaction "
        "splits sharply along religious and class lines across the electorate.",
        {
            "african_american": "slight_pos",
            "latino": "mild_neg",
            "asian": "slight_neg",
            "white": "mild_neg",
            "other_race": "neutral",
            "evangelical": "strong_neg",
            "catholic": "mild_neg",
            "protestant": "mild_neg",
            "secular": "mild_pos",
            "jewish": "slight_pos",
            "muslim": "slight_neg",
            "other_rel": "neutral",
            "women": "slight_pos",
            "men": "slight_neg",
            "other_gender": "slight_pos",
        },
    ),
]

_ALL_BLOCS = list(CANONICAL_RACES) + list(CANONICAL_RELIGIONS) + list(CANONICAL_GENDERS)


def _validate(idx: int, bins: dict[str, str]) -> None:
    keys = set(bins)
    expected = set(_ALL_BLOCS)
    if keys != expected:
        raise ValueError(
            f"record {idx}: bloc key mismatch. "
            f"missing={expected - keys} extra={keys - expected}"
        )
    for bloc, tok in bins.items():
        if tok not in DELTA_BINS:
            raise ValueError(f"record {idx}: bad bin {tok!r} for {bloc}")


def main() -> None:
    if len(RECORDS) < 15:
        raise SystemExit(f"need 15-20 records, have {len(RECORDS)}")
    now = datetime.now(timezone.utc).isoformat()
    rows: list[dict] = []
    for i, (desc, bins) in enumerate(RECORDS, start=1):
        _validate(i, bins)
        # Order keys race → religion → gender for readability.
        ordered = {b: bins[b] for b in _ALL_BLOCS}
        rows.append(
            {
                "id": f"synthetic_{SHOCK_ID}_{i:02d}",
                "description": desc,
                "party": "democrat",
                "delta_bins": ordered,
                "synthetic": True,
                "source": "hand_authored_prior",
                "mmd_weight": 0.5,
                "weight": 0.5,
                "collected_at": now,
            }
        )

    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    with open(NEWSOM_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {NEWSOM_PATH}")

    # Merge into synthetic.jsonl idempotently: drop any prior rows for this
    # shock_id, then append the freshly generated set (safe to re-run).
    existing: list[str] = []
    if SYNTHETIC_PATH.exists():
        prefix = f"synthetic_{SHOCK_ID}_"
        for line in open(SYNTHETIC_PATH, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                rid = json.loads(line).get("id", "")
            except json.JSONDecodeError:
                existing.append(line)  # preserve anything we can't parse
                continue
            if not rid.startswith(prefix):
                existing.append(line)
    before = len(existing)
    with open(SYNTHETIC_PATH, "w", encoding="utf-8") as f:
        for line in existing:
            f.write(line + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    after = before + len(rows)
    print(f"merged into {SYNTHETIC_PATH}: {before} (non-newsom) + {len(rows)} = {after} rows")


if __name__ == "__main__":
    main()
