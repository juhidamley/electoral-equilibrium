"""Canonical documented baseline vote-share benchmarks (single source of truth).

These are the Democrat two-party within-bloc vote shares averaged over competitive
cycles, with documented provenance — NOT eyeballed constants. They exist so that
both the offline moment-sanity checker (scripts/inspect_moments.py) and the live
API nominal priors (electoral/api/shock_endpoint.py) reference ONE set of values
instead of keeping divergent copies.

Provenance
----------
Race / gender: National Election Pool (NEP) exit-poll averages over the
  competitive cycles 2000–2024, cross-referenced in ESPINOSA.md §Q1.1–Q1.3.
    african_american: NEP 2016=89%, 2020=87%, 2024=86% → mean ≈ 0.87
    white:            NEP 2016=37%, 2020=41%, 2024=39% → mean ≈ 0.39 (§Q1.2)
    latino:           NEP 2016=66%, 2020=63%, 2024=58% → mean ≈ 0.62
    asian:            AAPI Data 2016=65%, 2020=72%, 2024=60% → mean ≈ 0.66
    other_race:       NEP average ≈ 0.58
    women ≈ 0.54, men ≈ 0.43, other_gender ≈ 0.78 (NEP 2000–2024)
Religion: Pew Research Center NPORS + CES cumulative averages
  (ESPINOSA.md §Q2.4, §Q6.1).

Republican within-bloc shares are taken as 1 − democrat by construction (two-party).
"""

from __future__ import annotations

# Democrat two-party within-bloc vote share, all three strata, documented above.
DEM_BENCHMARKS: dict[str, float] = {
    # Race — NEP exit-poll 2000–2024
    "african_american": 0.87,
    "latino": 0.62,
    "asian": 0.66,
    "white": 0.39,
    "other_race": 0.58,
    # Religion — Pew NPORS + CES cumulative
    "evangelical": 0.23,
    "catholic": 0.52,
    "protestant": 0.46,
    "secular": 0.68,
    "jewish": 0.72,
    "muslim": 0.80,
    "other_rel": 0.58,
    # Gender — NEP 2000–2024
    "women": 0.54,
    "men": 0.43,
    "other_gender": 0.78,
}

# Race-only convenience view (the 5 optimizer race blocs).
DEM_RACE_BENCHMARKS: dict[str, float] = {
    b: DEM_BENCHMARKS[b]
    for b in ("african_american", "asian", "latino", "other_race", "white")
}
