# Moment Inspection Report — Electoral Equilibrium

Generated : 2026-06-02 00:46
Config    : configs/base.json
Party     : republican
Panel     : 234 rows
Threshold : ±5% (WARN) / ±10% (ALERT)

==============================================================================
## (i) μ^(P) — Bloc Mean Vote Shares vs. Espinosa / NEP Benchmarks
==============================================================================

Party: REPUBLICAN   |   Winning cycles: [1952, 1956, 1968, 1972, 1980, 1984, 1988, 2004, 2024]

Bloc                    Stratum    Estimated  Benchmark    Delta  Flag
------------------------------------------------------------------------------
  — RACE —
  african_american        RACE          0.1436     0.1300  +0.0136  ok   
  latino                  RACE          0.4727     0.3800  +0.0927  WARN 
  asian                   RACE          0.6105     0.3400  +0.2705  ALERT
  white                   RACE          0.6287     0.6100  +0.0187  ok   
  other_race              RACE          0.3607     0.4200  -0.0593  WARN 
  — RELIGION —
  evangelical             RELIGION      0.7946     0.7700  +0.0246  ok   
  catholic                RELIGION      0.5352     0.4800  +0.0552  WARN 
  protestant              RELIGION      0.6336     0.5400  +0.0936  WARN 
  secular                 RELIGION      0.3206     0.3200  +0.0006  ok   
  jewish                  RELIGION      0.2875     0.2800  +0.0075  ok   
  muslim                  RELIGION      0.2391     0.2000  +0.0391  ok   
  other_rel               RELIGION      0.4865     0.4200  +0.0665  WARN 
  — GENDER —
  women                   GENDER        0.5570     0.4600  +0.0970  WARN 
  men                     GENDER        0.5998     0.5700  +0.0298  ok   
  other_gender            GENDER        0.2400     0.2200  +0.0200  ok   
------------------------------------------------------------------------------

Benchmark sources:
  Race    — National Election Pool (NEP) exit-poll averages 2000–2024
            (ESPINOSA.md §Q1.1–Q1.3 cross-reference)
  Religion— Pew NPORS + CES cumulative averages
            (Espinosa: evangelical ≈ 24% of electorate, Dem share ≈ 23%)
  Gender  — NEP exit-poll averages 2000–2024

==============================================================================
## (ii) Discrepancy Flags — Supervision Check-In Items
==============================================================================

  Flag    Bloc                    Estimated  Benchmark     Delta  Action
  ------------------------------------------------------------------------------
  ALERT   asian                      0.6105     0.3400  + 0.2705  verify data sources
  WARN    women                      0.5570     0.4600  + 0.0970  review coverage
  WARN    protestant                 0.6336     0.5400  + 0.0936  review coverage
  WARN    latino                     0.4727     0.3800  + 0.0927  review coverage
  WARN    other_rel                  0.4865     0.4200  + 0.0665  review coverage
  WARN    other_race                 0.3607     0.4200  -0.0593  review coverage
  WARN    catholic                   0.5352     0.4800  + 0.0552  review coverage

==============================================================================
## (iii) Σ — 5×5 Race-Bloc Covariance Matrix
==============================================================================

Estimated over ALL available cycles (not winning-cycles only).
ddof=1 (unbiased); psd_repair applied if min eigenvalue < 0.

                      african_american      latino       asian       white  other_race
  ------------------------------------------------------------------------------
  african_american        0.010744    0.001813   -0.002347    0.000302   -0.017224
  latino                  0.001813    0.011202    0.011169    0.003587    0.001813
  asian                  -0.002347    0.011169    0.060167    0.005351    0.018240
  white                   0.000302    0.003587    0.005351    0.005385    0.002909
  other_race             -0.017224    0.001813    0.018240    0.002909    0.054320
  ------------------------------------------------------------------------------
  Eigenvalues : [0.003552 0.004031 0.010177 0.044062 0.079995]
  Condition # : 22.52  (well-conditioned)
  Min eigenval: 3.55e-03  (PSD ok)

==============================================================================
## (iv) Notes for Prof. Espinosa
==============================================================================

Key questions raised by this inspection (see ESPINOSA.md for full context):

  Q1.1  ANES 2020 has fully null weight column → equal-weighted; white Dem
        estimate may be ~3 pp above NEP gold standard.
  Q1.2  Systematic ANES–NEP gap for white voters (~4-5 pp); merged panel
        moderates but does not eliminate it.
  Q1.3  CES 2024 african_american 81.9% vs NEP 86%; merged ≈ 83.8%.
  Q2.2  λ₁/λ₂/λ₃ are placeholder (50/30/20) — not empirically calibrated.
  Q2.3  V_eq = 0.535 hardcoded; not yet derived from winning-cycle average.

Winning cycles used for μ estimation:
  [1952, 1956, 1968, 1972, 1980, 1984, 1988, 2004, 2024]

Any ALERT-flagged blocs above should be discussed at the next check-in.
