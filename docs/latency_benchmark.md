# Latency Benchmark — ShockEstimator.estimate()

## Setup
- Model: mistralai/Mistral-7B-v0.3 + LoRA r16 adapter
- Hardware: NVIDIA L40S (46GB VRAM), CMC Hopper HPC
- Precision: float16
- Constrained decoding: outlines 0.0.37 with ShockResponseSchema

## Results (3 sequential calls after warmup)
| Call | Latency |
|------|---------|
| 1    | 11.40s  |
| 2    | 12.16s  |
| 3    | 11.90s  |

Mean: 11.82s  Median: 11.90s  p95: ~12.2s (estimated)

## Analysis
The p95 latency of ~12s exceeds the 3-second target. The bottleneck
is outlines FSM-based constrained decoding, which processes one token
at a time with regex constraint checking. The 15-bloc × 4-key JSON
output schema requires ~200 constrained tokens per call.

## Decision
Accepted for SRP batch inference — all 111 eval examples processed
overnight as a SLURM job (~20 minutes total). Interactive latency
target of <3s deferred to post-SRP optimization via vLLM structured
outputs or smaller distilled model.

## Next steps
- vLLM with guided_json decoding: expected 2-4s per call
- Distilled Mistral 3B: expected 4-6s per call
- Both deferred to post-SRP

## Last updated: 2026-06-11
