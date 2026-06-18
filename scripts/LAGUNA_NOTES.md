# USC Laguna HPC Notes

## Access
- SSH: JDamley28@cmc.edu@laguna.carc.usc.edu
- Login node: laguna1
- Tailscale not available on compute nodes

## Storage
- Home: /home1/JDamley28@cmc.edu/ (limited quota)
- Scratch: /scratch/JDamley28@cmc.edu/ (primary compute storage)
- Project data: /scratch/JDamley28@cmc.edu/electoralData/
- JUHIDRIVE mirror: rsync from /Volumes/JUHIDRIVE/electoralData/

## SLURM
- Partition: compute (all jobs use this)
- Conda: /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
- Env: conda activate electoral
- Home path: /home1/JDamley28@cmc.edu/ (NOT /home/)
- Log dir: /scratch/JDamley28@cmc.edu/electoralData/logs/
- Max array concurrency used: %50 (safe limit)
- Typical node names: c22-xx, c23-xx

## Confirmed working jobs
- sample_archives_laguna.slurm — stratified sampling array job
- filter_reddit_monthly_laguna.slurm — bz2 keyword filter array job
- discord_find_servers_laguna.slurm — 110GB zstd tar streaming
- discord_extract.slurm — single-pass server extraction
- discord_sample.slurm — date/content filtering
- clean_with_llm_laguna.slurm — async Gemini 2.5 Flash Lite cleaning
- score_array_laguna.slurm — RoBERTa scoring array job

## Notes
- /usr/bin/pzstd not available; use ~/.conda/envs/electoral/bin/zstd
- No Syncthing on compute nodes — use rsync for data transfer
- CMC Hopper HPC deprioritized due to quota limits; used for hosting only
- GPU nodes: not yet confirmed — check with sinfo before submitting 
  GPU jobs for Mistral fine-tuning

## Last updated: 2026-06-10
