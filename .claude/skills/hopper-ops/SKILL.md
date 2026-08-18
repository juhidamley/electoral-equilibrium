---
name: hopper-ops
description: Run commands on the CMC Hopper HPC cluster (host hopper.mckenna.edu, user jdamley28) over SSH for the Electoral Equilibrium project — training runs, GPU inference, evaluation jobs, and file transfer. Use this whenever the user asks to run, launch, check, or fetch something on Hopper, the cluster, "the HPC", or a GPU/SLURM job, or to move files between the Mac and Hopper. Do NOT use it for local-only work, for anything requiring the user's password (SSH key auth must be set up first), or to run destructive/irreversible cluster operations without explicit confirmation.
---

# Hopper HPC operations (Electoral Equilibrium)

Drive the CMC Hopper cluster over SSH to run training, inference, and eval jobs and to
move files. You (Claude Code) run on the user's Mac; you reach Hopper by wrapping
commands in `ssh`. You encode the *workflow*; the user owns the *credentials*.

- **Host / user:** `jdamley28@hopper.mckenna.edu` — verified working 2026-08-18. Do
  **not** "correct" this to `hopper.hpc.cmc.edu`, which appears in older repo docs and
  does not resolve at all (`nodename nor servname provided`).
- **Repo on Hopper:** `~/electoral-equilibrium`
- **Conda env:** `electoral`
- **Login node is CPU-only.** Real compute needs an interactive GPU session or a
  SLURM batch job (see below). Never run training/inference on the login node.

## Auth — set up ONCE by the user, never automated by this skill

This skill assumes **passwordless SSH key auth** is already working. It does NOT
handle passwords, does NOT store secrets, and does NOT automate login.

Preflight check (run first; if it fails, stop and give the user the setup steps):
```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 jdamley28@hopper.mckenna.edu 'echo OK' 2>&1
```
- Prints `OK` → key auth works; proceed.
- Prompts for a password / times out / `Permission denied` → STOP. Tell the user to
  set up a key themselves (do NOT try to do it for them):
  ```
  ssh-keygen -t ed25519            # if they don't have a key
  ssh-copy-id jdamley28@hopper.mckenna.edu
  ```
  Optionally add a `~/.ssh/config` Host alias `hopper` for brevity. Then re-run the
  preflight. Never pass a password on the command line; never echo secrets.

Use `-o BatchMode=yes` on non-interactive calls so a missing key fails fast instead
of hanging on a password prompt.

## The command wrapper

Run a Hopper command:
```bash
ssh jdamley28@hopper.mckenna.edu 'cd ~/electoral-equilibrium && <command>'
```
For anything needing the conda env, prefix with the env activation (login shells on
Hopper may not auto-activate):
```bash
ssh jdamley28@hopper.mckenna.edu 'source ~/.bashrc; conda activate electoral && cd ~/electoral-equilibrium && <command>'
```
Quote carefully — the whole remote command is a single shell string. For multi-line
scripts, prefer a heredoc written to a temp file on Hopper, then execute it, rather
than fighting nested quoting.

## PROJECT-SPECIFIC GOTCHAS (hard-won — these cost hours; honor them)

1. **SLURM resources are fixed and non-obvious.** Verified against live `sinfo`
   2026-08-18:
   - `partition=main`, `gres=gpu:l40s:1`. Every GPU node is `gpu:l40s:4` (4 per node,
     128 CPUs, 750G RAM), nodes `gpu[01-15]`, plus 2 GPU-less `himem` nodes.
   - There is **NO `gpu` partition and NO A100s** — jobs requesting them die. The only
     GRES on the whole cluster is `gpu:l40s:4`.
   - **The trap that keeps causing this:** the GPU nodes are *named* `gpu01`–`gpu15`,
     and the GRES spec starts with `gpu:`. So "gpu" is everywhere — but it is never a
     partition name. `--gres=gpu:l40s:1` ✅ / `--partition=gpu` ❌.

   **Partitions (pick deliberately — the default is 3h, not your job's length):**

   | Partition | Time limit | Default time | GPU nodes | Use for |
   |---|---|---|---|---|
   | `main` (default) | 3 days | 3 h | 15 | normal training / eval |
   | `debug` | **1 h** | 1 h | 15 | quick checks, smoke runs |
   | `nsf_main` | 3 days | 3 h | 15 | — |
   | `long` | **8 days** | 3 h | 2 | long training runs |

   Always pass `--time` explicitly; the 3-hour default will kill a longer run.
   - `module load cuda/13.2` (NOT 12.4 — bitsandbytes needs libnvJitLink.so.13;
     cuda/12.4 fails the load). Verified `module avail` 2026-08-18: the ONLY cuda
     modules are `cuda/12.4` and `cuda/13.2` (13.2 is the default), and the ONLY
     python module is `python/3.13.14`. Older scripts loaded `python/3.11 cuda/12.1`
     — neither exists, so with `|| true` they silently loaded **nothing**. If a job
     mysteriously has no CUDA, check the module names first.
   A correct interactive GPU session:
   ```bash
   srun --partition=main --gres=gpu:l40s:1 --time=00:30:00 --pty bash
   # then on the node:
   module load cuda/13.2
   conda activate electoral
   ```
   (Interactive `srun --pty` needs a real TTY — run it in a session where the user
   can see/attach, or use `sbatch` for unattended jobs. For fire-and-forget work,
   write an sbatch script and submit it rather than holding an interactive node.)

2. **Inference runs fine WITHOUT `--quantized`.** The bitsandbytes `libnvJitLink`
   error is harmless *noise* once that flag is dropped; full-precision Mistral-7B
   loads fine on an L40S (46GB). Do not add `--quantized` to inference commands, and
   do not treat the bitsandbytes stderr line as a failure.

3. **Generated data does NOT travel with git.** Splits, ground-truth JSONs, the
   held-out probe, and trained adapters are gitignored / not committed. To use them
   on Hopper you must scp them from the Mac and hash-verify. Never assume a data file
   is present just because the code references it — check first (`ls`, `sha256sum`).

4. **Never overwrite a trained adapter.** Training scripts have an existence guard;
   if a run would clobber an existing adapter, stop and confirm. New adapter = new
   timestamped directory. Held-out probe data is sacred — never let it into a train
   split.

## Common operations

**Check job / queue state:**
```bash
ssh jdamley28@hopper.mckenna.edu 'squeue -u jdamley28'
ssh jdamley28@hopper.mckenna.edu 'sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode'
```

**Submit a batch job:**
```bash
ssh jdamley28@hopper.mckenna.edu 'cd ~/electoral-equilibrium && sbatch <script>.slurm'
# then poll squeue; tail the log the script writes
```

**Tail a running job's output:**
```bash
ssh jdamley28@hopper.mckenna.edu 'tail -f ~/electoral-equilibrium/<logfile>'
# (tail -f holds the connection open; use a bounded `tail -n 50` for a snapshot)
```

**File transfer (run from the Mac, NOT wrapped in ssh):**
```bash
# Mac → Hopper
scp -r <local_path> jdamley28@hopper.mckenna.edu:~/electoral-equilibrium/<dest>/
# Hopper → Mac
scp -r jdamley28@hopper.mckenna.edu:~/electoral-equilibrium/<remote_path> <local_dest>/
# verify integrity after transfer:
ssh jdamley28@hopper.mckenna.edu 'sha256sum ~/electoral-equilibrium/<file>'
shasum -a 256 <local_file>   # compare
```

**Run an eval / inference script on a GPU node (unattended):** write an sbatch wrapper
that does `module load cuda/13.2`, `conda activate electoral`, then the python call
(no `--quantized`), submit with `sbatch`, poll, and fetch the output artifact back.

## Safety rules

- **Confirm before anything destructive or expensive:** deleting files, `scancel`
  on someone else's assumptions, overwriting adapters/data, launching long
  multi-hour jobs. State what will run and wait for a yes.
- **Never** run a command that would take production down or touch deploy state —
  Hopper is train/eval only; deployment is a separate Modal flow. Do not `modal
  deploy` or edit deploy configs from here.
- **Verify before declaring success.** For a job, check `sacct` ExitCode and the
  actual output artifact exists — not just that the submission returned a job id.
- **Report honestly:** the command run, the real output (including stderr), the job
  state, and whether the expected artifact was produced. Don't infer success from a
  clean submission.
- **Idempotency / connection loss:** SSH can drop. Prefer `sbatch` (survives
  disconnect) over long interactive `srun` for anything over a few minutes. Before
  re-submitting a job after a dropped connection, check `squeue` — it may still be
  running.

## When NOT to use this skill

- Local-only tasks (editing code on the Mac, git, running the webapp locally).
- Anything requiring the user's password (auth must be key-based; if it isn't, stop
  and have them set it up).
- Deployment (that's Modal, a separate flow).
- Destructive cluster ops without explicit confirmation.
