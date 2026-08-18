---
name: bullet-delegate
description: Delegate the writing of code to the Bullet CLI coding agent, then rigorously review its output before accepting it. Use this skill whenever the user asks to implement, build, add, refactor, or fix code in a repository — especially for multi-file or feature-sized tasks — and whenever the user mentions Bullet, "have bullet do it", delegating implementation, or wanting fast implementation with an independent review. Do NOT use it for trivial one-line edits, pure explanations/questions about code, or tasks the user explicitly asks you to code yourself.
---

# Bullet Delegate: implement with Bullet, review with Claude

## Division of labor (the whole point)

- **You (Claude Code) are the architect and the reviewer.** You write the spec, you
  judge the result, you own the final verdict. You do NOT write the feature code.
- **Bullet is the implementer.** It's fast; let it do the typing.
- The one exception: after review, fixes smaller than ~15 lines you make directly
  rather than paying another delegation round trip.

The review is the point of this workflow. Never rubber-stamp: a delegation you
didn't genuinely review is worse than no delegation, because it ships someone
else's unexamined code under your sign-off.

## Bullet CLI interface (verified against v1.4.5 — re-check if calls fail)

```
bullet -p "task"            non-interactive: stream the answer, then exit
bullet -c -p "follow-up"    continue the most recent session (use for feedback rounds)
-C <dir>                    workspace folder (default: cwd)
--level <l>                 simple | moderate | complex | hard | extreme
--effort <e>                low | medium | high | xhigh | max
--no-color --no-anim        clean output for logs
--no-thinking --no-diffs    quieter output
```

Piped stdin is appended to the prompt (`git diff | bullet -p "review this"`).
Bullet is young and evolving fast: **if any invocation errors or behaves
unexpectedly, run `bullet --help` and adapt to what it actually says now**
rather than retrying the same call.

## Step 0 — Preflight (once per session)

1. `bullet --version` (or `which bullet`). If missing, offer to install:
   `npm install -g @trybullet/cli` (needs Node 18+). If the user declines,
   say so and do the task yourself — this skill then doesn't apply.
2. Auth: if a run fails with an auth/login error, do NOT try to automate
   sign-in. Tell the user to run `bullet auth` (connect Claude/ChatGPT/API key)
   or `bullet account guest` once, interactively, then resume.
3. Read the project's CLAUDE.md (if present) for protected paths — files that
   must never be touched (deploy configs, production settings, lockfiles the
   project pins deliberately). List them; they go into every spec's
   out-of-scope section, and any diff touching them is an automatic review FAIL.

## Step 1 — Git safety (before every delegation)

Never let an agent edit an uninspectable tree.

1. Require a clean working tree. If there are uncommitted changes, ask the user:
   commit, stash, or abort. Never delegate on top of someone's WIP.
2. Create a work branch: `git checkout -b bullet/<short-task-slug>`.
   Never run Bullet on main.
3. Record the baseline: `BASE=$(git rev-parse HEAD)`. Every review diffs
   against `$BASE`, and a broken run is recoverable with
   `git reset --hard $BASE`.

## Step 2 — Write the spec

Bullet's output quality is bounded by your spec quality. A vague spec produces
plausible-looking wrong code that costs more to review than writing it yourself.
The spec must contain:

- **Goal** — one paragraph, what done looks like.
- **Files in scope** — the specific files/directories to create or modify.
- **Out of scope** — explicit: protected paths from Step 0, and anything
  adjacent the agent might "helpfully" touch (configs, deps, unrelated cleanup).
  State: "Do not modify any file not listed in scope. Do not add dependencies.
  Do not search or read the repo beyond the files listed — everything you need
  is named here." (Exploration is the biggest token sink; a fully-specified
  file list eliminates it.)
- **Interfaces/contracts** — exact function signatures, API shapes, types the
  new code must satisfy, existing conventions to follow (point at an example
  file in the repo).
- **Acceptance criteria** — the commands that must pass (test suite, lint,
  build) and any specific behaviors to verify. These are the same commands
  you will run in review, so state them precisely.

## Step 3 — Invoke

```bash
bullet -p "<the full spec>" --no-color --no-anim 2>&1 | tee /tmp/bullet_run_1.log
```

- ALWAYS pass `--level` and `--effort` explicitly — never let the auto-router
  decide (it errs toward expensive models and heavy reasoning). Default to
  `--level moderate --effort medium`; drop to `simple`/`low` for mechanical
  changes; reserve `hard`/`high` for genuinely complex work and say why in the
  report. For trivial mechanical tasks, consider pinning a cheap model with
  `-m` if one is configured.
- Set a timeout appropriate to the task (e.g. wrap in `timeout 15m ...`). If it
  hangs: kill it, inspect the partial diff, and either reset to `$BASE` or
  review what landed.
- After the run, immediately check `git status`: **verify Bullet actually
  edited files.** If the tree is unchanged, `-p` answered instead of acting —
  re-read the log; if it produced code in stdout instead of files, re-invoke
  with an explicit instruction to write the files, or check `bullet --help`
  for a changed flag.

## Step 4 — Review (the heart; never skip, never soften)

1. **Read the entire diff**: `git diff $BASE`. All of it. No skimming.
2. **Scope check first**: `git diff $BASE --name-only` — every touched file
   must be in the spec's scope. Any protected path touched → FAIL, reset that
   file, and note it. Unrequested dependency changes (package.json,
   pyproject.toml, lockfiles) → treat as scope violations unless specced.
3. **Correctness walk-through** against the spec: does the logic do what was
   asked? Trace the main path and at least two edge cases by hand. Check error
   handling, off-by-ones, async/await correctness, resource cleanup.
4. **Security pass**: no hardcoded secrets, no injection-prone string
   building (SQL/shell/HTML), no disabled validation, no weakened auth.
5. **Consistency**: matches the repo's existing style, naming, and patterns
   (compare against the example file you cited in the spec).
6. **Tests**: were tests added/updated where the spec asked? Do they actually
   assert behavior, or are they vacuous?
7. **Run the acceptance commands** from the spec — test suite, lint, build.
   Paste the real output into your notes. A review without running the
   verification commands is not a review; never report "passes" from reading
   alone.

## Step 5 — Verdict routing

- **PASS** → commit on the work branch with a message crediting the flow
  (e.g. `feat: <thing> (implemented via bullet, reviewed)`), then report.
  Do not merge to main without the user's say-so.
- **MINOR issues** (each fix < ~15 lines, no design change) → fix them
  yourself directly, list every fix in the report, re-run the acceptance
  commands, then commit.
- **MAJOR issues** (wrong approach, failed tests you can't trivially fix,
  scope violations, security problems) → send targeted feedback in a FRESH
  session with only the diff piped in — do NOT use `-c` (resuming re-pays the
  entire prior session's context; the diff + numbered feedback is a fraction
  of it):
  ```bash
  git diff $BASE | bullet -p "This diff implements: <one-line goal restated>.
  Fix ONLY these issues: <numbered, specific, file:line, what correct looks
  like>. Acceptance: <the commands that must pass>. Touch no other files." \
    --level moderate --effort medium --no-color --no-anim 2>&1 \
    | tee /tmp/bullet_run_2.log
  ```
  Reserve `-c` for the rare case where the fix genuinely requires the full
  session context (e.g. Bullet must recall why it chose an approach) — and
  note in the report that you used it. Then review again from Step 4.
- **Cap: 3 Bullet rounds total.** If it still fails review after round 3,
  stop delegating — either finish the remainder yourself (telling the user
  you did) or escalate to the user with the diff and your findings. Do not
  loop indefinitely; that burns the speed advantage the delegation exists for.

## Step 6 — Report to the user

Always include, honestly:
- What was delegated (the spec, briefly) and how many rounds it took.
- The diff summary (files touched, +/- lines).
- Review findings: what passed, what you fixed yourself (itemized), what
  Bullet fixed on feedback.
- Acceptance results: the actual commands run and their real outcomes.
- Anything you're not sure about — flag it rather than absorbing it silently.
- The branch name and the exact command to merge, left for the user.

## Token economy (delegation is not free tokens)

If Bullet is authed via the user's Claude/ChatGPT subscription, its tokens draw
from the SAME pool as this session — delegation buys wall-clock speed and
division of labor, not budget. Agent loops re-send their whole accumulated
context on every tool call, so cost scales with turns and context, not task
size. Discipline that keeps runs cheap:

- Explicit `--level`/`--effort` on every call (Step 3) — never auto-route.
- Fully-named file scope with the no-exploration clause (Step 2) — repo
  exploration is the largest avoidable sink.
- Fresh session + piped diff for feedback rounds (Step 5) — not `-c`.
- DECOMPOSE large tasks into 2-4 scoped delegations that each start with a
  near-empty context, rather than one mega-run that accumulates everything.
  More runs is often cheaper than one big one.
- If the task is trivial enough that spec-writing + review costs more than the
  code, don't delegate at all (see below).

## When NOT to delegate

- Trivial edits (a rename, a one-liner, a comment) — cheaper to just do.
- Tasks touching protected paths as their primary subject (deploy configs,
  production settings) — do these yourself with full care.
- Pure analysis/explanation — no code is being written; Bullet adds nothing.
- When the user explicitly asks YOU to write it — respect that.

## Failure modes to guard against

- **Rubber-stamping**: the temptation after a clean-looking run is to skim.
  The acceptance commands and the full-diff read are non-negotiable.
- **Spec drift across rounds**: when giving feedback, restate the relevant
  acceptance criteria — don't let round 3 solve a different problem.
- **Silent self-completion**: if you end up writing significant code yourself
  (round-cap hit, Bullet unavailable), say so plainly in the report. Never
  present your own code as delegated-and-reviewed.
- **Stale interface**: Bullet ships fast; a flag that worked last week may
  change. On any CLI error, `bullet --help` first, adapt, and note the change.
