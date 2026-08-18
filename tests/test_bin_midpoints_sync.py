"""Drift guard for the delta-bin midpoint scale (Phase 5, Step 5.2).

WHY THIS EXISTS: `BIN_MIDPOINTS` was duplicated into three Python modules and
one TypeScript test. Step 2.1 rescaled the canonical copy from an ungrounded
±0.12/±0.15 ceiling to a panel-grounded ±0.03 and *deliberately left the other
two Python copies stale*, flagging them as a known trap; Step 5.1 hand-synced
them; Step 5.2 deleted them outright in favour of imports. That is two full
rescales during which a decode-at-the-wrong-scale bug was live in at least one
code path. This test makes a third occurrence fail loudly instead of silently.

It guards three distinct failure modes, because they fail differently:

  1. RE-DUPLICATION — someone adds a fresh local `BIN_MIDPOINTS = {...}` to a
     module instead of importing. Caught by AST-scanning the whole tree rather
     than checking a hardcoded module list, because a hardcoded list cannot
     catch a copy in a module nobody thought to list.
  2. SILENT REBINDING — a module imports the canonical dict but then mutates or
     shadows it, so the values diverge at runtime even with no local literal.
  3. CROSS-LANGUAGE DRIFT — the frontend cannot import Python. Its own guard
     (webapp/lib/__tests__/deltaScale.test.ts) hardcodes a comparison copy; this
     test parses that copy and compares it to the Python canonical source, so
     the two guards close the loop on each other rather than each trusting a
     hand-copied constant.

DELIBERATELY NOT GUARDED: `electoral.nlp.elasticity._BIN_THRESHOLDS` contains
literals (0.15 / 0.30 / 0.50) that look exactly like the old pre-Step-2.1 delta
scale, and a naive "grep the tree for 0.15" tripwire would flag it. It is NOT
stale: it partitions the RoBERTa *sentiment score* domain [-1, 1], mapping a
continuous score to a bin LABEL, which `BIN_MIDPOINTS` then maps to a delta.
The two tables are on different scales by design. Rescaling the delta bins
changes what a given sentiment score implies in vote-share terms — the intended
effect — without moving the sentiment cutpoints. See that module's comment.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from electoral.core.types import BIN_MIDPOINTS, DELTA_BINS

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_MODULE = REPO_ROOT / "electoral" / "core" / "types.py"
FRONTEND_GUARD = REPO_ROOT / "webapp" / "lib" / "__tests__" / "deltaScale.test.ts"

# The Step 2.1 panel-grounded scale, hardcoded on purpose. This mirrors what
# deltaScale.test.ts does on the frontend side: if someone rescales the
# canonical source again, that is a deliberate act which must also update this
# literal, rather than something a green test suite quietly ratifies.
EXPECTED_CANONICAL: dict[str, float] = {
    "strong_neg": -0.0300,
    "mod_neg": -0.0175,
    "mild_neg": -0.00875,
    "slight_neg": -0.00300,
    "neutral": 0.000,
    "slight_pos": +0.00300,
    "mild_pos": +0.00875,
    "mod_pos": +0.0175,
    "strong_pos": +0.0300,
}

# Directories with no first-party source worth scanning.
_SKIP_DIRS = {".venv", "node_modules", ".next", "__pycache__", ".git", "build", "dist"}


def _python_files() -> list[Path]:
    return [p for p in REPO_ROOT.rglob("*.py") if not any(part in _SKIP_DIRS for part in p.parts)]


def _defines_bin_midpoints_literal(path: Path) -> bool:
    """True if this file assigns a dict/collection literal to BIN_MIDPOINTS.

    An `import`ed name is an ast.ImportFrom, never an ast.Assign, so importing
    modules are correctly not flagged. Annotated assignments (`X: dict = {...}`)
    are checked too, since the canonical definition uses that form.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return False

    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if node.value is None or not isinstance(node.value, (ast.Dict, ast.Call)):
            continue
        for t in targets:
            if isinstance(t, ast.Name) and t.id == "BIN_MIDPOINTS":
                return True
    return False


# ── 1. Re-duplication guard ───────────────────────────────────────────────────


def test_exactly_one_bin_midpoints_definition_in_the_tree():
    """Only electoral/core/types.py may DEFINE BIN_MIDPOINTS; everyone imports."""
    definers = [p for p in _python_files() if _defines_bin_midpoints_literal(p)]
    # This test file hardcodes EXPECTED_CANONICAL, not BIN_MIDPOINTS, so it
    # should never appear here -- if it does, the AST check is over-matching.
    relative = sorted(str(p.relative_to(REPO_ROOT)) for p in definers)

    assert relative == ["electoral/core/types.py"], (
        "BIN_MIDPOINTS must be defined in exactly one place "
        "(electoral/core/types.py) and imported everywhere else.\n"
        f"Found definitions in: {relative}\n"
        "A local copy is how this drifted through two separate rescales "
        "(Step 2.1 left copies stale; Step 5.1 hand-synced them; Step 5.2 "
        "replaced them with imports). Import it instead of redefining it."
    )


# ── 2. Silent-rebinding guard ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "module_path",
    ["electoral.nlp.elasticity", "electoral.llm.eval", "electoral.llm.inference"],
)
def test_importing_modules_expose_the_canonical_object(module_path: str):
    """Modules re-exporting BIN_MIDPOINTS must expose the canonical object itself."""
    import importlib

    mod = importlib.import_module(module_path)
    if not hasattr(mod, "BIN_MIDPOINTS"):
        pytest.skip(f"{module_path} does not re-export BIN_MIDPOINTS")
    assert mod.BIN_MIDPOINTS is BIN_MIDPOINTS, (
        f"{module_path}.BIN_MIDPOINTS is not the canonical object from "
        "electoral.core.types -- it has been shadowed, copied, or rebound."
    )


def test_generate_synthetic_script_uses_canonical_object():
    """scripts/ is not a package; load it by path and check the same property."""
    import importlib.util

    path = REPO_ROOT / "scripts" / "generate_synthetic.py"
    spec = importlib.util.spec_from_file_location("_gs_under_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.BIN_MIDPOINTS is BIN_MIDPOINTS, (
        "scripts/generate_synthetic.py no longer uses the canonical " "BIN_MIDPOINTS object."
    )


# ── 3. Canonical-value and internal-consistency guards ────────────────────────


def test_canonical_values_match_the_panel_grounded_scale():
    assert BIN_MIDPOINTS == EXPECTED_CANONICAL, (
        "electoral.core.types.BIN_MIDPOINTS no longer matches the Step 2.1 "
        "panel-grounded ±0.03 scale. If this rescale is intentional, update "
        "EXPECTED_CANONICAL here AND the frontend copy in "
        "webapp/lib/__tests__/deltaScale.test.ts AND re-check every constant "
        "carrying a 'rescaled Step 2.1' comment (clips, noise std, neutral "
        "thresholds) -- those were all derived from this scale."
    )


def test_midpoints_cover_exactly_the_declared_bins():
    assert set(BIN_MIDPOINTS) == set(DELTA_BINS)


def test_midpoints_are_strictly_increasing_in_declared_bin_order():
    values = [BIN_MIDPOINTS[b] for b in DELTA_BINS]
    assert values == sorted(values), "DELTA_BINS order must be monotonic in midpoint"
    assert len(set(values)) == len(values), "midpoints must be distinct"


def test_scale_is_symmetric_about_neutral():
    assert BIN_MIDPOINTS["neutral"] == 0.0
    for neg, pos in [
        ("strong_neg", "strong_pos"),
        ("mod_neg", "mod_pos"),
        ("mild_neg", "mild_pos"),
        ("slight_neg", "slight_pos"),
    ]:
        assert BIN_MIDPOINTS[neg] == pytest.approx(-BIN_MIDPOINTS[pos])


# ── 4. Cross-language drift guard ─────────────────────────────────────────────


def test_frontend_guard_copy_matches_python_canonical():
    """The TS guard hardcodes its own copy; verify it against Python.

    The frontend cannot import Python, so deltaScale.test.ts necessarily
    hand-copies these values. That copy is exactly the kind of thing that
    drifts, so parse and compare it here rather than trusting it.
    """
    if not FRONTEND_GUARD.exists():
        pytest.skip(f"frontend guard not present at {FRONTEND_GUARD}")

    src = FRONTEND_GUARD.read_text(encoding="utf-8")
    match = re.search(
        r"const BIN_MIDPOINTS:\s*Record<string,\s*number>\s*=\s*\{(.*?)\}\s*;",
        src,
        re.DOTALL,
    )
    assert match, (
        "Could not locate the BIN_MIDPOINTS literal in "
        f"{FRONTEND_GUARD.relative_to(REPO_ROOT)} -- if it was renamed or "
        "restructured, update this parser so the cross-language guard keeps working."
    )

    body = match.group(1)
    # Tolerate trailing commas and unquoted keys; reject anything non-numeric.
    parsed: dict[str, float] = {}
    for key, raw in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(-?[0-9.]+)", body):
        parsed[key] = float(raw)

    assert parsed, "parsed no entries from the frontend BIN_MIDPOINTS literal"
    assert parsed == pytest.approx(BIN_MIDPOINTS), (
        "Frontend and backend delta scales have drifted.\n"
        f"  frontend ({FRONTEND_GUARD.relative_to(REPO_ROOT)}): {json.dumps(parsed, sort_keys=True)}\n"
        f"  backend  (electoral/core/types.py):                 {json.dumps(BIN_MIDPOINTS, sort_keys=True)}"
    )
