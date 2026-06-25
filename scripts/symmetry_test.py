"""
symmetry_test.py — Probe the deployed model for partisan symmetry.

For a model that bills itself as "bipartisan," symmetric events (the same shock
applied to each party) should produce mirror-image effects — same magnitude,
opposite or coalition-appropriate direction. Systematic asymmetry (one party
consistently punished harder across matched pairs) is evidence of directional
bias and must be surfaced, not hidden.

This hits the LIVE Modal endpoint and tabulates deltas side by side.
Run: python scripts/symmetry_test.py
"""
from __future__ import annotations
import json
import sys
import time
import urllib.parse
import urllib.request
from statistics import mean

ENDPOINT = "https://juhidamley--electoral-equilibrium-serve.modal.run/estimate/stream"

# Matched event pairs. Each pair is the SAME shock type, framed for each party,
# naming the affected candidate explicitly so the party toggle and the text agree.
# Spread across event TYPES so we're not testing a single scenario.
PAIRS = [
    ("A major financial scandal engulfs the Democratic presidential candidate",
     "A major financial scandal engulfs the Republican presidential candidate"),
    ("The Democratic presidential candidate delivers a widely praised convention speech",
     "The Republican presidential candidate delivers a widely praised convention speech"),
    ("The Democratic candidate is endorsed by a popular former president",
     "The Republican candidate is endorsed by a popular former president"),
    ("A sympathetic personal story about the Democratic candidate goes viral",
     "A sympathetic personal story about the Republican candidate goes viral"),
    ("The Democratic candidate performs poorly in a nationally televised debate",
     "The Republican candidate performs poorly in a nationally televised debate"),
    ("A respected newspaper endorses the Democratic candidate",
     "A respected newspaper endorses the Republican candidate"),
    ("The Democratic candidate faces credible allegations of corruption",
     "The Republican candidate faces credible allegations of corruption"),
    ("The Democratic candidate proposes a popular middle-class tax cut",
     "The Republican candidate proposes a popular middle-class tax cut"),
]

ALL_BLOCS_FIELDS = ("deltas_race", "deltas_religion", "deltas_gender")


def fetch_deltas(event: str, party: str, seed: int = 42) -> dict | None:
    """Hit the SSE endpoint, return the parsed 'deltas' frame, or None on failure."""
    qs = urllib.parse.urlencode({"event": event, "intensity": 1.0, "party": party})
    url = f"{ENDPOINT}?{qs}"
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            current_event = None
            for raw in resp:
                line = raw.decode("utf-8").rstrip("\n")
                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip()
                elif line.startswith("data:") and current_event == "deltas":
                    return json.loads(line.split(":", 1)[1].strip())
    except Exception as e:  # noqa: BLE001
        print(f"  [error] {party}: {e}", file=sys.stderr)
    return None


def mean_delta(frame: dict) -> float:
    """Average delta across all 15 blocs — the aggregate directional effect."""
    vals = []
    for f in ALL_BLOCS_FIELDS:
        vals.extend(frame.get(f, {}).values())
    return mean(vals) if vals else float("nan")


def main() -> None:
    print("=" * 78)
    print("SYMMETRY TEST — matched event pairs against the live model")
    print("=" * 78)
    print(f"{'event type':<42} {'D-mean':>8} {'R-mean':>8} {'asym':>8}")
    print("-" * 78)

    asymmetries = []
    for dem_event, rep_event in PAIRS:
        dem = fetch_deltas(dem_event, "democrat")
        time.sleep(1)
        rep = fetch_deltas(rep_event, "republican")
        time.sleep(1)
        if dem is None or rep is None:
            print(f"{dem_event[:40]:<42} {'FAILED':>8}")
            continue
        d_mean = mean_delta(dem)
        r_mean = mean_delta(rep)
        # Symmetric model: a Dem-helping event for the Dem ≈ a Rep-helping event
        # for the Rep. So d_mean and r_mean should be CLOSE for mirrored framings.
        # asym > 0 means the Republican framing came out more negative.
        asym = d_mean - r_mean
        asymmetries.append(asym)
        label = dem_event.split()[2:5]  # rough type tag
        print(f"{' '.join(label)[:40]:<42} {d_mean:>+8.4f} {r_mean:>+8.4f} {asym:>+8.4f}")

    print("-" * 78)
    if asymmetries:
        m = mean(asymmetries)
        print(f"{'MEAN ASYMMETRY (D-mean minus R-mean)':<42} {'':>8} {'':>8} {m:>+8.4f}")
        print()
        print("INTERPRETATION:")
        print("  asym ≈ 0      → symmetric: parties treated equivalently.")
        print("  asym > +0.01  → Republican framings systematically more negative")
        print("                  (model leans Democratic).")
        print("  asym < -0.01  → Democratic framings systematically more negative")
        print("                  (model leans Republican).")
        print()
        print("  NOTE: per-pair asymmetry is expected (coalitions differ in shape).")
        print("  The MEAN across many pairs is the bias signal. A single event")
        print("  proves nothing; the aggregate across types is what matters.")
    else:
        print("No successful pairs — check the endpoint is live.")


if __name__ == "__main__":
    main()
