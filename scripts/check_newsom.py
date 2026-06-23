"""Directional validation of the retrained adapter on the Newsom 2028 event.

PASS: the bug was every race bloc (esp. white, latino) showing POSITIVE.
A fix shows white and/or latino at neutral or NEGATIVE, politically sensible.
"""
import logging
logging.basicConfig(level=logging.INFO)

from electoral.llm.inference import load_model, predict_delta_bins
from electoral.core.types import CANONICAL_RACES

NEG_BINS = {"strong_neg", "mod_neg", "mild_neg", "slight_neg"}
POS_BINS = {"slight_pos", "mild_pos", "mod_pos", "strong_pos"}

NEWSOM_EVENT = (
    "Gavin Newsom secures the Democratic presidential nomination amid debate "
    "over his California progressive record and electability in swing states"
)

def sign(b):
    if b in NEG_BINS: return "neg"
    if b in POS_BINS: return "pos"
    return "neutral"

def main():
    print("Loading base Mistral + retrained adapter...")
    model, tokenizer = load_model(adapter_path="models/mistral-r16-v2")
    print(f"\nEvent: {NEWSOM_EVENT}\nParty: democrat\n")
    bins = predict_delta_bins(
        shock_text=NEWSOM_EVENT, party="democrat",
        model=model, tokenizer=tokenizer,
    )
    print("Race-bloc predictions:")
    signs = {}
    for bloc in CANONICAL_RACES:
        b = bins.get(bloc, "?")
        signs[bloc] = sign(b)
        print(f"  {bloc:18s} {b:12s} ({signs[bloc]})")

    all_pos = all(signs.get(b) == "pos" for b in CANONICAL_RACES)
    white, latino = signs.get("white"), signs.get("latino")
    print("\n--- VERDICT ---")
    if all_pos:
        print("FAIL: every race bloc positive — directional bug PERSISTS.")
        print("647-record human review becomes blocking.")
    elif white == "pos" and latino == "pos":
        print("PARTIAL: white and latino still positive — weak fix.")
    else:
        print(f"PASS (directional): white={white}, latino={latino}.")
        print("Sanity-check the full vector before trusting it.")

if __name__ == "__main__":
    main()
