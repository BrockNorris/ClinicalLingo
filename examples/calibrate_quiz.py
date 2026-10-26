"""Calibrate the feedback-loop similarity thresholds against the real embedder.

For a handful of dictionary entries, scores three kinds of student answers
(correct paraphrase, partial, wrong/off-topic) against the entry text, and
prints the distributions so the deeper/related cut-offs can be chosen from
data instead of guessed.  Ollama must be running.

    python examples/calibrate_quiz.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clinicallingo.retrieval.build_index import load_all  # noqa: E402
from clinicallingo.retrieval.embedder import OllamaEmbedder  # noqa: E402

CASES = {
    "hypoechoic": {
        "correct": ["It looks darker than the tissue around it because fewer echoes come back",
                    "Darker than the surrounding tissue, but not black",
                    "An area that reflects less sound than its neighbours so it appears dark"],
        "partial": ["It is dark on the screen", "Something about echoes being lower"],
        "wrong": ["something to do with blood flow", "It means the probe frequency is too high", "I don't know"],
    },
    "anechoic": {
        "correct": ["No echoes at all, so it shows up black, like fluid in a cyst",
                    "Completely black because nothing reflects back, typical of simple fluid"],
        "partial": ["black", "like a cyst"],
        "wrong": ["very bright because it reflects everything", "a kind of artifact behind bone"],
    },
    "acoustic shadowing": {
        "correct": ["A dark band behind something like a stone or bone because the sound can't get through",
                    "The area behind a strong reflector looks dark since little sound passes beyond it"],
        "partial": ["dark area behind a stone", "caused by gallstones"],
        "wrong": ["bright area behind a cyst", "blood flow toward the probe", "no idea"],
    },
    "pericholecystic fluid": {
        "correct": ["Fluid around the outside of the gallbladder", "Fluid surrounding the gallbladder, a sign of cholecystitis"],
        "partial": ["fluid", "gallbladder problem"],
        "wrong": ["stones inside the gallbladder", "thickening of the liver capsule"],
    },
    "resistive index": {
        "correct": ["Peak systolic minus end diastolic velocity, divided by peak systolic; it measures downstream resistance",
                    "A Doppler ratio from the waveform that tells you how much resistance there is downstream"],
        "partial": ["a Doppler measurement", "has to do with resistance"],
        "wrong": ["the angle between the beam and the vessel", "the depth of the focal zone"],
    },
    "portal hypertension": {
        "correct": ["High pressure in the portal vein system, usually from cirrhosis, with signs like a big spleen, ascites and reversed flow",
                    "Raised pressure in the portal venous system caused by liver scarring"],
        "partial": ["high blood pressure", "related to the liver"],
        "wrong": ["an enlarged thyroid", "gas in the bile ducts"],
    },
}


def main() -> int:
    med, _ = load_all()
    emb = OllamaEmbedder()
    rows: dict[str, list[float]] = {"correct": [], "partial": [], "wrong": []}
    for term, answers in CASES.items():
        entry = med.find_term(term)
        if not entry:
            print(f"skip {term}: not in dictionary"); continue
        targets = [entry["definition"]]
        for kind, texts in answers.items():
            vecs = emb.embed(texts + targets)
            for i, t in enumerate(texts):
                s = float(max(vecs[i] @ v for v in vecs[len(texts):]))
                rows[kind].append(s)
                print(f"{term:22s} {kind:8s} {s:5.2f}  {t[:70]}")
    print()
    for kind, vals in rows.items():
        vals.sort()
        print(f"{kind:8s} n={len(vals):2d} min={vals[0]:.2f} median={vals[len(vals)//2]:.2f} max={vals[-1]:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
