"""V4: predict the best method from the error budget -- and check it.

The natural destination of this project's plan: circuit and noise
characteristics in, predicted best mitigation strategy out. Building that
is the easy half. The hard half is that a predictor nobody has validated
is WORSE than no predictor -- it answers every question confidently, and
from the inside there is no way to tell whether it learned anything or is
repeating whatever the training set contained most.

So this runs the predictor against the only honest test: leave-one-out,
compared with the dumbest possible strategy of ignoring every feature and
always naming the method that wins most often.

The training data is measured, not assumed. `benchmarks/measured_winners.json`
holds sixteen noise regimes -- a 4x4 grid over two-qubit and readout error
-- in each of which every method in the catalogue was actually run on H2
and the winner recorded against the known energy. The two deliberate
frauds are excluded as labels: a training set that can name a fraud as
the winner teaches a model to recommend one.

The result is reported whichever way it comes out.

Run:  python examples/knowledge_model.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qem_auditor.predict import (Predictor, TrainingCase,  # noqa: E402
                                 cross_validate)

CASES = Path(__file__).resolve().parent.parent / "benchmarks" / "measured_winners.json"


def load():
    return [TrainingCase(budget_shares=entry["budget_shares"],
                         winner=entry["winner"], label=entry["label"])
            for entry in json.loads(CASES.read_text())]


def main() -> int:
    cases = load()

    print("=" * 74)
    print("  What actually won, across 16 measured noise regimes")
    print("=" * 74)
    for method, count in Counter(case.winner for case in cases).most_common():
        print(f"  {count:2d}x  {method}")
    print("\n  The winner is not constant. Readout-heavy regimes go to readout")
    print("  mitigation, gate-heavy ones to CDR. So there is, in principle,")
    print("  something here for a model to learn.")

    print("\n" + "=" * 74)
    print("  Does the predictor learn it?")
    print("=" * 74)
    report = cross_validate(cases, minimum_cases=6)
    print(report.describe())

    print("\n" + "=" * 74)
    print("  What that means")
    print("=" * 74)
    if report.beats_baseline:
        print("  The features beat the do-nothing strategy. The predictor is")
        print("  worth consulting -- within the range of budgets it has seen.")
    else:
        print("  The predictor is WORSE than ignoring the features entirely.")
        print("  Sixteen cases split three ways leaves roughly five per class,")
        print("  and nearest-neighbour voting on five examples is noise. This")
        print("  is not a claim that the features are uninformative -- the")
        print("  table above shows they are. It is a claim that THIS corpus")
        print("  is too small to extract it, which is a different and more")
        print("  fixable problem.")
        print()
        print("  The right response is more measured cases, not a cleverer")
        print("  model. A cleverer model on sixteen points would fit the")
        print("  noise and report a better number, which is how a knowledge")
        print("  model starts lying.")

    print("\n" + "=" * 74)
    print("  And it refuses when it should")
    print("=" * 74)
    predictor = Predictor().fit(cases)
    unseen = predictor.predict({"DECOHERENCE": 0.95, "SHOT_NOISE": 0.05})
    print(unseen.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
