"""Predicting the best method, and measuring whether the prediction helps.

The plan this project was built against ends here: circuit features plus
noise characteristics in, predicted best mitigation strategy out. It is
the natural destination, and it is the one place where building the thing
is the easy half.

The hard half is that **a predictor nobody has validated is worse than no
predictor**. It answers every question confidently, it is right often
enough to feel useful, and there is no way to tell from the inside
whether it learned anything or is simply repeating whatever the training
set contained most. That failure mode is this package's entire subject
matter, so a knowledge model here ships with its own refutation attached
or it does not ship.

Three things are therefore built into the module rather than bolted on:

**A baseline it has to beat.** The dumbest possible strategy is to ignore
the features and always name the method that won most often. A predictor
that does not beat that has learned nothing, whatever its raw accuracy.
`cross_validate` reports both numbers side by side and says which won.

**Leave-one-out validation.** Every case is predicted by a model fitted
without it. Reporting accuracy on the training set would measure memory,
not prediction.

**A refusal.** Below `MINIMUM_CASES` the predictor declines. Outside the
range of budgets it has seen, it declines and says so. "I do not know" is
an answer this package is willing to give.

The predictor itself is deliberately the simplest thing that could work:
nearest neighbours by error-budget similarity, majority vote. Not because
something cleverer would not fit better, but because a fit this small a
corpus can support is not worth hiding behind a model nobody can read.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from .ledger import budget_similarity, shares_of

#: Cases needed before the predictor will answer at all. Not a
#: statistical threshold -- a refusal to let a handful of runs look like
#: knowledge.
MINIMUM_CASES = 12

#: How similar the nearest neighbours must be for a prediction to be in
#: distribution. Budget similarity is 1.0 for identical share vectors and
#: falls toward 0 as they diverge.
NEAREST_ENOUGH = 0.75

#: Neighbours consulted. Odd, so a two-way vote cannot tie.
NEIGHBOURS = 5


@dataclass(frozen=True)
class TrainingCase:
    """One measured outcome: this budget, and the method that actually won.

    `winner` is measured, never predicted, and never the method somebody
    recommended. A training set built from recommendations would teach a
    model to reproduce the catalogue's opinions rather than the physics.
    """

    budget_shares: dict
    winner: str
    label: str = ""

    def __post_init__(self) -> None:
        total = sum(self.budget_shares.values())
        if total and abs(total - 1.0) > 0.02:
            raise ValueError(
                f"{self.label or 'case'}: budget shares sum to {total:.3f}, "
                "not 1 -- these are shares of one budget")


@dataclass(frozen=True)
class Prediction:
    """A predicted method, or an explicit refusal to predict one."""

    method: Optional[str]
    reason: str
    votes: dict = field(default_factory=dict)
    nearest_similarity: Optional[float] = None
    n_cases: int = 0

    @property
    def confident(self) -> bool:
        return self.method is not None

    @property
    def agreement(self) -> Optional[float]:
        """Fraction of consulted neighbours that agreed."""
        if not self.votes:
            return None
        return max(self.votes.values()) / sum(self.votes.values())

    def describe(self) -> str:
        if not self.confident:
            return f"  no prediction: {self.reason}"
        return (f"  predicted: {self.method}\n"
                f"    {self.reason}\n"
                f"    neighbours agreeing: {self.agreement:.0%}, "
                f"closest budget {self.nearest_similarity:.2f} similar")


@dataclass(frozen=True)
class Validation:
    """What the predictor scored, and what ignoring it would have scored."""

    n: int
    correct: int
    baseline_correct: int
    baseline_method: str
    refused: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def baseline_accuracy(self) -> float:
        """What ignoring the features actually achieves.

        Measured as the majority class's share of the whole corpus, NOT
        by refitting the majority for each held-out case. Refitting looks
        more even-handed and is a trap: on a corpus split three ways,
        removing one member of the leading class can hand the majority to
        a rival, so the "baseline" misses every case it would really have
        got right.

        That happened here. On sixteen measured cases split 7/6/3, the
        refitted baseline scored 0 of 16 while always naming the most
        common winner scores 7 of 16. Publishing the first number would
        have shown this predictor beating a strawman it had built.
        """
        return self.baseline_correct / self.n if self.n else 0.0

    @property
    def skill(self) -> float:
        """Accuracy above the do-nothing baseline. Zero or negative means
        the features taught it nothing."""
        return self.accuracy - self.baseline_accuracy

    @property
    def beats_baseline(self) -> bool:
        return self.skill > 0

    def describe(self) -> str:
        lines = [
            f"  leave-one-out over {self.n} measured cases",
            f"    predictor {self.accuracy:.1%}",
            f"    always saying {self.baseline_method!r}: "
            f"{self.baseline_accuracy:.1%}",
            f"    skill {self.skill:+.1%}",
        ]
        if self.refused:
            lines.append(f"    refused to predict on {self.refused} "
                         "(counted as wrong, which is the harsher reading)")
        if self.beats_baseline:
            lines.append("  -> the features carry information about the winner")
        else:
            lines.append("  -> no skill: on this corpus the predictor does "
                         "not beat ignoring the features.")
            lines.append("     That is a statement about THIS corpus, not about "
                         "the features. They may")
            lines.append("     well carry the signal and there may simply be too "
                         "few cases to extract it --")
            lines.append("     the fix is more measured cases, not a cleverer "
                         "model. A cleverer model on")
            lines.append("     a corpus this size fits the noise and reports a "
                         "better number, which is how")
            lines.append("     a knowledge model starts lying.")
        return "\n".join(lines)


@dataclass
class Predictor:
    """Nearest neighbours over error budgets. Small, and readable on purpose."""

    cases: list = field(default_factory=list)

    def fit(self, cases) -> "Predictor":
        self.cases = list(cases)
        return self

    def __len__(self) -> int:
        return len(self.cases)

    @property
    def majority(self) -> Optional[str]:
        """The winner that wins most often, ignoring every feature."""
        if not self.cases:
            return None
        return Counter(case.winner for case in self.cases).most_common(1)[0][0]

    def predict(self, budget_shares: dict,
                minimum_cases: int = MINIMUM_CASES) -> Prediction:
        if len(self.cases) < minimum_cases:
            return Prediction(
                None,
                f"only {len(self.cases)} measured cases, need {minimum_cases}. "
                "A handful of runs is not knowledge.",
                n_cases=len(self.cases))

        scored = sorted(
            ((budget_similarity(budget_shares, case.budget_shares), case)
             for case in self.cases),
            key=lambda pair: pair[0], reverse=True)
        nearest = scored[0][0]
        if nearest < NEAREST_ENOUGH:
            return Prediction(
                None,
                f"the closest budget seen is only {nearest:.2f} similar to this "
                f"one (need {NEAREST_ENOUGH}). This is outside what the corpus "
                "has measured, and guessing here is how a model earns trust it "
                "has not been given.",
                nearest_similarity=nearest, n_cases=len(self.cases))

        votes = Counter(case.winner for _, case in scored[:NEIGHBOURS])
        method, count = votes.most_common(1)[0]
        return Prediction(
            method,
            f"{count} of {min(NEIGHBOURS, len(scored))} most similar measured "
            "budgets had this method win",
            votes=dict(votes), nearest_similarity=nearest,
            n_cases=len(self.cases))


def cross_validate(cases, minimum_cases: int = MINIMUM_CASES) -> Validation:
    """Leave-one-out, against the baseline of ignoring the features.

    A refusal counts as wrong. That is harsher than the predictor
    deserves -- refusing is often the right answer -- but the question
    here is whether the features help, and a model that refuses on half
    the corpus has not shown they do.
    """
    cases = list(cases)
    if len(cases) < 2:
        raise ValueError("cross-validation needs at least two cases")

    baseline = Counter(case.winner for case in cases).most_common(1)[0][0]
    correct = refused = 0
    for index, held_out in enumerate(cases):
        rest = cases[:index] + cases[index + 1:]
        prediction = Predictor().fit(rest).predict(
            held_out.budget_shares, minimum_cases=minimum_cases)
        if prediction.method is None:
            refused += 1
        elif prediction.method == held_out.winner:
            correct += 1

    # The baseline is the majority class's share of the corpus: what a
    # person achieves by ignoring every feature and always naming the
    # method that wins most often. See Validation.baseline_accuracy for
    # why this is NOT refit per fold.
    baseline_correct = sum(1 for case in cases if case.winner == baseline)

    return Validation(n=len(cases), correct=correct,
                      baseline_correct=baseline_correct,
                      baseline_method=baseline, refused=refused)


def case_from_budget(budget, winner: str, label: str = "") -> TrainingCase:
    """A training case from an `ErrorBudget` and the method that won."""
    return TrainingCase(budget_shares=shares_of(budget), winner=winner,
                        label=label)
