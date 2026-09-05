"""From a table of results to a recommendation, with the reason attached.

Everything needed to answer "which mitigation should I use" already
existed here -- the method comparisons, the error budget, the fraud
detector, the power analysis -- but it existed as an EXAMPLE. A person
with a circuit had to read `method_shootout.py` and assemble the
judgement themselves, which is exactly the 60 days of assembling
judgement this project is supposed to remove.

This is the decision layer. It takes what the methods actually scored and
returns a recommendation, and it is deliberately made of refusals:

  * If no signal survives the circuit, it refuses to recommend a method
    at all. Every method in the catalogue improves an estimate that
    exists; none creates one.
  * A method that does not read its data is disqualified whatever its
    accuracy. This is the rule the fraud detector exists to enforce, and
    the fraud beats every real method on accuracy alone.
  * Methods the runs cannot separate are returned as a TIER, not an
    order. The shootout's own top three -- the fraud included -- are one
    tier on the measured device.

And one positive rule, which is the point of admitting the tie: **when
methods cannot be separated on accuracy, cost decides.** Ranking a
statistical tie by median is ranking noise. Ranking it by shot cost is
ranking something real, and it gives the user a defensible answer instead
of a coin flip dressed as a measurement.

Stdlib only. The methods are run by the caller -- this module never
touches a backend, so the same decision layer serves a simulation sweep,
a hardware post-mortem, and a service.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any, Optional

from .power import compare, rank_with_ties

#: An honest method's answer moves about as much as the raw estimate does
#: when its data is scrambled. Everything real measured in this project
#: sits between 0.57 and 1.24; the fraud sits at 0.020. The floor is
#: placed in the empty middle rather than tuned against either side.
SENSITIVITY_FLOOR = 0.5


@dataclass(frozen=True)
class MethodOutcome:
    """What one method scored, and whether it was reading the data."""

    name: str
    errors: tuple
    cost: float = 1.0
    sensitivity: Optional[float] = None

    @property
    def error(self) -> float:
        return median(self.errors)

    @property
    def reads_its_data(self) -> Optional[bool]:
        """None when no scramble attack was run.

        Not the same as False, and not the same as True. An unrun attack
        is not a pass -- the same rule this package applies to every
        other control.
        """
        if self.sensitivity is None:
            return None
        return self.sensitivity >= SENSITIVITY_FLOOR


@dataclass(frozen=True)
class Recommendation:
    """What to use, why, and what the evidence does not settle."""

    recommended: Optional[str]
    reason: str
    tiers: tuple = ()
    disqualified: tuple = ()
    unverified: tuple = ()
    tied_with: tuple = ()
    outcomes: dict = field(default_factory=dict)
    feasibility: Any = None

    @property
    def is_a_recommendation(self) -> bool:
        return self.recommended is not None

    def format_report(self) -> str:
        lines = []
        if not self.is_a_recommendation:
            lines.append(f"  no recommendation: {self.reason}")
        else:
            outcome = self.outcomes[self.recommended]
            lines.append(f"  use: {self.recommended}")
            lines.append(f"    error {outcome.error:.4g}, cost {outcome.cost:g}x")
            lines.append(f"    because: {self.reason}")
        if self.tied_with:
            lines.append("    statistically tied with: " + ", ".join(self.tied_with))
            lines.append("      -- these cannot be separated by the runs that were "
                         "done, so cost decided, not accuracy")
        if self.disqualified:
            lines.append("  disqualified for not reading their own data: "
                         + ", ".join(self.disqualified))
        if self.unverified:
            lines.append("  never attacked, so not eligible: " + ", ".join(self.unverified))
        return "\n".join(lines)


def recommend(outcomes, feasibility: Any = None,
              confidence: float = 0.95,
              require_sensitivity: bool = True) -> Recommendation:
    """Rank, disqualify, and pick -- or refuse.

    `outcomes` is a sequence of `MethodOutcome`. `feasibility` is an
    optional `prescribe.Feasibility`; when it says nothing survives the
    circuit, this refuses rather than naming a winner, because a ranked
    table for an experiment that cannot run is a precise answer to a
    question nobody can ask.

    `require_sensitivity` makes an unrun scramble attack disqualifying.
    Left true by default: the fraud in this project's own suite is
    indistinguishable from the best real method on accuracy, so accuracy
    without the attack cannot tell them apart.
    """
    outcomes = list(outcomes)
    if not outcomes:
        raise ValueError("no method outcomes to choose between")
    table = {outcome.name: outcome for outcome in outcomes}

    if feasibility is not None and not feasibility.is_mitigable:
        return Recommendation(
            recommended=None,
            reason=("no signal survives this circuit, so there is nothing for a "
                    "method to improve. "
                    + feasibility.format_verdict().strip().splitlines()[-1].strip()),
            outcomes=table, feasibility=feasibility)

    disqualified = tuple(o.name for o in outcomes if o.reads_its_data is False)
    unverified = tuple(o.name for o in outcomes if o.reads_its_data is None)

    eligible = [o for o in outcomes if o.reads_its_data is True]
    if not eligible and not require_sensitivity:
        eligible = [o for o in outcomes if o.reads_its_data is not False]
    if not eligible:
        return Recommendation(
            recommended=None,
            reason=("no method both read its data and was checked for it. "
                    "Accuracy alone cannot separate a real method from one "
                    "that peeks at the answer."),
            disqualified=disqualified, unverified=unverified,
            outcomes=table, feasibility=feasibility)

    tiers = tuple(tuple(tier) for tier in rank_with_ties(
        {o.name: list(o.errors) for o in eligible}, confidence=confidence))
    best_tier = list(tiers[0])

    # Within a tier the runs cannot tell the methods apart, so ordering
    # them by error is ordering noise. Cost is the thing that is actually
    # known, so it decides -- and ties on cost fall back to the nominal
    # error, which at least is deterministic.
    winner = min(best_tier, key=lambda name: (table[name].cost, table[name].error))
    tied = tuple(name for name in best_tier if name != winner)

    # A winner admitted only because the sensitivity rule was relaxed has
    # NOT been shown to read its data, and saying so would be the
    # over-claiming this package exists to catch.
    sensitivity = table[winner].sensitivity
    standing = (f"it reads its data (sensitivity {sensitivity:.3g})"
                if sensitivity is not None else
                "it was never checked for whether it reads its data, so this "
                "rests on accuracy alone")
    if tied:
        rival = compare(winner, list(table[winner].errors),
                        tied[0], list(table[tied[0]].errors),
                        confidence=confidence)
        reason = (f"it is in the best tier the runs can establish, {standing}, "
                  f"and at {table[winner].cost:g}x it is the cheapest of a group "
                  f"that cannot be separated on accuracy -- {rival.describe()}")
    else:
        reason = (f"it is the best method the runs can establish, and {standing}")

    return Recommendation(recommended=winner, reason=reason, tiers=tiers,
                          disqualified=disqualified, unverified=unverified,
                          tied_with=tied, outcomes=table, feasibility=feasibility)


@dataclass(frozen=True)
class Consensus:
    """One answer from many methods, with their disagreement inside the bar.

    The request this exists to serve is "run everything and give me the
    answer". The honest form of that is not one method's number: picking
    a single winner throws away the fact that the others landed
    elsewhere, and on this project's own suite the top methods are
    routinely inseparable.

    So the estimate is the median over methods that survived their
    attacks, and the uncertainty carries TWO terms:

      * **statistical** -- the shot noise each method reports;
      * **systematic** -- how far the surviving methods sit from each
        other.

    The second is usually the larger, and it is the one a single-method
    pipeline cannot compute at all. Two methods agreeing to within their
    shot noise is evidence; two disagreeing by ten times it means the
    answer depends on which assumption you made, and no amount of extra
    shots fixes that.
    """

    estimate: Optional[float]
    statistical: float
    systematic: float
    methods: tuple
    excluded: tuple = ()
    reason: str = ""

    @property
    def uncertainty(self) -> float:
        """The two terms added in quadrature."""
        return (self.statistical ** 2 + self.systematic ** 2) ** 0.5

    @property
    def dominated_by_disagreement(self) -> bool:
        """When true, more shots will not help. The methods disagree."""
        return self.systematic > self.statistical

    @property
    def is_an_answer(self) -> bool:
        return self.estimate is not None

    def format_report(self) -> str:
        if not self.is_an_answer:
            return f"  no consensus: {self.reason}"
        lines = [
            f"  answer: {self.estimate:.6g} +- {self.uncertainty:.4g}",
            f"    shot noise      {self.statistical:.4g}",
            f"    method spread   {self.systematic:.4g}",
            f"    agreed by {len(self.methods)} method"
            f"{'s' if len(self.methods) != 1 else ''}: "
            + ", ".join(self.methods),
        ]
        if self.excluded:
            lines.append("    excluded: " + ", ".join(self.excluded))
        if self.dominated_by_disagreement:
            lines.append("    -> the bar is dominated by DISAGREEMENT between "
                         "methods, not by shot")
            lines.append("       noise. More shots will not narrow it; the "
                         "methods are making")
            lines.append("       different assumptions and those assumptions "
                         "are pulling apart.")
        else:
            lines.append("    -> the surviving methods agree within their own "
                         "shot noise, so the")
            lines.append("       bar is statistical and more shots would narrow it.")
        return "\n".join(lines)


def consensus(outcomes, statistical: Optional[float] = None,
              require_sensitivity: bool = True) -> Consensus:
    """Run everything, then answer with what survived.

    `outcomes` is a sequence of `MethodOutcome`. A method that failed its
    scramble attack is excluded outright: including a fraud's number in a
    median would let it move the answer it was caught not reading.

    `statistical` is the shot-noise half-width if the caller has one --
    from `results.ShotNoise`, say. Without it, the spread of each
    method's own repeats stands in.
    """
    outcomes = list(outcomes)
    if not outcomes:
        raise ValueError("no method outcomes to form a consensus from")

    excluded = tuple(o.name for o in outcomes if o.reads_its_data is False)
    if require_sensitivity:
        surviving = [o for o in outcomes if o.reads_its_data is True]
    else:
        surviving = [o for o in outcomes if o.reads_its_data is not False]

    if not surviving:
        return Consensus(
            None, 0.0, 0.0, (), excluded,
            "no method both read its data and was checked for it. Averaging "
            "methods that may not be reading the data would produce a number "
            "with no relationship to the experiment.")
    if len(surviving) == 1:
        only = surviving[0]
        spread = (max(only.errors) - min(only.errors)) / 2 if len(only.errors) > 1 else 0.0
        return Consensus(
            only.error, statistical if statistical is not None else spread, 0.0,
            (only.name,), excluded,
            "one surviving method, so there is no disagreement to measure -- "
            "which is not the same as agreement")

    values = sorted(o.error for o in surviving)
    middle = len(values) // 2
    estimate = (values[middle] if len(values) % 2
                else (values[middle - 1] + values[middle]) / 2)
    # Half the full range: a plain spread, not a standard deviation, since
    # three or four methods is too few for a deviation to mean much.
    systematic = (max(values) - min(values)) / 2
    if statistical is None:
        spreads = [(max(o.errors) - min(o.errors)) / 2
                   for o in surviving if len(o.errors) > 1]
        statistical = sum(spreads) / len(spreads) if spreads else 0.0

    return Consensus(estimate, statistical, systematic,
                     tuple(o.name for o in surviving), excluded)
