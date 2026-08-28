"""QEM-Trust: scoring an auditor, not an experiment.

Everything else in this package points one way -- a record goes in, a
verdict comes out. This module points the other way. It takes an AUDITOR
(this package's, a competitor's, an LLM's, a human's) and asks how well
it separates real results from artifacts on a suite of cases whose truth
is already known.

The reason this is a separate exercise, and not just "run the benchmarks",
is that `run_benchmarks.py` pins qem-auditor's own verdicts against
themselves. That is a regression test and it is worth having, but it is
circular as evidence: it shows the auditor is stable, not that it is
discriminating. A benchmark earns the name only when something OTHER than
the tool it ships with can be scored on it.

Three design decisions carry the weight here.

**Errors are not symmetric, so accuracy is the wrong metric.** Calling a
sound result invalid wastes a person's month. Calling an artifact
certified puts a wrong number into the literature, where the next group
builds on it. Those are not the same mistake and no single accuracy
figure can hold both. So the score names the error classes separately,
and one of them -- endorsing something the suite knows to be broken -- is
disqualifying on its own, at any aggregate score. That is the same rule
`verdict.py` applies to experiments (one failed hard gate forces INVALID,
however good everything else looks) turned on the auditor itself.

**Hedging must score zero, and be MADE to score zero rather than merely
discouraged.** An auditor that answers NOT ESTABLISHED to every case is
never wrong in the damaging direction, and on a suite with enough
uncertain cases it can beat a real auditor on raw credit. So the headline
number is not credit; it is credit measured against the best a CONSTANT
answer could have achieved on this same suite. The perpetual hedger
scores exactly 0 by construction. So does the perpetual condemner.

**The suite is small, and the score says so.** Six cases cannot support
a claim like "94% accurate". Every report carries the Wilson interval on
its own exact-match rate, computed the same way `power.py` computes
intervals for the experiments being audited. A benchmark that over-claims
about its own resolution has no standing to audit anything.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Sequence, Union

from .failure_modes import classify
from .power import normal_quantile
from .schema import Experiment, FailureMode
from .verdict import Verdict, audit


class Stance(Enum):
    """What a verdict does to the person who receives it.

    The eight verdicts differ in diagnosis, but they act on a reader in
    only three ways: stop, wait, or proceed. Scoring happens at this
    resolution because this is the resolution at which a mistake costs
    something.
    """

    CONDEMNING = "condemning"    # stop: this result is not what it claims
    WITHHOLDING = "withholding"  # wait: not shown either way
    ENDORSING = "endorsing"      # proceed: build on this


STANCE: dict[Verdict, Stance] = {
    Verdict.INVALID_RECORD: Stance.CONDEMNING,
    Verdict.INVALID: Stance.CONDEMNING,
    Verdict.REFUTED: Stance.CONDEMNING,
    Verdict.CONFLICT: Stance.WITHHOLDING,
    Verdict.NOT_ESTABLISHED: Stance.WITHHOLDING,
    Verdict.MODEL_CONDITIONAL: Stance.WITHHOLDING,
    Verdict.PROMISING: Stance.ENDORSING,
    Verdict.CERTIFIED_UNDER_SCOPE: Stance.ENDORSING,
}


class ErrorKind(Enum):
    """Why an answer was wrong, at the resolution that matters.

    CONFLICT and MODEL_CONDITIONAL both withhold, so confusing them is a
    different kind of mistake from certifying a broken result, and the
    scoring must not average them together.
    """

    NONE = "correct"
    TIER_SLIP = "right stance, wrong verdict"
    OVER_HEDGE = "withheld from a sound result"
    MISSED_CONDEMNATION = "withheld from a broken result"
    FALSE_CONDEMNATION = "condemned a result the suite does not condemn"
    OVER_CLAIM = "endorsed a result the suite only withholds on"
    FALSE_ENDORSEMENT = "endorsed a result the suite knows to be broken"


#: Credit per case. These weights order the error classes; they are not
#: physical constants, and the headline `skill` is deliberately built to
#: be insensitive to their exact values (it is measured against the best
#: constant answer scored under the same weights). What they must get
#: right is the ORDER: an exact call beats a near miss, a near miss beats
#: a hedge, a hedge beats a wrong-direction call, and the two ways of
#: endorsing something unearned pay nothing at all.
CREDIT: dict[ErrorKind, float] = {
    ErrorKind.NONE: 1.0,
    ErrorKind.TIER_SLIP: 0.6,
    ErrorKind.OVER_HEDGE: 0.3,
    ErrorKind.MISSED_CONDEMNATION: 0.2,
    ErrorKind.FALSE_CONDEMNATION: 0.0,
    ErrorKind.OVER_CLAIM: 0.0,
    ErrorKind.FALSE_ENDORSEMENT: 0.0,
}

#: The one error that is disqualifying regardless of aggregate score.
DISQUALIFYING = ErrorKind.FALSE_ENDORSEMENT


class TrustGrade(Enum):
    """Deliberately not a percentage.

    SUITE_SATURATED is the top and it is not a compliment: it means this
    suite can no longer tell this auditor apart from a perfect one, which
    is a statement about the suite's resolution, not about the auditor's
    reliability. Six cases cannot certify an auditor and the name should
    not let anyone pretend otherwise.
    """

    DISQUALIFIED = "DISQUALIFIED (endorsed a known artifact)"
    NO_SKILL = "NO SKILL (no better than a constant answer)"
    PARTIAL_SKILL = "PARTIAL SKILL"
    SUITE_SATURATED = "SUITE SATURATED (this suite cannot resolve further)"


class CaseProvenance(Enum):
    """Where a case came from.

    DISCLOSED cases are real published results whose truth was settled by
    the experiment's own history -- someone ran the follow-up work and
    found out. CONSTRUCTED cases are built to isolate one discrimination;
    their truth follows from the record as written, which is weaker
    evidence about the world and stronger evidence about the auditor.

    Both belong in a suite and they must never be blended into one
    headline number without saying so. A tool that scores well only on
    constructed cases has learned the schema; a suite that hides the
    split lets it look like it has learned the physics. Reports break the
    score down by provenance for exactly that reason.

    CONSTRUCTED is the default: claiming a case is a real disclosed
    result should take a deliberate act, not an omission.
    """

    DISCLOSED = "disclosed"
    CONSTRUCTED = "constructed"


@dataclass(frozen=True)
class Case:
    """One scored case: a record, and what it is known to deserve.

    `what_it_tests` is not decoration. A suite whose cases cannot each
    say what they discriminate is a pile of records, and its aggregate
    score means nothing.
    """

    case_id: str
    experiment: Experiment
    truth: Verdict
    what_it_tests: str
    truth_mode: Optional[FailureMode] = None
    provenance: CaseProvenance = CaseProvenance.CONSTRUCTED

    def __post_init__(self) -> None:
        if not self.what_it_tests.strip():
            raise ValueError(
                f"case {self.case_id!r} does not say what it discriminates; "
                "a case that cannot say that cannot be scored against"
            )
        if self.truth_mode is not None and STANCE[self.truth] is Stance.ENDORSING:
            raise ValueError(
                f"case {self.case_id!r} pins a failure mode but its truth "
                f"verdict {self.truth.name} endorses the result; there is no "
                "failure to attribute"
            )


@dataclass(frozen=True)
class Answer:
    """What an auditor returned. The mode is optional: an auditor that
    only ranks verdicts is still scoreable, it just earns no attribution
    credit."""

    verdict: Verdict
    primary_mode: Optional[FailureMode] = None


@dataclass(frozen=True)
class Pair:
    """Two cases differing in ONE stated respect, where that difference
    is supposed to move the verdict.

    Aggregate accuracy is generous to an auditor that has learned which
    records tend to be bad. A minimal pair is not: both members look
    alike, so the only way to score the pair is to react to the thing
    that actually differs. Credit is all-or-nothing across the two --
    getting one right and one wrong is what guessing looks like, and it
    should pay what guessing is worth.

    `difference` is the one field that moved; `why_it_matters` is the
    argument that it should. A pair that cannot state both is two cases
    that happen to resemble each other.
    """

    pair_id: str
    case_a: str
    case_b: str
    difference: str
    why_it_matters: str

    def __post_init__(self) -> None:
        if self.case_a == self.case_b:
            raise ValueError(
                f"pair {self.pair_id!r} names the same case twice; a case "
                "cannot be a minimal pair with itself"
            )
        for text, label in ((self.difference, "difference"),
                            (self.why_it_matters, "why_it_matters")):
            if not text.strip():
                raise ValueError(
                    f"pair {self.pair_id!r} does not state its {label}; a pair "
                    "that cannot say what differs is two unrelated cases"
                )


#: An auditor may return a bare Verdict; `normalise_answer` widens it.
AuditorFn = Callable[[Experiment], Union[Answer, Verdict]]


def normalise_answer(returned: Union[Answer, Verdict]) -> Answer:
    if isinstance(returned, Answer):
        return returned
    if isinstance(returned, Verdict):
        return Answer(returned)
    raise TypeError(
        f"an auditor must return a Verdict or an Answer, got {type(returned).__name__}"
    )


def classify_error(truth: Verdict, given: Verdict) -> ErrorKind:
    if truth is given:
        return ErrorKind.NONE
    t, g = STANCE[truth], STANCE[given]
    if t is g:
        return ErrorKind.TIER_SLIP
    if g is Stance.ENDORSING:
        return (ErrorKind.FALSE_ENDORSEMENT if t is Stance.CONDEMNING
                else ErrorKind.OVER_CLAIM)
    if g is Stance.CONDEMNING:
        return ErrorKind.FALSE_CONDEMNATION
    return (ErrorKind.MISSED_CONDEMNATION if t is Stance.CONDEMNING
            else ErrorKind.OVER_HEDGE)


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    truth: Verdict
    given: Verdict
    error: ErrorKind
    credit: float
    truth_mode: Optional[FailureMode]
    given_mode: Optional[FailureMode]
    provenance: CaseProvenance
    #: None when attribution was not in play: the case has no pinned
    #: cause, or the auditor endorsed the record, so it never offered
    #: one. Scoring a diagnosis on a case the auditor waved through would
    #: pay it twice for one answer.
    attribution: Optional[bool]


@dataclass(frozen=True)
class Interval:
    low: float
    high: float

    def __str__(self) -> str:
        return f"[{self.low:.2f}, {self.high:.2f}]"


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval on a proportion.

    Used rather than the textbook normal interval because at n=6 the
    normal interval runs off the ends of [0,1] and reports impossible
    bounds -- which is exactly the kind of over-claim this package exists
    to catch.
    """
    if n <= 0:
        raise ValueError("no cases: there is no proportion to bound")
    if not 0 <= successes <= n:
        raise ValueError(f"{successes} successes out of {n} is not a proportion")
    z = normal_quantile(1.0 - (1.0 - confidence) / 2.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Interval(max(0.0, centre - half), min(1.0, centre + half))


def _constant_credit(cases: Sequence[Case], verdict: Verdict) -> float:
    return sum(CREDIT[classify_error(c.truth, verdict)] for c in cases) / len(cases)


def best_constant(cases: Sequence[Case]) -> tuple[Verdict, float]:
    """The strongest answer available to something that never looks at
    the record. This is the bar `skill` is measured against."""
    scored = [(v, _constant_credit(cases, v)) for v in Verdict]
    best = max(scored, key=lambda vc: vc[1])
    return best


@dataclass(frozen=True)
class TrustReport:
    auditor_name: str
    results: list[CaseResult]
    baseline_verdict: Verdict
    baseline_credit: float
    pairs: list[Pair] = field(default_factory=list)

    @property
    def by_id(self) -> dict[str, CaseResult]:
        return {r.case_id: r for r in self.results}

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def credit(self) -> float:
        return sum(r.credit for r in self.results) / self.n

    @property
    def exact(self) -> int:
        return sum(1 for r in self.results if r.error is ErrorKind.NONE)

    @property
    def skill(self) -> float:
        """Credit above the best constant answer, normalised so that a
        perfect auditor scores 1.0 and any constant answerer scores <= 0.

        Not clipped at zero: an auditor that is actively worse than
        saying the same thing every time should see a negative number.
        """
        ceiling = 1.0 - self.baseline_credit
        if ceiling <= 0.0:
            # Degenerate suite: one constant answer is already perfect,
            # so the suite cannot measure skill at all.
            return 0.0
        return (self.credit - self.baseline_credit) / ceiling

    @property
    def false_endorsements(self) -> list[CaseResult]:
        return [r for r in self.results if r.error is DISQUALIFYING]

    def errors_of(self, kind: ErrorKind) -> list[CaseResult]:
        return [r for r in self.results if r.error is kind]

    @property
    def exact_interval(self) -> Interval:
        return wilson_interval(self.exact, self.n)

    @property
    def attribution(self) -> Optional[tuple[int, int]]:
        """(correct, scoreable) over cases where a diagnosis was actually
        in play, or None if none were."""
        scored = [r for r in self.results if r.attribution is not None]
        if not scored:
            return None
        return sum(1 for r in scored if r.attribution), len(scored)

    @property
    def pair_score(self) -> Optional[tuple[int, int]]:
        """(pairs solved, pairs offered), where a pair is solved only if
        BOTH members are exact. None when the suite defines no pairs."""
        if not self.pairs:
            return None
        results = self.by_id
        solved = 0
        for pair in self.pairs:
            a, b = results.get(pair.case_a), results.get(pair.case_b)
            if a is None or b is None:
                raise ValueError(
                    f"pair {pair.pair_id!r} names a case not in this suite; "
                    "the pair cannot be scored"
                )
            if a.error is ErrorKind.NONE and b.error is ErrorKind.NONE:
                solved += 1
        return solved, len(self.pairs)

    def credit_on(self, provenance: CaseProvenance) -> Optional[float]:
        subset = [r for r in self.results if r.provenance is provenance]
        if not subset:
            return None
        return sum(r.credit for r in subset) / len(subset)

    def exact_on(self, provenance: CaseProvenance) -> tuple[int, int]:
        subset = [r for r in self.results if r.provenance is provenance]
        return sum(1 for r in subset if r.error is ErrorKind.NONE), len(subset)

    @property
    def grade(self) -> TrustGrade:
        if self.false_endorsements:
            return TrustGrade.DISQUALIFIED
        if self.skill <= 0.0:
            return TrustGrade.NO_SKILL
        if self.exact == self.n:
            return TrustGrade.SUITE_SATURATED
        return TrustGrade.PARTIAL_SKILL

    def format_report(self) -> str:
        lines = [
            "=" * 68,
            f"QEM-Trust  |  auditor: {self.auditor_name}",
            "=" * 68,
        ]
        for r in self.results:
            mark = "OK  " if r.error is ErrorKind.NONE else "MISS"
            lines.append(f"  [{mark}] {r.case_id}")
            lines.append(f"         truth: {r.truth.value}")
            lines.append(f"         given: {r.given.value}")
            if r.error is not ErrorKind.NONE:
                lines.append(f"         error: {r.error.value}")
            if r.attribution is not None:
                verdict_word = "correct" if r.attribution else "wrong"
                given = r.given_mode.name if r.given_mode else "none offered"
                lines.append(f"         cause: {verdict_word} ({given})")
        lines.append("-" * 68)
        lines.append(f"  exact          {self.exact}/{self.n}   "
                     f"95% CI {self.exact_interval}")
        lines.append(f"  credit         {self.credit:.3f}")
        lines.append(f"  best constant  {self.baseline_credit:.3f} "
                     f"(always {self.baseline_verdict.value})")
        lines.append(f"  SKILL          {self.skill:+.3f}")
        attr = self.attribution
        if attr is not None:
            lines.append(f"  attribution    {attr[0]}/{attr[1]} causes named correctly")
        pairs = self.pair_score
        if pairs is not None:
            lines.append(f"  PAIRS          {pairs[0]}/{pairs[1]} solved "
                         "(both members exact, or no credit)")
        # Never one blended number: a tool can score well on constructed
        # cases by learning the schema, and the split is what shows it.
        split = [(p, self.exact_on(p)) for p in CaseProvenance]
        if len([1 for _p, (_e, n) in split if n]) > 1:
            for provenance, (exact, n) in split:
                if n:
                    lines.append(f"    {provenance.value:12s} {exact}/{n} exact, "
                                 f"credit {self.credit_on(provenance):.3f}")
        for kind in (ErrorKind.FALSE_ENDORSEMENT, ErrorKind.OVER_CLAIM,
                     ErrorKind.FALSE_CONDEMNATION):
            hits = self.errors_of(kind)
            if hits:
                lines.append(f"  {kind.value}: "
                             + ", ".join(h.case_id for h in hits))
        lines.append("-" * 68)
        lines.append(f"  GRADE: {self.grade.value}")
        if self.grade is TrustGrade.DISQUALIFIED:
            lines.append("  A single endorsement of a known artifact disqualifies,")
            lines.append("  whatever the aggregate score. Same rule this package")
            lines.append("  applies to the experiments it audits.")
        elif self.grade is TrustGrade.SUITE_SATURATED:
            lines.append(f"  Saturating {self.n} cases is not a certificate. It means")
            lines.append("  the suite is out of resolution and needs harder cases.")
        lines.append("=" * 68)
        return "\n".join(lines)

    def print_report(self) -> None:
        print(self.format_report())


def score(auditor: AuditorFn, cases: Sequence[Case],
          name: str = "unnamed",
          pairs: Sequence[Pair] = ()) -> TrustReport:
    """Run `auditor` over `cases` and score it.

    The auditor sees only the Experiment -- never the truth verdict, never
    the case's `what_it_tests` note. That is what makes the score mean
    anything.
    """
    if not cases:
        raise ValueError("an empty suite scores nothing")
    ids = [c.case_id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case ids: the suite would double-count")

    results = []
    for case in cases:
        answer = normalise_answer(auditor(case.experiment))
        error = classify_error(case.truth, answer.verdict)
        attribution: Optional[bool] = None
        # Attribution is in play whenever the case pins a cause and the
        # auditor did not wave the record through. A withholding verdict
        # still carries a cause -- "not established BECAUSE it was never
        # replicated" is an attribution, and a suite that only scored
        # causes on condemnations would leave two of its six cases
        # unscored on the axis they were chosen to test.
        if case.truth_mode is not None and STANCE[answer.verdict] is not Stance.ENDORSING:
            attribution = answer.primary_mode is case.truth_mode
        results.append(CaseResult(
            case_id=case.case_id,
            truth=case.truth,
            given=answer.verdict,
            error=error,
            credit=CREDIT[error],
            truth_mode=case.truth_mode,
            given_mode=answer.primary_mode,
            provenance=case.provenance,
            attribution=attribution,
        ))
    baseline_verdict, baseline_credit = best_constant(cases)
    report = TrustReport(name, results, baseline_verdict, baseline_credit, list(pairs))
    # Surface a pair naming a missing case now, at scoring time, rather
    # than when someone reads the number off the report.
    report.pair_score
    return report


def constant_auditor(verdict: Verdict) -> AuditorFn:
    """A baseline that ignores the record entirely. Included so the
    zero point of `skill` can be demonstrated rather than asserted."""

    def _auditor(_exp: Experiment) -> Answer:
        return Answer(verdict)

    return _auditor


def number_reading_auditor(exp: Experiment) -> Answer:
    """A baseline that reads the results table and nothing else.

    Not a straw man: this is what eyeballing a paper's summary numbers
    looks like, and it is a fair description of how most claims are
    actually assessed. It condemns when the mitigation made things worse,
    certifies a small error with enough replicates, and hedges otherwise.

    It is included because it is the demonstration that the constructed
    minimal pairs earn their place. On the six disclosed cases it scores
    partial skill and looks like a passable auditor. On the pairs it
    solves none, because every pair holds the numbers fixed and moves
    something it cannot see -- and it endorses two records the suite
    knows to be broken, which disqualifies it outright.
    """
    raw = exp.outputs.raw_error_kcal
    mitigated = exp.outputs.mitigated_error_kcal
    if raw is None or mitigated is None:
        return Answer(Verdict.NOT_ESTABLISHED)
    if mitigated > raw:
        return Answer(Verdict.INVALID)
    if mitigated <= 0.25 and len(exp.outputs.replicates) >= 8:
        return Answer(Verdict.CERTIFIED_UNDER_SCOPE)
    if mitigated < raw:
        return Answer(Verdict.PROMISING)
    return Answer(Verdict.NOT_ESTABLISHED)


def builtin_auditor(exp: Experiment) -> Answer:
    """This package, wired up as a scoreable contestant on its own
    benchmark -- with no access to the answer key that anyone else's
    auditor also lacks."""
    report = audit(exp)
    diagnosis = classify(exp, report).primary
    return Answer(report.verdict, diagnosis.mode if diagnosis else None)
