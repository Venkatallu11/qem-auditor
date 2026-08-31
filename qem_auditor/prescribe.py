"""What to do about it: from a refusal to a better experiment.

An auditor that only says no wastes the time of the honest people it is
supposed to serve. Someone who brings a circuit and gets INVALID back has
learned that their afternoon was wasted, and nothing about what would not
have wasted it. This module is the other half: given where the error
actually comes from, what is the best available thing to do?

The reason this can be more than folklore is that this package measured
it. `examples/method_shootout.py` runs nine methods under two noise
models, and the ranking inverts between them:

    method            invented depolarizing    measured fake_kyiv
    ZNE                            3.97 <-- best      30.95 <-- useless
    REM (readout)                 21.70 <-- useless    6.18 <-- best
    REM + ZNE                      3.77               1.56

Same circuit, same shots, same seeds. Only the noise changed. So a
recommender that names a method without first asking where the error
COMES FROM is guessing, however confident it sounds, and would have told
half the users of this tool to use the worst available option.

The organising idea is SCALING. Gate folding multiplies the number of
gates, so it multiplies every error that grows with gate count -- and
leaves untouched every error that does not. Readout error happens once,
at measurement, whatever the fold factor. That is not a tuning problem or
a bad-luck result; no extrapolation in the fold factor can ever reach a
term that is constant in the fold factor. The measured consequence was
5.53x becoming 1.14x when the noise changed from a model with no readout
error to a device with 2.93% of it.

So every prescription here begins from an error budget and reasons about
what each candidate can physically reach. Where the budget cannot be
established the recommendation is to go and measure it, not to guess --
"try ZNE and see" is exactly the advice that produced the failures in
`benchmarks/`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .schema import Provenance


class ErrorSource(Enum):
    """Where the error in an expectation value comes from.

    Deliberately cut by MECHANISM rather than by blame. "Hardware noise"
    is one bucket to a user and four different problems to a mitigator,
    and the four want different answers.
    """

    SHOT_NOISE = "finite sampling"
    READOUT = "measurement assignment error"
    GATE_STOCHASTIC = "incoherent gate error"
    DECOHERENCE = "T1/T2 relaxation and dephasing"
    COHERENT = "systematic over- or under-rotation"
    ANSATZ = "the circuit cannot represent the answer"


class Scaling(Enum):
    """How an error term responds to unitary folding.

    This is the property that decides whether zero-noise extrapolation
    can reach a term at all, and it is a fact about the physics rather
    than a tuning choice. A term constant in the fold factor is invisible
    to any extrapolation in the fold factor, no matter how good the fit.
    """

    WITH_GATE_COUNT = "grows with the number of gates"
    WITH_DURATION = "grows with how long the circuit takes"
    CONSTANT = "unchanged by folding -- extrapolation cannot reach it"
    WITH_INVERSE_SHOTS = "shrinks with more shots, and mitigation AMPLIFIES it"


#: How each source scales, and therefore what can touch it.
SCALES_AS: dict[ErrorSource, Scaling] = {
    ErrorSource.SHOT_NOISE: Scaling.WITH_INVERSE_SHOTS,
    ErrorSource.READOUT: Scaling.CONSTANT,
    ErrorSource.GATE_STOCHASTIC: Scaling.WITH_GATE_COUNT,
    ErrorSource.DECOHERENCE: Scaling.WITH_DURATION,
    ErrorSource.COHERENT: Scaling.WITH_GATE_COUNT,
    ErrorSource.ANSATZ: Scaling.CONSTANT,
}


@dataclass(frozen=True)
class Cost:
    """What a prescription asks for, in the units a user actually pays.

    Wall-clock and money follow from these two on any backend, and
    neither is knowable here, so neither is invented.
    """

    extra_circuits: int
    shot_multiplier: float
    note: str = ""

    def __str__(self) -> str:
        parts = []
        if self.extra_circuits:
            parts.append(f"+{self.extra_circuits} circuits")
        if self.shot_multiplier != 1.0:
            parts.append(f"{self.shot_multiplier:g}x shots")
        if not parts:
            parts.append("free")
        return ", ".join(parts) + (f" ({self.note})" if self.note else "")


@dataclass(frozen=True)
class Method:
    """A mitigation method, described by what it can physically reach.

    `reaches` is the load-bearing field. It is not a summary of how well
    the method scored somewhere; it is which error mechanisms the method
    is capable of acting on at all. A method that cannot reach the
    dominant term cannot help however well it performs elsewhere, and
    that is a statement about mechanism, not about benchmark luck.
    """

    name: str
    reaches: frozenset
    cost: Cost
    assumes: str
    evidence: str
    #: Methods whose correctness rests on an assumed noise model are
    #: flagged: when the assumption is wrong they do not degrade, they
    #: invert. Measured at 2.03 -> 17.13 kcal/mol for PEC when its
    #: assumed model stopped matching the device.
    model_dependent: bool = False
    #: What share of a reachable term this method can actually act on.
    #:
    #: Most methods act on the whole expectation value, so 1.0. Symmetry
    #: post-selection does not: it can only discard shots in a basis
    #: where the symmetry is visible, and terms measured after a basis
    #: rotation that breaks the symmetry pass through untouched. Ranking
    #: it as though it reached all of a term put it first on a budget
    #: where it delivered 5.14x against CDR's 20.46x -- caught by
    #: examples/prescribe_for_circuit.py checking the order it predicted
    #: against the order that happened.
    reach_fraction: float = 1.0


CATALOGUE: tuple[Method, ...] = (
    Method(
        name="more shots",
        reaches=frozenset({ErrorSource.SHOT_NOISE}),
        cost=Cost(0, 4.0, "4x shots halves shot noise"),
        assumes="nothing",
        evidence="shot noise falls as 1/sqrt(N) by construction; no mitigation "
                 "method reduces it and every extrapolator amplifies it",
    ),
    Method(
        name="readout error mitigation (REM)",
        reaches=frozenset({ErrorSource.READOUT}),
        cost=Cost(4, 1.0, "one calibration circuit per basis state"),
        assumes="readout errors are stable between calibration and the run",
        evidence="measured 36.46 -> 6.18 kcal/mol on fake_kyiv, where readout "
                 "error is 9x the two-qubit gate error; and 21.95 -> 21.70 on a "
                 "model with no readout error, where it correctly does nothing",
    ),
    Method(
        name="symmetry verification (post-selection)",
        reaches=frozenset({ErrorSource.READOUT, ErrorSource.GATE_STOCHASTIC,
                           ErrorSource.DECOHERENCE}),
        cost=Cost(0, 1.5, "discarded shots must be replaced"),
        assumes="the state obeys a symmetry you can check in the measured basis",
        evidence="measured 36.46 -> 7.09 kcal/mol; reaches any error that "
                 "pushes the state out of the symmetry sector, and nothing that "
                 "keeps it inside",
        # Three of the H2 Hamiltonian's five terms are measured in the Z
        # basis, where this symmetry is visible; the XX term is measured
        # after a rotation under which it is not. Post-selection acts on
        # the former and not the latter.
        reach_fraction=0.6,
    ),
    Method(
        name="zero-noise extrapolation (ZNE)",
        reaches=frozenset({ErrorSource.GATE_STOCHASTIC, ErrorSource.DECOHERENCE,
                           ErrorSource.COHERENT}),
        cost=Cost(2, 3.0, "one circuit per noise scale factor"),
        assumes="every error present grows with the amplification knob",
        evidence="measured 21.95 -> 3.97 kcal/mol where gate error dominates, "
                 "and 36.46 -> 30.95 where readout error does -- folding cannot "
                 "reach a term that is constant in the fold factor",
    ),
    Method(
        name="REM then ZNE",
        reaches=frozenset({ErrorSource.READOUT, ErrorSource.GATE_STOCHASTIC,
                           ErrorSource.DECOHERENCE, ErrorSource.COHERENT}),
        cost=Cost(6, 3.0, "both methods' circuits"),
        assumes="both of the above",
        evidence="measured 36.46 -> 1.56 kcal/mol, the best honest result on "
                 "the device -- remove what folding cannot reach, then "
                 "extrapolate what it can",
    ),
    Method(
        name="Clifford data regression (CDR)",
        reaches=frozenset({ErrorSource.READOUT, ErrorSource.GATE_STOCHASTIC,
                           ErrorSource.DECOHERENCE, ErrorSource.COHERENT}),
        cost=Cost(5, 2.0, "one circuit per training angle"),
        assumes="the noise map learned at Clifford points transfers to the "
                "angle you actually use -- an assumption about the CIRCUIT, "
                "traded for an assumption about the NOISE",
        evidence="measured 1.77 and 1.78 kcal/mol under two noise models that "
                 "moved every other method -- the only one whose accuracy "
                 "barely changed, because it learns the map instead of "
                 "assuming its shape",
    ),
    Method(
        name="probabilistic error cancellation (PEC)",
        reaches=frozenset({ErrorSource.GATE_STOCHASTIC, ErrorSource.DECOHERENCE,
                           ErrorSource.COHERENT}),
        cost=Cost(0, 10.0, "quasi-probability sampling raises variance sharply"),
        assumes="the assumed noise model IS the device's noise",
        evidence="measured 2.03 kcal/mol when its assumed model matched, and "
                 "17.13 when it did not -- it does not degrade gracefully, it "
                 "inverts",
        model_dependent=True,
    ),
)

METHODS_BY_NAME = {m.name: m for m in CATALOGUE}


# ---------------------------------------------------------------------------
# The error budget
# ---------------------------------------------------------------------------
@dataclass
class ErrorBudget:
    """Where the error actually comes from, and how well that is known.

    `contributions` are in whatever unit the caller measures in -- kcal/mol,
    Hartree, or a bare fraction -- since every question asked of them is
    about ratios. What matters is that they are ATTRIBUTED, because the
    recommendation turns entirely on which term is on top.

    `provenance` is not decoration. A budget MEASURED by switching each
    noise source off is evidence; a budget ESTIMATED from calibration data
    and gate counts is an argument. Both are useful and they support
    different strengths of recommendation, so the difference is carried
    rather than flattened.
    """

    contributions: dict[ErrorSource, float]
    provenance: Provenance = Provenance.SELF_REPORTED
    note: str = ""

    def __post_init__(self) -> None:
        for source, value in self.contributions.items():
            if not isinstance(source, ErrorSource):
                raise TypeError(f"{source!r} is not an ErrorSource")
            if value < 0:
                raise ValueError(
                    f"{source.name} contributes {value}: an error budget cannot "
                    "have a negative term. If a term was not measured, leave it "
                    "out rather than recording it as less than nothing."
                )

    @property
    def total(self) -> float:
        return sum(self.contributions.values())

    @property
    def ranked(self) -> list[tuple[ErrorSource, float]]:
        return sorted(self.contributions.items(), key=lambda kv: -kv[1])

    def share(self, source: ErrorSource) -> float:
        return self.contributions.get(source, 0.0) / self.total if self.total else 0.0

    @property
    def dominant(self) -> Optional[ErrorSource]:
        return self.ranked[0][0] if self.contributions and self.total > 0 else None

    @property
    def is_decisive(self) -> bool:
        """Is one term clearly on top?

        Two terms within a factor of 1.5 do not decide between methods
        that reach one each, and pretending otherwise is where a
        recommender starts inventing confidence. When this is False the
        prescription targets the set, not the leader.
        """
        ranked = self.ranked
        if len(ranked) < 2:
            return bool(ranked)
        return ranked[1][1] <= 0.0 or ranked[0][1] / ranked[1][1] >= 1.5

    @property
    def significant(self) -> list[ErrorSource]:
        """Every term worth acting on: at least 15% of the total.

        A method that only reaches a 3% term cannot deliver more than 3%,
        and recommending it wastes a run on a rounding error.
        """
        return [s for s, v in self.ranked if self.total and v / self.total >= 0.15]

    @property
    def reachable_ceiling(self) -> float:
        """The best any mitigation could do, as a fraction of error removed.

        Shot noise and an inadequate ansatz are not mitigation's to
        remove -- one needs more shots, the other a different circuit --
        so a method that perfectly removed everything else would still
        leave these. Quoting an achievable gain that ignores them is how
        a recommender promises 20x and delivers 1.2x.
        """
        if not self.total:
            return 0.0
        unreachable = (self.contributions.get(ErrorSource.SHOT_NOISE, 0.0)
                       + self.contributions.get(ErrorSource.ANSATZ, 0.0))
        return (self.total - unreachable) / self.total

    @property
    def best_possible_gain(self) -> float:
        """The error-reduction factor if every reachable term vanished.

        An upper bound and openly a loose one: no method removes a term
        completely, and every one adds variance of its own. It is here to
        cap claims, not to make them.
        """
        floor = 1.0 - self.reachable_ceiling
        return 1.0 / floor if floor > 0 else float("inf")

    def format_budget(self) -> str:
        lines = [f"  error budget ({self.provenance.value}):"]
        for source, value in self.ranked:
            bar = "#" * max(1, round(30 * self.share(source)))
            lines.append(f"    {source.name:17s} {value:9.3f}  "
                         f"{self.share(source) * 100:5.1f}%  {bar}")
        lines.append(f"    {'TOTAL':17s} {self.total:9.3f}")
        if not self.is_decisive:
            lines.append("    (no single term dominates -- the leaders are within "
                         "1.5x, so no one method decides it)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The prescription
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Prescription:
    """One recommended action, with what backs it and what could go wrong.

    `best_case` is what this method would achieve if it removed
    everything it can reach, perfectly, AND the error terms add in
    magnitude. That second assumption is the reason this is not called a
    bound: contributions attributed by ablation can partially cancel in
    the expectation value, and a method that removes one of them can then
    beat the arithmetic. Measured here at 5.90x against a best case of
    5.65x for readout mitigation -- small, real, and a fact about the
    physics rather than an error in the sum.

    So it caps a claim rather than making one, and the ordering is what
    the prescription actually asserts.
    """

    action: str
    because: str
    #: None when the budget was estimated rather than measured.
    #:
    #: A ceiling is 1/(1 - coverage), so it depends on the budget's
    #: absolute shares -- and an estimate from calibration is only
    #: trustworthy for its RANKING, which is what the docstring of
    #: `budget_from_calibration` says and what this originally ignored.
    #: Quoting 8.2x from an estimate whose shares were off produced
    #: methods that "beat" their own bound by 2.5x. An estimate earns an
    #: ordering, not a number.
    best_case: Optional[float]
    cost: Cost
    evidence: str
    risks: tuple = ()
    #: What a supplied corpus of past audits says about this method on
    #: budgets like this one. None when no ledger was consulted.
    #:
    #: Typed loosely on purpose: `ledger.py` imports this module for the
    #: catalogue, so this module must not import it back. A ledger is
    #: anything answering `evidence_for` and `contradictions`.
    observed: object = None

    def format_lines(self) -> list[str]:
        lines = [f"  -> {self.action}",
                 f"       because: {self.because}"]
        if self.best_case is None:
            lines.append("       best case: not quoted -- this budget was estimated, "
                         "and an estimate")
            lines.append("                  supports an ordering, not a number")
        else:
            lines.append(f"       best case: {self.best_case:.1f}x, if it removes "
                         "everything it reaches and")
            lines.append("                  the error terms add in magnitude")
        lines += [f"       cost:    {self.cost}",
                  f"       basis:   {self.evidence}"]
        if self.observed is not None and getattr(self.observed, "n", 0):
            lines.append(f"       seen:    {self.observed.summarise()}")
        for risk in self.risks:
            lines.append(f"       risk:    {risk}")
        return lines


@dataclass(frozen=True)
class Consult:
    """The whole answer: what is wrong, what to do, and what will not help.

    The `will_not_help` list is not filler. Half the value of knowing
    readout error dominates is knowing not to spend three times the shots
    on an extrapolation that cannot touch it -- and that is the run
    someone would otherwise have made, because ZNE is the method everyone
    reaches for first.
    """

    budget: ErrorBudget
    prescriptions: tuple
    will_not_help: tuple
    #: Methods that reach a real but small share of the error. Kept apart
    #: from `prescriptions` because listing a 1.2x method beside a 20x one
    #: under a single heading invites someone to spend three times the
    #: shots on the wrong one.
    marginal: tuple = ()
    structural: tuple = ()
    caveats: tuple = ()

    @property
    def leading(self) -> Optional[Prescription]:
        return self.prescriptions[0] if self.prescriptions else None

    def format_consult(self) -> str:
        lines = ["=" * 70, "  WHAT TO DO ABOUT IT", "=" * 70,
                 self.budget.format_budget(), ""]
        if self.prescriptions:
            lines.append("  Recommended, best first:")
            for prescription in self.prescriptions:
                lines.extend(prescription.format_lines())
                lines.append("")
        else:
            lines.append("  No mitigation method is recommended. See below.\n")
        if self.marginal:
            lines.append("  Reaches something, but not much -- probably not worth "
                         "the run:")
            for prescription in self.marginal:
                lines.append(f"  ~  {prescription.action}: {prescription.because}")
            lines.append("")
        if self.structural:
            lines.append("  Changes to the experiment itself:")
            for action, reason in self.structural:
                lines.append(f"  -> {action}")
                lines.append(f"       {reason}")
            lines.append("")
        if self.will_not_help:
            lines.append("  Will NOT help here, and why:")
            for name, reason in self.will_not_help:
                lines.append(f"  x  {name}: {reason}")
            lines.append("")
        if self.caveats:
            lines.append("  Read this before acting on the above:")
            for caveat in self.caveats:
                lines.append(f"  !  {caveat}")
            lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)

    def print_consult(self) -> None:
        print(self.format_consult())


def _coverage(method: Method, budget: ErrorBudget) -> float:
    """The fraction of total error this method is capable of acting on."""
    if not budget.total:
        return 0.0
    return method.reach_fraction * sum(
        v for s, v in budget.contributions.items()
        if s in method.reaches) / budget.total


def _best_case(coverage: float, budget: ErrorBudget) -> Optional[float]:
    if budget.provenance is not Provenance.MEASURED:
        return None
    return 1.0 / (1.0 - coverage) if coverage < 1.0 else float("inf")


def prescribe(budget: ErrorBudget,
              noise_model_verified: bool = False,
              symmetry_available: bool = False,
              shots: Optional[int] = None,
              ledger=None) -> Consult:
    """Turn an error budget into ranked advice.

    `noise_model_verified` gates the model-dependent methods. PEC is not
    ranked below the others by preference; it is withheld unless its one
    assumption has been checked, because when that assumption is wrong it
    does not lose a little accuracy, it inverts -- 2.03 to 17.13 kcal/mol
    in the measured case.

    `symmetry_available` must be asserted by the caller, because whether
    a state obeys a checkable symmetry in the measured basis is a fact
    about the physics that no amount of looking at an error budget
    reveals.

    `ledger` is an optional corpus of past audits (see `ledger.py`). When
    supplied it is consulted, cited, and allowed to REORDER the
    recommendations -- but only when it holds enough observations on
    similar budgets to support a ranking for every candidate being
    compared. Below that the mechanism ordering stands and the
    observations are reported beside it, because a median of three runs
    that happens to disagree with the physics is three runs, not a
    finding. Which ordering was used is stated in the report rather than
    left for a reader to infer.
    """
    if not budget.contributions or budget.total <= 0:
        return Consult(
            budget=budget,
            prescriptions=(),
            will_not_help=(),
            caveats=("The error budget is empty. Nothing can be recommended from "
                     "it: go and measure where the error comes from first, by "
                     "switching each noise source off in turn if you have a "
                     "simulator, or estimating from device calibration and gate "
                     "counts if you do not.",),
        )

    caveats = []
    if budget.provenance is not Provenance.MEASURED:
        caveats.append(
            "This budget was not measured by the auditor. Every ranking below "
            "inherits that: if the attribution is wrong the recommendation is "
            "wrong, and confidently so.")
    if not budget.is_decisive:
        caveats.append(
            "No single error source dominates, so the leading recommendation is "
            "chosen on coverage of the whole significant set rather than on one "
            "term. Expect less of it than the ceiling suggests.")

    structural = []
    will_not_help = []

    # An inadequate ansatz is not mitigation's to fix, and saying so is
    # the most useful thing here when it is true.
    ansatz_share = budget.share(ErrorSource.ANSATZ)
    if ansatz_share >= 0.5:
        structural.append((
            "Change the circuit, not the post-processing.",
            f"{ansatz_share * 100:.0f}% of the error is the ansatz being unable "
            "to represent the target state. Mitigation removes noise; it cannot "
            "add expressiveness. Every method below would be spent recovering a "
            "wrong answer more precisely."))

    shot_share = budget.share(ErrorSource.SHOT_NOISE)
    if shot_share >= 0.3:
        needed = (shot_share / 0.1) ** 2
        structural.append((
            f"Take about {needed:.0f}x more shots before mitigating anything.",
            f"Shot noise is {shot_share * 100:.0f}% of the error and falls as "
            "1/sqrt(N). Mitigation does not reduce it and extrapolation "
            "amplifies it by roughly the norm of the fit coefficients, so "
            "mitigating first makes this term worse, not better."))

    ranked = []
    for method in CATALOGUE:
        coverage = _coverage(method, budget)
        if method.name == "symmetry verification (post-selection)" and not symmetry_available:
            will_not_help.append((
                method.name,
                "no checkable symmetry was declared for this state; post-selection "
                "needs one that is visible in the measured basis"))
            continue
        if method.model_dependent and not noise_model_verified:
            will_not_help.append((
                method.name,
                "withheld: its correctness rests entirely on the assumed noise "
                "model being the real one, and that has not been verified here. "
                "When the assumption fails this method does not degrade, it "
                "inverts -- measured at 2.03 -> 17.13 kcal/mol"))
            continue
        if method.name == "more shots" and shot_share < 0.15:
            will_not_help.append((
                method.name,
                f"shot noise is only {shot_share * 100:.0f}% of the error; more "
                "shots cannot fix what is not statistical"))
            continue
        if coverage < 0.15:
            unreached = [s.name for s in budget.significant if s not in method.reaches]
            will_not_help.append((
                method.name,
                f"reaches only {coverage * 100:.0f}% of the error here. "
                f"{', '.join(unreached)} "
                f"{'is' if len(unreached) == 1 else 'are'} outside what it can act "
                f"on: {SCALES_AS[budget.dominant].value}"))
            continue
        ranked.append((method, coverage))

    # Best coverage first; among equals, the cheaper one. Coverage is
    # rounded before comparison so that two methods reaching the same
    # terms are not separated by floating-point dust.
    ranked.sort(key=lambda mc: (-round(mc[1], 6),
                                mc[0].cost.extra_circuits,
                                mc[0].cost.shot_multiplier))

    #: A method reaching less than this cannot deliver much even if it
    #: works perfectly. 0.4 corresponds to a best case of about 1.7x --
    #: below that the run usually costs more than the answer improves.
    WORTH_A_RUN = 0.4

    observations = {}
    if ledger is not None:
        for method, _ in ranked:
            observations[method.name] = ledger.evidence_for(method.name, budget)
        for name, why in ledger.contradictions(budget):
            caveats.append(f"The corpus disagrees with the catalogue about "
                           f"{name}: {why}. That is a finding about this "
                           f"package's own mechanism table, not a rounding "
                           f"error, and it is reported rather than averaged in.")

    prescriptions, marginal = [], []
    for method, coverage in ranked:
        reached = [s.name for s, _ in budget.ranked
                   if s in method.reaches and budget.share(s) >= 0.05]
        risks = []
        if method.model_dependent:
            risks.append("correctness depends on the assumed noise model being right")
        if method.cost.shot_multiplier >= 3.0:
            risks.append(f"amplifies shot noise; budget for {method.cost}")
        missed = [s.name for s in budget.significant if s not in method.reaches]
        if missed:
            risks.append(f"leaves {', '.join(missed)} untouched "
                         f"({(1 - coverage) * 100:.0f}% of the error)")
        prescription = Prescription(
            action=method.name,
            because=(f"reaches {', '.join(reached)}, which is "
                     f"{coverage * 100:.0f}% of the error here"
                     + ("" if method.reach_fraction == 1.0 else
                        f" (only {method.reach_fraction:.0%} of each term is "
                        f"within its reach)")),
            best_case=_best_case(coverage, budget),
            cost=method.cost,
            evidence=method.evidence,
            risks=tuple(risks),
            observed=observations.get(method.name),
        )
        (prescriptions if coverage >= WORTH_A_RUN else marginal).append(prescription)

    # Observations outrank mechanism only when there are enough of them
    # for every method being compared. A partial corpus reordering a
    # ranking would let the best-studied method win rather than the best
    # one.
    if prescriptions and observations:
        relevant = [observations.get(p.action) for p in prescriptions]
        if all(e is not None and e.supports_a_ranking for e in relevant):
            prescriptions.sort(key=lambda p: -observations[p.action].median_gain)
            caveats.append(
                "Ordered by what past audits measured on budgets like this one, "
                "not by mechanism: the corpus holds enough observations on every "
                "method compared here to support a ranking.")

    return Consult(budget=budget,
                   prescriptions=tuple(prescriptions),
                   will_not_help=tuple(will_not_help),
                   marginal=tuple(marginal),
                   structural=tuple(structural),
                   caveats=tuple(caveats))


# ---------------------------------------------------------------------------
# Getting a budget when you cannot switch the noise off
# ---------------------------------------------------------------------------
def budget_from_calibration(*,
                            two_qubit_gates: int,
                            one_qubit_gates: int,
                            measured_qubits: int,
                            two_qubit_error: float,
                            one_qubit_error: float,
                            readout_error: float,
                            shots: int,
                            circuit_duration_s: Optional[float] = None,
                            t2_s: Optional[float] = None) -> ErrorBudget:
    """Estimate the budget from device calibration and circuit structure.

    On a simulator the honest way to attribute error is to switch each
    source off and watch what happens. On hardware nobody can do that,
    and nobody has the exact answer to compare against either -- which is
    the whole reason the experiment is being run. So the budget has to
    come from what IS knowable there: the calibration data the provider
    publishes, and the gate counts of the circuit as submitted.

    Each term is the fraction of signal that mechanism is expected to
    cost, so they share a scale and their RATIOS are meaningful. The
    absolute numbers are order-of-magnitude and should not be quoted as
    predicted errors -- the prescription only ever asks which term is on
    top, and by how much.

    Marked SELF_REPORTED rather than MEASURED, deliberately: this is an
    argument from a model, not an observation. `prescribe` carries that
    through into a caveat on every recommendation built from it.
    """
    for name, value in (("two_qubit_error", two_qubit_error),
                        ("one_qubit_error", one_qubit_error),
                        ("readout_error", readout_error)):
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name}={value} is not an error probability")
    if shots <= 0:
        raise ValueError("shots must be positive")

    contributions = {
        # Probability that at least one gate erred, which is what costs
        # signal in an expectation value.
        ErrorSource.GATE_STOCHASTIC: 1.0 - (
            (1.0 - two_qubit_error) ** two_qubit_gates
            * (1.0 - one_qubit_error) ** one_qubit_gates),
        # Readout is charged once per measured qubit and does not care
        # how many gates ran before it -- which is exactly why folding
        # cannot reach it.
        ErrorSource.READOUT: 1.0 - (1.0 - readout_error) ** measured_qubits,
        # Standard error as a FRACTION of the observable's scale, so it
        # shares units with the terms above. Quoting it as an absolute
        # expectation-value error instead put one term on a different
        # scale from the rest and inflated its share -- caught by
        # examples/prescribe_for_circuit.py checking a prescription
        # against what the methods actually delivered.
        ErrorSource.SHOT_NOISE: 1.0 / (shots ** 0.5),
    }
    if circuit_duration_s is not None and t2_s:
        import math

        contributions[ErrorSource.DECOHERENCE] = 1.0 - math.exp(
            -circuit_duration_s / t2_s)

    return ErrorBudget(
        contributions={k: v for k, v in contributions.items() if v > 0},
        provenance=Provenance.SELF_REPORTED,
        note="estimated from calibration and gate counts, not measured by ablation",
    )


def residual_budget(budget: ErrorBudget, method: "Method",
                    effectiveness: float = 0.9) -> ErrorBudget:
    """What is left over after a method runs.

    Needed because choosing qubits and choosing a mitigation method are
    not independent, which is easy to miss and expensive to get wrong.
    Measured case: on fake_kyiv the lowest-readout pair beats the
    lowest-gate-error pair 13.87 to 36.46 kcal/mol unmitigated -- and
    then LOSES 11.02 to 6.18 once readout mitigation is applied, because
    REM removes the readout error that made the first pair attractive and
    leaves the gate error where it is 2.2x worse.

    So a placement chosen against the raw budget is the right answer only
    for someone running raw. Anyone planning to mitigate should choose
    against what their method will leave behind, which is what this
    returns.

    `effectiveness` is how much of a reachable term the method actually
    removes. It is deliberately below 1: no method removes a term
    completely, and a residual budget that zeroed a source would claim
    the placement no longer cares about it at all.
    """
    if not 0.0 < effectiveness <= 1.0:
        raise ValueError(f"effectiveness={effectiveness} is not a fraction")
    removed = method.reach_fraction * effectiveness
    return ErrorBudget(
        contributions={
            source: value * (1.0 - removed) if source in method.reaches else value
            for source, value in budget.contributions.items()},
        provenance=budget.provenance,
        note=f"what remains after {method.name}",
    )


def budget_from_ablation(measurements: dict,
                         note: str = "") -> ErrorBudget:
    """Build a budget from errors measured with each source switched off.

    `measurements` maps ErrorSource to the error attributable to it --
    typically total error minus the error with that source disabled. This
    is the strong form, available in simulation, and it is what the
    calibration estimate above should be checked against before anyone
    trusts the estimate on hardware.
    """
    return ErrorBudget(contributions=dict(measurements),
                       provenance=Provenance.MEASURED,
                       note=note or "measured by switching each noise source off")


@dataclass(frozen=True)
class Feasibility:
    """Whether there is a signal left for a method to mitigate.

    Every mitigation method in the catalogue improves an estimate that
    still exists. None of them creates one. A circuit deep enough that
    the probability of finishing without an error is 1e-8 does not have a
    small signal-to-noise problem; it has no signal, and a prescription
    ranking methods for it would be answering a question that cannot be
    asked.

    This came from an 18-qubit oracle whose correct implementation needs
    about 5,900 two-qubit gates. On a device with 0.31% two-qubit error
    that is a survival probability of 1e-8: a hundred million shots to
    see one uncorrupted run. The useful output there is not "use CDR", it
    is "this cannot run, and here is the gate count that would let it".
    """

    two_qubit_gates: int
    two_qubit_error: float
    readout_error: float
    n_qubits: int

    @property
    def gate_survival(self) -> float:
        return (1.0 - self.two_qubit_error) ** self.two_qubit_gates

    @property
    def readout_survival(self) -> float:
        return (1.0 - self.readout_error) ** self.n_qubits

    @property
    def survival(self) -> float:
        """Probability a shot finishes uncorrupted. Optimistic on purpose:
        it counts only gate and readout error, ignoring decoherence and
        crosstalk, so a circuit this call calls hopeless really is."""
        return self.gate_survival * self.readout_survival

    @property
    def shots_for_a_signal(self) -> float:
        """Roughly what it takes to resolve the surviving signal at 3 sigma.

        The surviving fraction sets the effective sample size, so the
        shots needed grow as 1/survival**2. Quoted to an order of
        magnitude, which is all it deserves."""
        return 9.0 / max(self.survival, 1e-300) ** 2

    @property
    def is_mitigable(self) -> bool:
        """Below a 1% survival there is no estimate worth correcting.

        The threshold is a judgement, and it is placed where mitigation
        stops being the binding constraint rather than where it stops
        working perfectly: at 1% survival a method that halved the
        remaining error would still leave an answer dominated by the
        circuit never having finished.
        """
        return self.survival >= 0.01

    @property
    def affordable_two_qubit_gates(self) -> int:
        """The gate count that would put this back above the threshold."""
        import math
        if self.two_qubit_error <= 0:
            return self.two_qubit_gates
        budget = math.log(0.01 / max(self.readout_survival, 1e-300))
        return max(0, int(budget / math.log(1.0 - self.two_qubit_error)))

    def format_verdict(self) -> str:
        lines = [
            f"  two-qubit gates:  {self.two_qubit_gates}",
            f"  survival:         {self.survival:.3g} "
            f"(gates {self.gate_survival:.3g}, readout {self.readout_survival:.3f})",
        ]
        if self.is_mitigable:
            lines.append("  -> there is a signal here; mitigation is the right question")
        else:
            lines.append(
                f"  -> no method mitigates this: it would take about "
                f"{self.shots_for_a_signal:.1g} shots to see the signal at all.")
            lines.append(
                f"     Getting under {self.affordable_two_qubit_gates} two-qubit "
                "gates is the prerequisite, and it is a compilation problem, "
                "not a mitigation one.")
        return "\n".join(lines)


def feasibility(two_qubit_gates: int, calibration: dict,
                n_qubits: int) -> Feasibility:
    """Is there a signal left to mitigate? Ask before ranking methods."""
    return Feasibility(
        two_qubit_gates=two_qubit_gates,
        two_qubit_error=calibration.get("ecr_error", calibration.get("cx_error", 0.0)),
        readout_error=calibration.get("readout_error", 0.0),
        n_qubits=n_qubits,
    )
