"""Core experiment record schema.

An Experiment is the unit of evidence the auditor reasons over. It does
NOT store a verdict -- verdicts are computed by gates.py/verdict.py from
the record's controls and outputs, never asserted by whoever creates the
record. That separation is the whole point: a claim's author should not
be the one who gets to say whether it passed.

Every field here exists because a real experiment in this project's
history needed it. The three that matter most, and that a naive schema
gets wrong:

1. `Replicate.kind` -- bootstrap-resampling ONE submission's counts is
   not replication. The H4 project ran the identical circuit set through
   IonQ's simulator twice and got energies 3.27 kcal/mol apart, three
   times its own reproducibility bar, while its 8-seed bootstrap bars
   (which only ever resampled a single submission) showed nothing of the
   kind. A record that cannot distinguish the two will certify drift as
   agreement.

2. `UncertaintyCoverage` -- what the uncertainty bar actually varied. A number
   that resamples shot noise says nothing about calibration uncertainty.
   The project's 0.115 kcal/mol headline was disowned exactly this way:
   real under one fixed noise model, Q95=51.22 once the noise model's own
   parameters were allowed to vary.

3. `ClaimType` -- "this method beats that method" and "this method
   reaches chemical accuracy" are different claims needing different
   evidence. The joint Schmidt frame is a large, real, validated
   improvement (88.8% MSE reduction) that still does not reach chemical
   accuracy. Grading it against an absolute bar would wrongly call a real
   result a failure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ReplicateKind(Enum):
    """How independent a replicate actually is, weakest first.

    Only INDEPENDENT_SUBMISSION counts toward a replication target: it is
    the only kind that re-executes the experiment and can therefore see
    submission-to-submission drift.
    """

    BOOTSTRAP_RESAMPLE = "bootstrap resample of one submission's counts"
    SEED_SPLIT = "split-half / different analysis seed, same underlying counts"
    INDEPENDENT_SUBMISSION = "independently executed submission"

    @property
    def is_independent(self) -> bool:
        return self is ReplicateKind.INDEPENDENT_SUBMISSION


@dataclass
class UncertaintyCoverage:
    """What a stated uncertainty bar actually varied.

    Deliberately a set of independent flags, not a ladder. The H4 history
    shows these axes come apart: the joint Schmidt frame was validated
    through a randomized noise-model envelope (29 draws) while never
    including cross-submission drift, and this project's earlier headline
    numbers had the opposite gap -- independent submissions, one fixed
    noise model. Ranking them on a single scale would call one of those
    strictly stronger than the other, which is false.
    """

    shot_noise: bool = False
    """Resampling the measured counts."""

    method_monte_carlo: bool = False
    """The method's own sampling -- PEC draws, twirls, optimizer restarts.
    Dominated the real variance budget by 9-570x over every other source."""

    cross_submission: bool = False
    """Re-executed submissions, so run-to-run platform drift is included."""

    noise_model: bool = False
    """The assumed calibration/noise parameters were themselves varied."""

    @property
    def covered(self) -> list[str]:
        return [n for n in ("shot_noise", "method_monte_carlo", "cross_submission",
                            "noise_model") if getattr(self, n)]

    @property
    def missing(self) -> list[str]:
        return [n for n in ("shot_noise", "method_monte_carlo", "cross_submission",
                            "noise_model") if not getattr(self, n)]

    @property
    def is_complete(self) -> bool:
        return not self.missing

    def describe(self) -> str:
        return ", ".join(self.covered) if self.covered else "nothing"


class ClaimType(Enum):
    """What is actually being claimed. Different claims need different gates."""

    ABSOLUTE_ACCURACY = "this method reaches a stated absolute accuracy"
    RELATIVE_IMPROVEMENT = "this method beats a stated baseline method"


class Provenance(Enum):
    """Where a control's value came from.

    This is the difference between a rubric and a verifier. A control the
    claimant asserts is a statement about their own work; a control the
    auditor executed is evidence. Both are recorded, and they are never
    conflated -- an auditor that trusts the claimant is not a gate.
    """

    SELF_REPORTED = "asserted by whoever wrote the record"
    MEASURED = "executed by the auditor against the claimant's artifacts"


class TranspilationStatus(Enum):
    """Whether the circuit that RAN is the circuit that was designed.

    The H4 project lost an entire ZNE result to this: abstract-gate
    folding inserted G.G^-1 pairs that the compiler optimized straight
    back out before submission. The locally-built circuit was correct;
    the executed one was not the same circuit.
    """

    UNVERIFIED = "submitted circuit never checked against the intended circuit"
    VERIFIED_EQUIVALENT = "submitted circuit verified to match the intended unitary"
    VERIFIED_MODIFIED = "submitted circuit verified to DIFFER from the intended circuit"


class FailureMode(Enum):
    """Root-cause taxonomy, taken from this project's own retrospective of
    every disqualification and real bug it recorded. These are not
    hypothetical categories: each one disqualified a real result."""

    COMPILER_CANCELLATION = "inserted gates optimized back out before execution"
    EXTRAPOLATION_INSTABILITY = "estimator extrapolates outside its validated domain"
    TARGET_LEAKAGE = "the known exact answer influenced tuning or selection"
    FREE_PARAMETER_DEGENERACY = "a free parameter has no floor, degenerating toward the target"
    MONTE_CARLO_VARIANCE = "the method's own Monte Carlo sampling dominates the error"
    OPTIMIZER_INSTABILITY = "nonconvex fit lands in different local optima"
    NONDETERMINISM = "identical inputs produce different outputs across runs"
    CALIBRATION_MISMATCH = "assumed noise parameters differ from the true ones"
    DRIFT = "submission-to-submission variation on nominally identical runs"
    HEAVY_TAIL = "rare catastrophic outliers dominate the risk"
    STRUCTURAL_NONIDENTIFIABILITY = "the data cannot determine the parameters, even in principle"
    SIGN_CONVENTION = "a sign or bookkeeping convention was applied inconsistently"
    BOOKKEEPING = "an indexing, unit, or normalization error"
    UNDER_POWERED = "not enough data collected to support the claim"
    UNKNOWN = "not yet root-caused"


@dataclass
class Replicate:
    """One repetition of an experiment, tagged with how independent it is.

    `error_kcal` may be None: a replicate that was executed but whose
    value is withheld (a blinded challenge) or not yet transcribed is
    still a replicate, and the count and kind are real evidence about
    method even when the number is unavailable. Gates that need values
    report "not enough to judge" rather than treating a withheld value as
    zero -- which is the whole reason this is None and not a sentinel
    number.
    """

    error_kcal: Optional[float] = None
    kind: ReplicateKind = ReplicateKind.INDEPENDENT_SUBMISSION
    source_id: str = ""


@dataclass
class CircuitSpec:
    """What was actually executed, as opposed to what was designed."""

    circuit_id: str = ""
    native_gate_set: str = ""
    transpilation_status: TranspilationStatus = TranspilationStatus.UNVERIFIED
    optimization_level: Optional[int] = None
    n_1q_gates: Optional[int] = None
    n_2q_gates: Optional[int] = None
    n_qubits: Optional[int] = None


@dataclass
class NoiseSpec:
    """Where the noise model came from, and whether its own uncertainty
    was propagated. A noise model accurate for raw expectation values is
    not automatically accurate for a mitigation built on top of it."""

    noise_model: str = ""
    calibration_source: str = ""
    calibration_uncertainty_propagated: bool = False


@dataclass
class Controls:
    """Each field is None (not yet run), True (passed), or False (failed).
    'Passed' for an adversarial/negative control means it behaved as a
    genuine effect should: the perturbation destroyed the result."""

    ideal_control: Optional[bool] = None
    target_leakage_check: Optional[bool] = None
    adversarial_check: Optional[bool] = None
    reproducibility_checked: bool = False

    # Controls added in Phase 3, each from a real historical failure.
    unitary_equivalence: Optional[bool] = None
    """Does the SUBMITTED circuit implement the intended unitary?"""

    heldout_check: Optional[bool] = None
    """Was the method validated against data it did not see while fitting?"""

    extrapolation_in_domain: Optional[bool] = None
    """Was the held-out validation testing the SAME direction production
    uses? Validating interpolation and then extrapolating in production is
    how a 0.065 kcal/mol error became 33.48."""

    free_parameter_floor_test: Optional[bool] = None
    """Does every free parameter have a floor, or can it degenerate toward
    re-evaluating the known answer?"""

    determinism_check: Optional[bool] = None
    """Does re-running the identical computation on identical inputs give
    an identical result?"""

    provenance: dict[str, Provenance] = field(default_factory=dict)
    """Per-control: SELF_REPORTED (the default for anything absent) or
    MEASURED. Set by adapters when the auditor runs a control itself."""

    def provenance_of(self, control: str) -> Provenance:
        return self.provenance.get(control, Provenance.SELF_REPORTED)

    def record_measured(self, control: str, value: Optional[bool]) -> None:
        """Sets a control from the auditor's own execution, marking it as
        measured. Refuses unknown names so a typo cannot silently create a
        control nothing reads."""
        if not hasattr(self, control) or control == "provenance":
            raise AttributeError(f"no such control: {control}")
        setattr(self, control, value)
        self.provenance[control] = Provenance.MEASURED

    @property
    def measured_controls(self) -> list[str]:
        return sorted(k for k, v in self.provenance.items() if v is Provenance.MEASURED)


@dataclass
class Outputs:
    raw_error_kcal: Optional[float] = None
    mitigated_error_kcal: Optional[float] = None
    replicates: list[Replicate] = field(default_factory=list)
    q95_kcal: Optional[float] = None
    n_replicates_target: int = 8  # this project's own established replication convention

    # Distribution shape, not just central tendency. "Accuracy vs
    # reliability" is the distinction this project's error budget has
    # turned on since iteration 31: a median of 0.29 kcal/mol with a Q95
    # of 1.29 is not a result that clears a 0.5 bar.
    q50_kcal: Optional[float] = None
    q99_kcal: Optional[float] = None
    n_trials: Optional[int] = None
    n_outlier_trials: Optional[int] = None
    """Trials whose error was catastrophically large. Heavy tails killed
    cross-fitting (2/32 blowups at ~2100 kcal/mol) even though its median
    was fine."""

    uncertainty: Optional[UncertaintyCoverage] = None
    """What the stated uncertainty bar actually varied."""

    # Relative claims are graded against a baseline, not an absolute bar.
    baseline_error_kcal: Optional[float] = None
    baseline_label: str = ""

    cost_usd: Optional[float] = None
    runtime_seconds: Optional[float] = None

    @property
    def replicate_errors_kcal(self) -> list[float]:
        """Only the replicates that actually carry a value."""
        return [r.error_kcal for r in self.replicates
                if r.error_kcal is not None and math.isfinite(r.error_kcal)]

    @property
    def independent_replicates(self) -> list[Replicate]:
        return [r for r in self.replicates if r.kind.is_independent]

    @property
    def independent_errors_kcal(self) -> list[float]:
        return [r.error_kcal for r in self.independent_replicates
                if r.error_kcal is not None and math.isfinite(r.error_kcal)]

    @property
    def outlier_fraction(self) -> Optional[float]:
        if self.n_trials in (None, 0) or self.n_outlier_trials is None:
            return None
        return self.n_outlier_trials / self.n_trials


@dataclass
class Experiment:
    experiment_id: str
    description: str
    backend: str
    shots: int
    controls: Controls
    outputs: Outputs
    real_hardware_full_validation: bool = False
    notes: str = ""

    claim: str = ""
    claim_type: ClaimType = ClaimType.ABSOLUTE_ACCURACY
    circuit: CircuitSpec = field(default_factory=CircuitSpec)
    noise: NoiseSpec = field(default_factory=NoiseSpec)
    suspected_failure_modes: list[FailureMode] = field(default_factory=list)
    """Only ever a hypothesis supplied with the record. The auditor's own
    classifier (failure_modes.py) derives its findings independently and
    is never handed this list."""
