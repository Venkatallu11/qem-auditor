"""Hard gates -- each one returns a GateResult, never a bare bool, so the
verdict layer can explain *why* something failed, not just that it did.

These are deliberately simple, auditable functions over an Experiment
record. No LLM involvement here on purpose: an AI should never be the
thing that decides whether a scientific claim passed. It can propose
experiments and interpret results in prose, but the gates are plain,
inspectable code.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Optional

from .schema import Experiment


@dataclass
class GateResult:
    name: str
    passed: Optional[bool]  # None = not applicable / not enough data to judge
    reason: str


def ideal_control_gate(exp: Experiment) -> GateResult:
    """If a noiseless/ideal-model run of the SAME production code doesn't
    recover a near-zero (or at least non-degraded) result, the method is
    broken independent of any real hardware noise -- an automatic
    disqualifier. (This is exactly what caught the 513x ZNE blowup.)"""
    ic = exp.controls.ideal_control
    if ic is None:
        return GateResult("ideal_control", None, "not yet run")
    if ic is False:
        return GateResult("ideal_control", False,
                           "ideal/noiseless control did NOT recover a sane result -- "
                           "the method is broken independent of real hardware noise")
    return GateResult("ideal_control", True, "ideal control passed")


def target_leakage_gate(exp: Experiment) -> GateResult:
    tl = exp.controls.target_leakage_check
    if tl is None:
        return GateResult("target_leakage", None, "not yet run")
    if tl is False:
        return GateResult("target_leakage", False,
                           "evidence of target leakage -- the known exact answer was used "
                           "to select or tune the method")
    return GateResult("target_leakage", True, "no target leakage detected")


def adversarial_gate(exp: Experiment) -> GateResult:
    """Passed means the adversarial/negative controls (wrong parity,
    shuffled labels, wrong sign, etc.) DID fail loudly, as a genuine
    effect requires. If they did NOT fail -- i.e. garbage-in produced a
    similarly good-looking result -- the claim is not trustworthy."""
    adv = exp.controls.adversarial_check
    if adv is None:
        return GateResult("adversarial", None, "not yet run")
    if adv is False:
        return GateResult("adversarial", False,
                           "adversarial/negative controls did NOT fail as required -- "
                           "the result may be an artifact, not a genuine effect")
    return GateResult("adversarial", True, "adversarial controls failed loudly, as required")


def reproducibility_gate(exp: Experiment, rel_tolerance: float = 0.5) -> GateResult:
    """Checks two things: (1) do independent replicates agree with each
    other within a loose relative tolerance, and (2) have ENOUGH
    replicates been collected to call this established, per this
    project's own convention (n_replicates_target, default 8)."""
    reps = exp.outputs.replicate_errors_kcal
    n_target = exp.outputs.n_replicates_target
    if not exp.controls.reproducibility_checked or len(reps) < 2:
        return GateResult("reproducibility", None,
                           f"insufficient replicates to judge ({len(reps)} collected)")
    m = mean(reps)
    spread = pstdev(reps) if len(reps) > 1 else 0.0
    agree = spread <= rel_tolerance * m if m > 0 else spread == 0.0
    if not agree:
        return GateResult("reproducibility", False,
                           f"{len(reps)} replicates disagree beyond tolerance "
                           f"(mean={m:.4f}, spread={spread:.4f} kcal/mol)")
    note = "" if len(reps) >= n_target else f" -- below this project's own {n_target}-replicate target"
    return GateResult("reproducibility", True,
                       f"{len(reps)}/{n_target} replicates collected and mutually consistent "
                       f"(mean={m:.4f} kcal/mol){note}")


def chemical_accuracy_gate(exp: Experiment, threshold_kcal: float = 0.25) -> GateResult:
    """Informational, not a hard fail: reports whether the best available
    uncertainty estimate (Q95 if present, else the point mitigated error)
    clears the standard 0.25 kcal/mol chemical-accuracy bar."""
    value = exp.outputs.q95_kcal if exp.outputs.q95_kcal is not None else exp.outputs.mitigated_error_kcal
    if value is None:
        return GateResult("chemical_accuracy", None, "no mitigated error or Q95 recorded")
    passed = value < threshold_kcal
    basis = "Q95" if exp.outputs.q95_kcal is not None else "point estimate"
    return GateResult("chemical_accuracy", passed,
                       f"{basis}={value:.4f} kcal/mol vs {threshold_kcal} kcal/mol target")


ALL_GATES = [ideal_control_gate, target_leakage_gate, adversarial_gate,
             reproducibility_gate, chemical_accuracy_gate]
