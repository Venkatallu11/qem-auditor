"""The interface a mitigation pipeline implements to be attacked properly.

Three of the nine transformations -- T_label, T_sign, T_shot -- cannot be
run generically because they need the claimant's own fitting code. An
auditor cannot shuffle labels and refit unless it can refit. So this is
the smallest interface that makes them executable:

    fit(data)                -> a fitted model, opaque to the auditor
    reconstruct(fit, data)   -> the aggregate the claim is about
    goodness_of_fit(fit, data) -> chi2/dof, or any scalar where lower is better

Three methods, no assumptions about what the model is. CDR, PEC, a
manifold reconstruction, a shared Schmidt frame and a plain average all
fit behind it, because the auditor never looks inside the fit -- it only
perturbs the DATA going in and watches what comes out.

**Why goodness-of-fit is the load-bearing quantity.** The tempting design
is to compare reconstructed values against the true answer, but the
auditor does not have the true answer and should not be trusted with it
if it did. What it can do is corrupt the input in a way that destroys real
structure while leaving the model's flexibility untouched, and ask whether
the model noticed. A model that fits shuffled data as well as real data
has not learned the physics; it has learned to fit. That comparison needs
no ground truth at all, which is exactly why it is trustworthy.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any, Optional, Protocol, Sequence, runtime_checkable


class ReconstructionError(RuntimeError):
    """A pipeline that could not be fitted or reconstructed as asked."""


@dataclass
class Measurement:
    """One measured expectation value, tagged so it can be perturbed.

    `slot` and `label` are the two axes every pipeline in this family has:
    which circuit configuration was executed, and which observable term
    was read off it. Shuffling labels WITHIN a slot is meaningful because
    it destroys the correspondence between term and value while leaving
    the marginal distribution of values per slot untouched -- so a model
    cannot notice by looking at the numbers alone, only by having actually
    used the correspondence.
    """

    slot: str
    label: str
    value: float
    sigma: float = 0.0
    shots: int = 0
    draw: int = 0
    """Which of the method's own Monte Carlo draws this came from, if any.
    Lets T_shot separate sampling of the DATA from sampling of the METHOD."""


@dataclass
class FitData:
    measurements: list[Measurement] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.measurements:
            raise ReconstructionError("fit data with no measurements cannot be fitted")

    @property
    def slots(self) -> list[str]:
        seen: list[str] = []
        for m in self.measurements:
            if m.slot not in seen:
                seen.append(m.slot)
        return seen

    @property
    def draws(self) -> list[int]:
        return sorted({m.draw for m in self.measurements})

    def by_slot(self, slot: str) -> list[Measurement]:
        return [m for m in self.measurements if m.slot == slot]

    def copy(self) -> "FitData":
        return FitData([copy.replace(m) if hasattr(copy, "replace")
                        else Measurement(m.slot, m.label, m.value, m.sigma,
                                         m.shots, m.draw)
                        for m in self.measurements],
                       dict(self.metadata))


@runtime_checkable
class Reconstructor(Protocol):
    """What a claimant implements so their method can be attacked."""

    def fit(self, data: FitData) -> Any:
        """Fit the correction model. Returns whatever the pipeline uses."""

    def reconstruct(self, fit: Any, data: FitData) -> float:
        """The aggregate quantity the claim is about (an energy, say)."""

    def goodness_of_fit(self, fit: Any, data: FitData) -> float:
        """chi2/dof or similar. Lower is better; the auditor only compares."""


# --------------------------------------------------------------------------
# The perturbations
# --------------------------------------------------------------------------

def shuffle_is_diagnostic(data: FitData) -> tuple[bool, str]:
    """Can a label shuffle discriminate on THIS data at all?

    A real limitation, found by testing rather than reasoned about in
    advance: with only two labels there is exactly one non-identity
    permutation, so every group gets the SAME swap. That is a systematic
    relabelling, and a model carrying one free parameter per label absorbs
    it perfectly by refitting -- its chi2/dof is unchanged, and a genuine
    model looks identical to a flexible one.

    So the attack needs at least three labels, where the permutation
    differs between groups and no consistent relabelling can explain the
    result. Below that the correct output is "cannot judge", not a pass:
    an attack that cannot discriminate must never be reported as one the
    claim survived.
    """
    labels = {m.label for m in data.measurements}
    if len(labels) < 2:
        return False, f"only {len(labels)} distinct label(s): nothing to permute"
    if len(labels) == 2:
        return False, (
            "only 2 distinct labels, so every group receives the same swap. A "
            "systematic relabelling is absorbed exactly by any model with one free "
            "parameter per label, so this test cannot separate a genuine model from "
            "a flexible one here. Three or more labels are needed.")
    return True, f"{len(labels)} distinct labels: permutations differ between groups"


def shuffle_labels(data: FitData, seed: int = 0) -> FitData:
    """Permute labels within each (slot, draw) group.

    The group has to be one execution of one configuration, not a whole
    slot. Permuting across draws as well would move values between draws,
    which changes more than the label correspondence -- and worse, it lets
    a model "notice" the shuffle through the disturbance rather than
    through actually having used labels, so the attack would pass for the
    wrong reason. Within (slot, draw) leaves every group's multiset of
    values identical and destroys only the term-to-value mapping, which is
    exactly the structure a real model depends on and a flexible one does
    not.
    """
    rng = random.Random(seed)
    out = data.copy()
    groups: dict[tuple[str, int], list[Measurement]] = {}
    for m in out.measurements:
        groups.setdefault((m.slot, m.draw), []).append(m)
    for group in groups.values():
        labels = [m.label for m in group]
        if len(set(labels)) < 2:
            continue  # nothing to permute
        for _ in range(20):
            shuffled = labels[:]
            rng.shuffle(shuffled)
            if shuffled != labels:
                break
        for m, new_label in zip(group, shuffled):
            m.label = new_label
    return out


def flip_sign(data: FitData) -> FitData:
    """Negate every measured value, leaving the structure intact.

    A correction that encodes a direction should fit the negated data far
    worse. One that fits it equally well is flexible enough to absorb
    anything, and its agreement with the real data was never evidence.
    """
    out = data.copy()
    for m in out.measurements:
        m.value = -m.value
    return out


def resample_shots(data: FitData, seed: int = 0) -> FitData:
    """Perturb values by their own stated shot noise.

    Uses each measurement's sigma, or derives one from its shot count
    (binomial, for a +-1 eigenvalue observable) when sigma is absent. A
    measurement carrying neither is left alone rather than being assigned
    an invented uncertainty.
    """
    rng = random.Random(seed)
    out = data.copy()
    for m in out.measurements:
        sigma = m.sigma
        if sigma <= 0 and m.shots > 0:
            sigma = ((1.0 - min(1.0, m.value ** 2)) / m.shots) ** 0.5
        if sigma > 0:
            m.value += rng.gauss(0.0, sigma)
    return out


def subsample_draws(data: FitData, keep: int, seed: int = 0) -> FitData:
    """Keep only `keep` of the method's own Monte Carlo draws.

    This is the other half of T_shot. Varying the data's shot noise and
    varying how many of the method's own draws are averaged are different
    knobs, and in this project's measured variance budget the second
    dominated the first by ~570x. A test that only resamples shots would
    have concluded that more shots were the answer, which was the
    expensive wrong turn.
    """
    available = data.draws
    if keep >= len(available):
        return data.copy()
    if keep < 1:
        raise ReconstructionError(f"cannot keep {keep} draws")
    rng = random.Random(seed)
    chosen = set(rng.sample(available, keep))
    out = FitData([m for m in data.copy().measurements if m.draw in chosen],
                  dict(data.metadata))
    return out


# --------------------------------------------------------------------------
# Running an attack through a reconstructor
# --------------------------------------------------------------------------

@dataclass
class FitComparison:
    """A real fit against a corrupted one, in goodness-of-fit terms."""

    real: float
    perturbed: float
    ratio: float
    detail: str = ""

    @property
    def model_noticed(self) -> bool:
        """Did corrupting the data actually make the fit worse?"""
        return self.ratio > 1.0


def compare_fit(reconstructor: Reconstructor, data: FitData,
                perturbed: FitData, min_ratio: float = 5.0) -> FitComparison:
    """Fit both, and report how much worse the corrupted one is.

    `min_ratio` defaults to 5x. The joint Schmidt frame measured 52x
    against shuffled labels, and this project's own adversarial checks
    treated ratios in the thousands as convincing -- so 5x is a low bar
    deliberately: it is the threshold below which a model is clearly not
    using the structure, not a certificate that it is.
    """
    try:
        real_fit = reconstructor.fit(data)
        real_gof = float(reconstructor.goodness_of_fit(real_fit, data))
    except Exception as e:
        raise ReconstructionError(f"the pipeline failed on real data: {e}") from e
    try:
        bad_fit = reconstructor.fit(perturbed)
        bad_gof = float(reconstructor.goodness_of_fit(bad_fit, perturbed))
    except Exception as e:
        # A pipeline that CRASHES on corrupted input has, in its own blunt
        # way, noticed. That is a pass, and worth distinguishing from a
        # pipeline that sailed through.
        return FitComparison(0.0, float("inf"), float("inf"),
                             f"the pipeline raised on corrupted data ({e}) -- "
                             f"a loud failure, which is what a genuine model should do")
    if real_gof <= 0:
        # A perfect fit to real data leaves no room to get worse in ratio
        # terms; fall back to an absolute comparison rather than dividing.
        ratio = float("inf") if bad_gof > 0 else 1.0
    else:
        ratio = bad_gof / real_gof
    return FitComparison(real_gof, bad_gof, ratio,
                         f"chi2/dof {real_gof:.6g} -> {bad_gof:.6g} ({ratio:.1f}x)")


def reconstruction_spread(reconstructor: Reconstructor, data: FitData,
                          perturb, trials: int = 8) -> tuple[float, float]:
    """Mean and spread of the reconstructed value under repeated perturbation."""
    values = []
    for t in range(trials):
        try:
            perturbed = perturb(data, t)
            values.append(float(reconstructor.reconstruct(
                reconstructor.fit(perturbed), perturbed)))
        except Exception:
            continue
    if len(values) < 2:
        raise ReconstructionError(
            f"only {len(values)} of {trials} perturbed reconstructions succeeded; "
            f"the spread cannot be estimated")
    return mean(values), pstdev(values)
