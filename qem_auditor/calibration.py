"""Are the quoted error bars the right size?

Every other module here asks whether a result is believable given its
stated uncertainty. This one asks the question underneath: **is the
stated uncertainty itself honest?**

It needs something the rest of the package rarely has -- measurements
whose true answers are known. Benchmark circuits, Clifford circuits, a
statevector cross-check, or a measurement whose exact expectation value
follows from the prepared state. Where that exists, the deviations can
be compared against the error bars that were supposed to cover them,
and the error bars can be caught being wrong.

THE NUMBER THIS MODULE EXISTS TO COMPUTE
----------------------------------------
For each measurement, z = (measured - exact) / sigma. If sigma is
honest, z has mean 0 and variance 1, so chi-squared per degree of
freedom is 1. It usually is not.

On the first full-scale real hardware run behind this package -- 84
circuits on IonQ Forte Enterprise, 1,100 shots each, 732 residuals --
chi-squared per degree of freedom came out at **5.93**. The quoted bars
were shot noise. The real scatter was 2.4 times larger, which means
there is an error source roughly 2.2 times the shot noise that the
model did not contain. The same analysis on a simulator rehearsal of
the same 84 circuits gave 2.68, so a little over half the excess was
already visible in simulation and the rest only appeared on hardware.

TWO FAILURES THAT LOOK THE SAME IN ONE NUMBER
---------------------------------------------
chi-squared per dof above 1 can mean either of two different things,
and they want different fixes:

  mean(z) != 0   a SYSTEMATIC BIAS. The correction is wrong in one
                 direction. More shots will not help.
  var(z) > 1     EXCESS SCATTER. Something is noisier than the model
                 says. More shots narrows each bar but the scatter
                 stays, so chi-squared gets WORSE.

On that real run one measurement group had mean(z) = -1.54 against an
rms of 2.68, so about a third of its excess was bias rather than noise
-- a fact invisible in the single headline number, and the difference
between "collect more data" and "the correction is wrong".

DEGREES OF FREEDOM ARE NOT THE SAMPLE SIZE
------------------------------------------
Fitting p parameters to n points leaves n - p degrees of freedom.
Dividing chi-squared by n instead of by n - p understates the problem,
and understates it in the flattering direction: the real run's 732
residuals carried 15 fitted parameters, and chi-squared per point was
5.81 against 5.93 per degree of freedom. So `fitted_parameters` is a
parameter here rather than an assumption, and leaving it at zero is a
claim that nothing was fitted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from .power import normal_cdf

#: Gaussian coverage at 1, 2 and 3 sigma. What honest bars should show.
EXPECTED_COVERAGE = (0.682689, 0.954500, 0.997300)

#: How far from chi2/dof = 1 counts as dishonest, in sigmas of its own
#: null. THREE, not two, and the reason is measured rather than chosen:
#: chi2/dof concentrates as the data grows (its standard deviation is
#: about sqrt(2/dof)), so at 2,000 residuals a 2-sigma band is only
#: +/-0.063 wide and ordinary sampling trips it one time in twenty-two.
#: A module whose job is to accuse error bars of being wrong cannot
#: afford to be wrong that often: 2,000 genuinely Gaussian residuals
#: came out at chi2/dof 0.929, which a 2-sigma rule called dishonest.
#: Three sigma costs almost nothing against the cases that matter --
#: the real hardware run in this module's docstring sits at 46 sigma.
HONEST_SIGMA = 3.0

#: Below this many degrees of freedom the Wilson-Hilferty approximation
#: to the chi-squared tail is not trustworthy, and `significance_sigma`
#: says so rather than quoting a number it cannot support.
_WILSON_HILFERTY_MIN_DOF = 20


class CalibrationError(ValueError):
    """Raised when the inputs cannot support a calibration check."""


@dataclass(frozen=True)
class ErrorBarAudit:
    """What the deviations say about the bars that were meant to cover them."""

    n: int
    fitted_parameters: int
    chi2: float
    mean_z: float
    #: Population variance of z, about z's own mean.
    variance_z: float
    coverage: tuple

    def __post_init__(self) -> None:
        if self.dof <= 0:
            raise CalibrationError(
                f"{self.n} residuals with {self.fitted_parameters} fitted "
                "parameters leaves no degrees of freedom: a fit with as many "
                "parameters as points describes the data perfectly and says "
                "nothing about the error bars")

    @property
    def dof(self) -> int:
        return self.n - self.fitted_parameters

    @property
    def chi2_per_dof(self) -> float:
        return self.chi2 / self.dof

    @property
    def inflation(self) -> float:
        """How many times too TIGHT the quoted bars are.

        The honest bar is this factor times the quoted one. Below 1 the
        bars are too wide rather than too tight.
        """
        return math.sqrt(self.chi2_per_dof)

    @property
    def unmodelled_in_sigmas(self) -> float:
        """The size of what the model is missing, in units of the quoted
        sigma, assuming it adds in quadrature with it.

        None of this says WHAT the missing term is. It says how big it
        is, which is the part that decides whether looking for it is
        worth anyone's afternoon.
        """
        excess = self.chi2_per_dof - 1.0
        return math.sqrt(excess) if excess > 0 else 0.0

    @property
    def bias_share(self) -> float:
        """Fraction of the mean square that is systematic rather than scatter.

        Separates "the correction is off in one direction", which more
        shots cannot fix, from "something is noisier than modelled",
        which more shots makes look worse rather than better.
        """
        mean_square = self.mean_z ** 2 + self.variance_z
        if mean_square <= 0:
            return 0.0
        return (self.mean_z ** 2) / mean_square

    @property
    def significance_sigma(self) -> Optional[float]:
        """How far chi-squared sits from 1, in sigmas of its own null.

        Wilson-Hilferty's cube-root transform, which is good for a few
        tens of degrees of freedom upward and None below that rather
        than a number that cannot be supported. Reported in sigmas
        because at the scale real runs reach, a p-value underflows to
        zero and stops carrying information.
        """
        if self.dof < _WILSON_HILFERTY_MIN_DOF:
            return None
        scale = 2.0 / (9.0 * self.dof)
        return ((self.chi2_per_dof ** (1.0 / 3.0) - (1.0 - scale))
                / math.sqrt(scale))

    @property
    def p_value(self) -> Optional[float]:
        """Probability of chi-squared this large if the bars were honest.

        None where `significance_sigma` is None. Underflows to 0.0 for
        a badly mis-stated uncertainty, which is not a bug: it is what
        a 46-sigma departure means.
        """
        z = self.significance_sigma
        return None if z is None else 1.0 - normal_cdf(z)

    def is_honest_at(self, threshold: float = HONEST_SIGMA) -> bool:
        """Is chi-squared per dof within `threshold` sigmas of 1?

        Two-sided on purpose. Bars that are too WIDE are also wrong,
        and a chi-squared far below 1 usually means a fit with enough
        parameters to trace its own noise.

        Below the Wilson-Hilferty floor the band is derived from the
        spread of chi-squared per dof itself -- about sqrt(2/dof) --
        rather than from a fixed pair of numbers, because at five
        degrees of freedom that spread is 0.63 and any fixed band is
        either useless or an accusation waiting to happen.
        """
        z = self.significance_sigma
        if z is not None:
            return abs(z) <= threshold
        spread = math.sqrt(2.0 / self.dof)
        return abs(self.chi2_per_dof - 1.0) <= threshold * spread

    @property
    def bars_are_honest(self) -> bool:
        """`is_honest_at` with this module's default threshold."""
        return self.is_honest_at()

    def describe(self) -> str:
        lines = [
            f"  {self.n} residuals, {self.fitted_parameters} fitted "
            f"parameter(s), {self.dof} degrees of freedom",
            f"  chi2/dof = {self.chi2_per_dof:.3f}"
            + ("  (1.0 is what honest bars give)" if self.bars_are_honest
               else ""),
        ]
        z = self.significance_sigma
        if z is not None:
            lines.append(f"  that is {abs(z):.1f} sigma from 1.0")

        if self.bars_are_honest:
            lines.append("  the quoted uncertainties cover the deviations: "
                         "nothing here says they are wrong")
        elif self.chi2_per_dof > 1:
            lines.append(
                f"  the quoted bars are {self.inflation:.2f}x TOO TIGHT. An "
                f"honest bar is {self.inflation:.2f}x the quoted one")
            lines.append(
                f"  the missing error is about {self.unmodelled_in_sigmas:.2f}x "
                "the quoted sigma, added in quadrature")
            lines.append(
                f"  {self.bias_share:.0%} of it is systematic (mean z = "
                f"{self.mean_z:+.3f}), the rest is scatter")
            if self.bias_share > 0.25:
                lines.append("  -> a systematic shift that size does not go "
                             "away with more shots; the correction is off in "
                             "one direction")
            else:
                lines.append("  -> mostly scatter, so more shots TIGHTENS each "
                             "bar without reducing the spread: chi2/dof gets "
                             "worse, not better")
        else:
            lines.append(
                f"  the quoted bars are {1 / self.inflation:.2f}x too WIDE. "
                "Either the uncertainty is overstated, or the fit has enough "
                "parameters to follow its own noise")

        lines.append("  coverage: " + ", ".join(
            f"{observed:.1%} within {k + 1} sigma (expect {expected:.1%})"
            for k, (observed, expected)
            in enumerate(zip(self.coverage, EXPECTED_COVERAGE))))
        return "\n".join(lines)


def audit_residuals(residuals: Sequence[float],
                    fitted_parameters: int = 0) -> ErrorBarAudit:
    """Audit standardised residuals -- deviations already divided by sigma.

    `fitted_parameters` is how many free parameters were fitted to this
    data. It is not optional in spirit: leaving it at 0 asserts that
    nothing was fitted, and a fitted correction with that left at 0
    reports a chi-squared per dof smaller than the truth.
    """
    values = [float(z) for z in residuals]
    if not values:
        raise CalibrationError("no residuals were supplied")
    if fitted_parameters < 0:
        raise CalibrationError("fitted_parameters cannot be negative")

    n = len(values)
    chi2 = sum(z * z for z in values)
    mean_z = sum(values) / n
    variance_z = sum((z - mean_z) ** 2 for z in values) / n
    coverage = tuple(
        sum(1 for z in values if abs(z) <= k) / n for k in (1, 2, 3))
    return ErrorBarAudit(n=n, fitted_parameters=fitted_parameters, chi2=chi2,
                         mean_z=mean_z, variance_z=variance_z,
                         coverage=coverage)


def audit_against_truth(measured: Sequence[float], sigma: Sequence[float],
                        exact: Sequence[float],
                        fitted_parameters: int = 0) -> ErrorBarAudit:
    """Audit measurements against known answers and their quoted bars.

    The three sequences are matched element by element. A sigma of zero
    is refused rather than skipped: a measurement quoted with no
    uncertainty at all is either a mistake or an infinitely confident
    claim, and dropping it quietly would let the rest of the set look
    better than it is.
    """
    if not (len(measured) == len(sigma) == len(exact)):
        raise CalibrationError(
            f"measured ({len(measured)}), sigma ({len(sigma)}) and exact "
            f"({len(exact)}) must be the same length; mismatched lengths "
            "mean some measurement is being compared against the wrong "
            "answer")
    residuals = []
    for i, (value, error, truth) in enumerate(zip(measured, sigma, exact)):
        if error <= 0:
            raise CalibrationError(
                f"measurement {i} is quoted with sigma={error}: a value with "
                "no uncertainty cannot be tested against its error bar")
        residuals.append((float(value) - float(truth)) / float(error))
    return audit_residuals(residuals, fitted_parameters)


@dataclass(frozen=True)
class CalibrationComparison:
    """The same analysis on hardware and on the model that predicted it."""

    hardware: ErrorBarAudit
    model: ErrorBarAudit
    label_hardware: str = "hardware"
    label_model: str = "model"

    @property
    def excess_ratio(self) -> float:
        """How much of the hardware's excess the model did NOT predict.

        Both numbers are excess over 1, because 1 is where an honest
        bar sits. Comparing the raw chi-squareds instead would call a
        model that predicted the excess perfectly a partial success.
        """
        model_excess = self.model.chi2_per_dof - 1.0
        hardware_excess = self.hardware.chi2_per_dof - 1.0
        if model_excess <= 0:
            return float("inf") if hardware_excess > 0 else 1.0
        return hardware_excess / model_excess

    def describe(self) -> str:
        lines = [
            f"  {self.label_hardware}: chi2/dof "
            f"{self.hardware.chi2_per_dof:.2f}",
            f"  {self.label_model}: chi2/dof {self.model.chi2_per_dof:.2f}",
        ]
        ratio = self.excess_ratio
        if ratio == float("inf"):
            lines.append("  the model predicted NO excess at all, so none of "
                         "the hardware's is accounted for")
        else:
            share = 1.0 / ratio if ratio > 0 else 0.0
            lines.append(
                f"  the model accounts for {share:.0%} of the excess; the "
                f"remaining {1 - share:.0%} appeared only on hardware")
        lines.append("  Excess is measured from 1.0, not from the model's own "
                     "chi2: a model that reproduced the excess exactly would "
                     "otherwise score as half right.")
        return "\n".join(lines)


def compare_to_model(hardware: ErrorBarAudit, model: ErrorBarAudit,
                     label_hardware: str = "hardware",
                     label_model: str = "model") -> CalibrationComparison:
    """Set a hardware calibration against the simulation that predicted it.

    This is the check a noise model owes and almost never gets: not
    "did it predict the answer" but "did it predict how wrong the
    answer would be". A model that gets the central value right and the
    scatter wrong will mislead every experiment designed against it.
    """
    return CalibrationComparison(hardware=hardware, model=model,
                                 label_hardware=label_hardware,
                                 label_model=label_model)


@dataclass(frozen=True)
class GroupedCalibration:
    """One chi-squared per subgroup, and which subgroup carries the excess.

    A single chi-squared over a heterogeneous set says less than it
    appears to. The clearest instance is in this package's own source
    data: an IDEAL, noiseless simulator run scored chi2/dof 7.28 --
    worse than the real hardware's 5.93 -- because three real hardware
    circuits had been reused inside it. Three points out of 84, from a
    different distribution, moved the headline number past the hardware
    it was meant to be a baseline for.

    So the headline is reported next to its parts, and the part that
    dominates is named.
    """

    overall: ErrorBarAudit
    groups: dict

    @property
    def worst(self) -> Optional[str]:
        """The group with the largest chi-squared per degree of freedom."""
        if not self.groups:
            return None
        return max(self.groups, key=lambda key: self.groups[key].chi2_per_dof)

    def share_of_chi2(self, key) -> float:
        if self.overall.chi2 <= 0:
            return 0.0
        return self.groups[key].chi2 / self.overall.chi2

    def share_of_points(self, key) -> float:
        return self.groups[key].n / self.overall.n

    @property
    def concentrated(self) -> bool:
        """Does one group carry a share of chi-squared well past its share
        of the data?

        The threshold is deliberately loose: this flags "go look at that
        group" rather than deciding anything.
        """
        worst = self.worst
        if worst is None:
            return False
        return self.share_of_chi2(worst) > 1.5 * self.share_of_points(worst)

    def describe(self) -> str:
        lines = ["  overall:", self.overall.describe(), "", "  by group:"]
        for key in sorted(self.groups, key=str):
            audit = self.groups[key]
            lines.append(
                f"    {str(key):16s} n={audit.n:4d}  chi2/dof="
                f"{audit.chi2_per_dof:7.2f}  mean z={audit.mean_z:+.2f}  "
                f"{self.share_of_chi2(key):5.1%} of chi2 from "
                f"{self.share_of_points(key):.1%} of the points")
        worst = self.worst
        if worst is not None and self.concentrated:
            lines.append("")
            lines.append(
                f"  {str(worst)!r} carries {self.share_of_chi2(worst):.0%} of "
                f"the total chi-squared on {self.share_of_points(worst):.0%} "
                "of the points. The headline number is partly a fact about "
                "that subset, not about the run as a whole.")
        return "\n".join(lines)


def audit_groups(residuals_by_group: dict, fitted_parameters: int = 0,
                 parameters_by_group: Optional[dict] = None
                 ) -> GroupedCalibration:
    """Audit standardised residuals split into named groups.

    `fitted_parameters` applies to the pooled fit. Per-group parameter
    counts are usually not meaningful -- a correction fitted across the
    whole run does not divide cleanly between subsets -- so they default
    to zero and `parameters_by_group` is there for the case where a
    caller genuinely fitted each group separately.
    """
    if not residuals_by_group:
        raise CalibrationError("no groups were supplied")
    per_group = dict(parameters_by_group or {})
    groups = {key: audit_residuals(values, per_group.get(key, 0))
              for key, values in residuals_by_group.items()}
    pooled = [z for values in residuals_by_group.values() for z in values]
    return GroupedCalibration(
        overall=audit_residuals(pooled, fitted_parameters), groups=groups)
