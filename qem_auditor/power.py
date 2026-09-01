"""How much evidence would actually be enough.

`UNDER_POWERED` as a bare label tells a researcher nothing they can act
on. This module turns it into a number: given the spread already
observed and the difference that would matter, how many independent runs
does the claim need, how much power do the existing runs have, and how
many more are required.

The load-bearing design decision here is that **sigma must match the
claim's uncertainty scope**, and this module refuses to compute a sample
size when it does not. That is not pedantry, it is the single most
expensive mistake in this project's history restated in statistical
form: the H4 pipeline's 8-seed bootstrap bars had a within-submission
sigma near 0.0015 kcal/mol, while independent submissions of the
identical circuit set differed by 3.27. A power calculation fed the
bootstrap sigma would cheerfully report that 2 runs suffice for a claim
that 8 independent runs still could not establish, because it would be
sizing against the wrong variance by three orders of magnitude.

So `required_n` takes the sigma AND what that sigma covers, and a claim
that has to survive run-to-run drift cannot be sized with a sigma that
never saw any.

Stdlib only: the normal quantile is Acklam's rational approximation, the
CDF comes from math.erf.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, pstdev, stdev
from typing import Optional

from .schema import Experiment, UncertaintyCoverage


class PowerError(ValueError):
    """A sample size that cannot be computed as asked."""


# --------------------------------------------------------------------------
# Normal distribution, stdlib only
# --------------------------------------------------------------------------

def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# Acklam's inverse normal CDF. Accurate to ~1.15e-9 over the whole range,
# far beyond anything a sample-size calculation needs.
_A = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
_B = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01]
_C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
_D = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00]


def normal_quantile(p: float) -> float:
    """Inverse standard-normal CDF (the z for a given cumulative probability)."""
    if not 0.0 < p < 1.0:
        raise PowerError(f"probability must be strictly between 0 and 1, got {p}")
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
               ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
               ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
           (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1)


def chi2_quantile(p: float, k: int) -> float:
    """Chi-square quantile via Wilson-Hilferty. Accurate to well under 1%
    for k >= 2, which is all a variance bound needs."""
    if k < 1:
        raise PowerError(f"degrees of freedom must be >= 1, got {k}")
    z = normal_quantile(p)
    return k * (1 - 2.0 / (9 * k) + z * math.sqrt(2.0 / (9 * k))) ** 3


def sigma_upper_bound(values: list[float], confidence: float = 0.95) -> float:
    """An upper confidence bound on sigma, not the point estimate.

    A sample standard deviation from 4 points is not sigma; it is a very
    noisy guess at sigma. The one-sided upper bound

        sigma_upper = s * sqrt((n-1) / chi2_{1-confidence, n-1})

    is about 3.7x the point estimate at n=4, and 1.5x at n=20. Sizing an
    experiment against the point estimate of a barely-known sigma is the
    same species of overconfidence as quoting a bootstrap bar for a
    reproducibility claim: it reports the number you happened to get as if
    it were the number you needed.

    This is why a handful of tightly-agreeing draws does not settle a
    replication requirement. They constrain the mean well and sigma badly,
    and the sample size depends on sigma.
    """
    n = len(values)
    if n < 2:
        raise PowerError(f"need at least 2 values to bound sigma, got {n}")
    s = stdev(values)
    if s <= 0:
        raise PowerError("zero spread: sigma cannot be bounded from identical values")
    chi2_low = chi2_quantile(1 - confidence, n - 1)
    if chi2_low <= 0:
        raise PowerError(f"degenerate chi-square bound at n={n}")
    return s * math.sqrt((n - 1) / chi2_low)


# --------------------------------------------------------------------------
# Scope matching -- the part that makes the number mean something
# --------------------------------------------------------------------------

# What a claim of each kind must survive, and therefore which variance
# components its sigma has to contain.
SCOPE_REQUIREMENTS = {
    "single_run": ("shot_noise",),
    "reproducible": ("shot_noise", "method_monte_carlo", "cross_submission"),
    "hardware_ready": ("shot_noise", "method_monte_carlo", "cross_submission", "noise_model"),
}


def check_scope(sigma_covers: UncertaintyCoverage, claim_scope: str) -> list[str]:
    """Which variance components the claim needs and sigma does not contain."""
    if claim_scope not in SCOPE_REQUIREMENTS:
        raise PowerError(
            f"unknown claim scope {claim_scope!r}; expected one of "
            f"{sorted(SCOPE_REQUIREMENTS)}")
    return [axis for axis in SCOPE_REQUIREMENTS[claim_scope]
            if not getattr(sigma_covers, axis)]


# --------------------------------------------------------------------------
# The calculation
# --------------------------------------------------------------------------

@dataclass
class PowerAnalysis:
    """A quantitative answer to 'is this under-powered, and by how much?'"""

    effect_size_kcal: float
    sigma_kcal: float
    current_n: int
    required_n: int
    power: float
    alpha: float
    beta: float
    sigma_covers: Optional[UncertaintyCoverage] = None
    claim_scope: str = "reproducible"
    scope_gaps: list[str] = None  # type: ignore[assignment]
    cost_per_run_usd: float = 0.0
    sigma_is_upper_bound: bool = False
    """True when sigma is a confidence bound rather than a point estimate --
    the honest choice when sigma was itself estimated from few runs."""

    sigma_point_kcal: Optional[float] = None
    """The point estimate, kept alongside so the inflation is visible."""

    def __post_init__(self) -> None:
        if self.scope_gaps is None:
            self.scope_gaps = []

    @property
    def minimum_additional(self) -> int:
        return max(0, self.required_n - self.current_n)

    @property
    def expected_cost_usd(self) -> float:
        return self.minimum_additional * self.cost_per_run_usd

    @property
    def is_powered(self) -> bool:
        return self.current_n >= self.required_n and not self.scope_gaps

    @property
    def standardized_effect(self) -> float:
        return self.effect_size_kcal / self.sigma_kcal if self.sigma_kcal else float("inf")

    def summary(self) -> str:
        if self.scope_gaps:
            return (f"required_n is not computable for a '{self.claim_scope}' claim from a "
                    f"sigma that never varied {', '.join(self.scope_gaps)} -- sizing against "
                    f"the wrong variance")
        cost = (f", ~${self.expected_cost_usd:,.2f}"
                if self.cost_per_run_usd else "")
        basis = ""
        if self.sigma_is_upper_bound and self.sigma_point_kcal:
            basis = (f" (95% upper bound; point estimate {self.sigma_point_kcal:.4g} from "
                     f"only {self.current_n} runs)")
        return (f"power={self.power:.2f} at n={self.current_n}; "
                f"required_n={self.required_n} to detect {self.effect_size_kcal:.4g} kcal/mol "
                f"against sigma={self.sigma_kcal:.4g}{basis}; "
                f"minimum_additional={self.minimum_additional}{cost}")


def required_n(effect_size: float, sigma: float, alpha: float = 0.05,
               beta: float = 0.20) -> int:
    """Independent runs needed to detect `effect_size` against `sigma`.

        n = ((z_{1-alpha/2} + z_{1-beta}) / (delta/sigma))^2

    Two-sided, one-sample, normal approximation. Rounded UP: a sample size
    is a floor, and rounding a requirement down is how an under-powered
    study gets declared adequate.
    """
    if sigma <= 0:
        raise PowerError(f"sigma must be positive, got {sigma}")
    if effect_size <= 0:
        raise PowerError(
            f"effect size must be positive, got {effect_size}. An effect size of zero "
            f"needs infinite samples -- 'no difference' is not something a finite "
            f"experiment can establish.")
    if not 0 < alpha < 1 or not 0 < beta < 1:
        raise PowerError(f"alpha and beta must be in (0,1), got alpha={alpha}, beta={beta}")
    z_alpha = normal_quantile(1 - alpha / 2)
    z_beta = normal_quantile(1 - beta)
    return math.ceil(((z_alpha + z_beta) / (effect_size / sigma)) ** 2)


def power_at(n: int, effect_size: float, sigma: float, alpha: float = 0.05) -> float:
    """Probability of detecting `effect_size` with `n` independent runs."""
    if n < 1:
        return 0.0
    if sigma <= 0:
        raise PowerError(f"sigma must be positive, got {sigma}")
    z_alpha = normal_quantile(1 - alpha / 2)
    ncp = (effect_size / sigma) * math.sqrt(n)
    # Two-sided: both rejection tails, though the far tail is negligible
    # for any effect worth powering for.
    return normal_cdf(ncp - z_alpha) + normal_cdf(-ncp - z_alpha)


def analyze(effect_size: float, sigma: float, current_n: int,
            sigma_covers: UncertaintyCoverage | None = None,
            claim_scope: str = "reproducible",
            alpha: float = 0.05, beta: float = 0.20,
            cost_per_run_usd: float = 0.0,
            sigma_is_upper_bound: bool = False,
            sigma_point: float | None = None) -> PowerAnalysis:
    """Full analysis, including whether sigma is even the right sigma.

    When sigma's coverage is narrower than the claim requires, the
    required_n and power fields are still computed but are reported
    alongside the gaps, and `is_powered` stays False however large
    `current_n` is. A number sized against the wrong variance is not a
    smaller answer to the right question; it is an answer to a different
    one.
    """
    gaps = check_scope(sigma_covers, claim_scope) if sigma_covers is not None else []
    return PowerAnalysis(
        effect_size_kcal=effect_size,
        sigma_kcal=sigma,
        current_n=current_n,
        required_n=required_n(effect_size, sigma, alpha, beta),
        power=power_at(current_n, effect_size, sigma, alpha),
        alpha=alpha,
        beta=beta,
        sigma_covers=sigma_covers,
        claim_scope=claim_scope,
        scope_gaps=gaps,
        cost_per_run_usd=cost_per_run_usd,
        sigma_is_upper_bound=sigma_is_upper_bound,
        sigma_point_kcal=sigma_point,
    )


def analyze_experiment(exp: Experiment, threshold_kcal: float = 0.25,
                       claim_scope: str = "reproducible",
                       alpha: float = 0.05, beta: float = 0.20,
                       cost_per_run_usd: float = 0.0,
                       conservative: bool = True) -> Optional[PowerAnalysis]:
    """Power analysis straight off an experiment record.

    Sigma comes from the INDEPENDENT replicates only, for the same reason
    the reproducibility gate counts only those: a spread computed across
    bootstrap resamples of one execution is a within-run number wearing a
    between-run label.

    The effect size is the distance from the observed mean to the
    threshold the claim has to clear -- how big a difference the
    experiment must be able to see.

    `conservative=True` (the default) sizes against the 95% upper bound on
    sigma rather than the point estimate, because with a handful of runs
    the point estimate is not sigma. Pass False only when sigma is known
    from a large sample or from theory.
    """
    reps = exp.outputs.independent_errors_kcal
    if len(reps) < 2:
        return None
    observed = mean(reps)
    sigma = stdev(reps)  # sample sd: n-1, since these estimate a population
    if sigma <= 0:
        return None
    effect = abs(threshold_kcal - observed)
    if effect <= 0:
        return None
    sigma_used, point = sigma, None
    if conservative:
        sigma_used = sigma_upper_bound(reps)
        point = sigma
    return analyze(effect, sigma_used, len(reps), exp.outputs.uncertainty,
                   claim_scope, alpha, beta, cost_per_run_usd,
                   sigma_is_upper_bound=conservative, sigma_point=point)


@dataclass
class MeanInterval:
    """How tightly the mean is actually pinned down, and by how many runs."""

    mean_kcal: float
    half_width_kcal: float
    n: int
    confidence: float

    @property
    def low(self) -> float:
        return self.mean_kcal - self.half_width_kcal

    @property
    def high(self) -> float:
        return self.mean_kcal + self.half_width_kcal

    def describe(self) -> str:
        return (f"{self.mean_kcal:.4g} +- {self.half_width_kcal:.4g} kcal/mol "
                f"[{self.low:.4g}, {self.high:.4g}] at n={self.n}")


def mean_interval(values: list[float], confidence: float = 0.95,
                  conservative: bool = True) -> MeanInterval:
    """Confidence interval on the mean.

    Uses the upper confidence bound on sigma rather than the point
    estimate when `conservative`, which widens the interval to account for
    sigma itself being estimated from few runs. That is deliberately more
    conservative than a textbook t-interval: it answers "how tightly is
    this pinned, given I barely know the spread either" rather than
    assuming the observed spread is the truth.
    """
    n = len(values)
    if n < 2:
        raise PowerError(f"need at least 2 values for an interval, got {n}")
    sigma = sigma_upper_bound(values, confidence) if conservative else stdev(values)
    z = normal_quantile(1 - (1 - confidence) / 2)
    return MeanInterval(mean(values), z * sigma / math.sqrt(n), n, confidence)


def interval_at_n(values: list[float], target_n: int, confidence: float = 0.95) -> MeanInterval:
    """What the interval would become at `target_n` runs, holding the
    observed spread fixed.

    Projects two effects at once: the 1/sqrt(n) narrowing, and sigma
    itself becoming better known as n grows (the upper bound tightens from
    ~3.0x the point estimate at n=4 to ~1.8x at n=8). Both matter, and
    reporting only the first overstates how much a few more runs buy.
    """
    n = len(values)
    if target_n <= n:
        return mean_interval(values, confidence)
    s = stdev(values)
    # Rebuild the sigma bound as if target_n samples had produced the same
    # point-estimate spread.
    chi2_low = chi2_quantile(1 - confidence, target_n - 1)
    sigma_proj = s * math.sqrt((target_n - 1) / chi2_low)
    z = normal_quantile(1 - (1 - confidence) / 2)
    return MeanInterval(mean(values), z * sigma_proj / math.sqrt(target_n), target_n, confidence)


# --------------------------------------------------------------------------
# Sequential stopping
# --------------------------------------------------------------------------

def sequential_alpha(look: int, total_looks: int, alpha: float = 0.05) -> float:
    """Pocock-style constant alpha spend per interim look.

    Peeking at accumulating data and stopping at the first significant
    result inflates the false-positive rate well beyond the nominal alpha
    -- with 5 looks at alpha=0.05, the real rate is near 0.14. Anyone
    running replication draws one at a time and watching the mean is doing
    exactly this. Spending the alpha across the planned looks keeps the
    overall rate at alpha.
    """
    if look < 1 or look > total_looks:
        raise PowerError(f"look {look} is outside 1..{total_looks}")
    return alpha / total_looks


def may_stop_early(values: list[float], threshold: float, total_looks: int,
                   alpha: float = 0.05) -> tuple[bool, str]:
    """Can replication stop before the planned n, on the evidence so far?

    Returns (may_stop, reason). Deliberately conservative: it only ever
    licenses stopping when the accumulated evidence clears the
    alpha-adjusted bound, and it never licenses stopping to declare
    success at a nominal alpha that ignored the earlier looks.
    """
    n = len(values)
    if n < 2:
        return False, f"{n} value(s): too few to judge"
    if n > total_looks:
        return False, (f"{n} values exceeds the {total_looks} planned looks -- the alpha "
                       f"spend was budgeted for {total_looks}")
    m, s = mean(values), stdev(values)
    if s <= 0:
        return False, "zero spread: cannot form a test statistic"
    spent = sequential_alpha(n, total_looks, alpha)
    z_bound = normal_quantile(1 - spent / 2)
    z = abs(threshold - m) / (s / math.sqrt(n))
    if z > z_bound:
        return True, (f"z={z:.2f} clears the look-{n}/{total_looks} bound {z_bound:.2f} "
                      f"(alpha spent {spent:.4f}) -- the distance from {threshold} is "
                      f"established at n={n}")
    return False, (f"z={z:.2f} is below the look-{n}/{total_looks} bound {z_bound:.2f}; "
                   f"stopping here would use a nominal alpha that ignored the earlier looks")


@dataclass(frozen=True)
class Comparison:
    """Whether two methods can be told apart by the runs that were done.

    The shootout ranks methods by median error and prints them in order,
    which invites the reader to believe the order. On the measured device
    REM+ZNE came in at 1.15 and CDR at 1.29 kcal/mol, with spreads of
    1.35 and 0.79 -- a gap of 0.14 between two numbers that move by ten
    times that between seeds. Printing them one above the other asserts
    something the data does not support, and it is the same over-claiming
    this package objects to everywhere else.

    So a ranking now carries the question "could this have come out the
    other way", and when the answer is yes it says so and says what it
    would take to settle it.
    """

    name_a: str
    name_b: str
    mean_a: float
    mean_b: float
    sigma: float
    n: int
    alpha: float = 0.05
    beta: float = 0.20

    @property
    def gap(self) -> float:
        return abs(self.mean_a - self.mean_b)

    @property
    def standard_error(self) -> float:
        """Of the DIFFERENCE, which is where the factor of two lives.

        Each mean carries sigma/sqrt(n) and the difference carries both,
        so the separation a comparison needs is sqrt(2) wider than the
        one a single mean needs. Forgetting that is how two methods get
        declared distinguishable on half the runs it would really take.
        """
        return self.sigma * math.sqrt(2.0 / self.n) if self.n else float("inf")

    @property
    def separation(self) -> float:
        """The gap in standard errors of the difference."""
        error = self.standard_error
        return self.gap / error if error > 0 else float("inf")

    @property
    def distinguishable(self) -> bool:
        return self.separation >= normal_quantile(1 - self.alpha / 2)

    @property
    def better(self) -> Optional[str]:
        """The winner, or None when the runs cannot name one.

        Returning None rather than the nominal leader is the point: a
        caller that wants a ranking has to decide what to do about a tie
        instead of being handed one silently.
        """
        if not self.distinguishable:
            return None
        return self.name_a if self.mean_a < self.mean_b else self.name_b

    @property
    def required_n(self) -> Optional[int]:
        """Runs per method needed to resolve this gap, if it is real.

        None when the observed gap is zero -- no finite number of runs
        establishes that two methods are identical, and quoting one would
        promise something unachievable.
        """
        if self.gap <= 0:
            return None
        z = normal_quantile(1 - self.alpha / 2) + normal_quantile(1 - self.beta)
        return math.ceil(2.0 * (z * self.sigma / self.gap) ** 2)

    def describe(self) -> str:
        if self.distinguishable:
            return (f"{self.better} is better: {self.gap:.4g} apart, "
                    f"{self.separation:.1f} standard errors at n={self.n}")
        needed = self.required_n
        tail = ("the observed gap is zero, and no number of runs establishes "
                "that two methods are identical"
                if needed is None else
                f"about {needed} runs each would settle it (n={self.n} now)")
        return (f"{self.name_a} and {self.name_b} are not distinguishable: "
                f"{self.gap:.4g} apart against a spread of {self.sigma:.4g}; "
                f"{tail}")


def compare(name_a: str, values_a: list[float], name_b: str, values_b: list[float],
            confidence: float = 0.95, beta: float = 0.20,
            conservative: bool = True) -> Comparison:
    """Can these two sets of runs tell their methods apart?

    The spread is pooled across both methods, because the question is
    about the difference and both sides contribute noise to it. As in
    `mean_interval`, the pooled sigma is by default the upper confidence
    bound rather than the point estimate: with four or eight seeds the
    spread is barely known either, and an interval that pretends
    otherwise declares ties resolved that are not.
    """
    if len(values_a) != len(values_b):
        raise PowerError(
            f"comparing {len(values_a)} runs against {len(values_b)}: this "
            "assumes equal run counts, and silently using the smaller would "
            "overstate what the larger set bought")
    n = len(values_a)
    if n < 2:
        raise PowerError(f"need at least 2 runs per method, got {n}")
    pooled = [v - mean(values_a) for v in values_a] + [v - mean(values_b) for v in values_b]
    sigma = (sigma_upper_bound(pooled, confidence) if conservative else stdev(pooled))
    return Comparison(name_a=name_a, name_b=name_b,
                      mean_a=mean(values_a), mean_b=mean(values_b),
                      sigma=sigma, n=n, alpha=1 - confidence, beta=beta)


def rank_with_ties(results: dict, confidence: float = 0.95,
                   conservative: bool = True) -> list:
    """Order methods best-first, grouping the ones that cannot be separated.

    Returns a list of tiers; each tier is a list of names the runs cannot
    tell apart from the tier's leader. A single-method tier is a claim the
    data supports. A tier of three is the honest form of what a ranked
    list of three would have asserted.
    """
    ordered = sorted(results.items(), key=lambda kv: mean(kv[1]))
    tiers: list = []
    for name, values in ordered:
        if tiers and not compare(tiers[-1][0], results[tiers[-1][0]], name, values,
                                 confidence=confidence,
                                 conservative=conservative).distinguishable:
            tiers[-1].append(name)
        else:
            tiers.append([name])
    return tiers
