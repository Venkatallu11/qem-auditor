"""Record-integrity checks.

The gates in gates.py answer "did the science hold up?". This module
answers a prior question: "is this record even capable of supporting a
claim?" A record whose own numbers contradict each other cannot be
audited -- not because the method failed, but because there is nothing
coherent to audit. That is a different failure from INVALID and gets its
own verdict (INVALID RECORD) so the two are never conflated.

Every check here is a statement about internal consistency only. None of
them look at whether a result is *good*; that is the gates' job.
"""
from __future__ import annotations

from statistics import mean, pstdev

from .schema import Experiment

# A headline mitigated error is allowed to sit this far from the mean of
# the replicates that supposedly back it: one replicate standard
# deviation, or 5% of the mean, whichever is looser. Beyond that, the
# headline number is not the thing the replicates measured -- the classic
# shape of a best-draw being reported as "the" result.
_HEADLINE_REL_TOL = 0.05


def integrity_violations(exp: Experiment) -> list[str]:
    """Returns a list of human-readable violations; empty means clean."""
    v: list[str] = []
    out, ctl = exp.outputs, exp.controls

    if not exp.experiment_id.strip():
        v.append("experiment_id is empty -- a record with no identity cannot be cited")
    if not exp.backend.strip():
        v.append("backend is empty -- a result with no stated backend has no scope")
    if exp.shots <= 0:
        v.append(f"shots={exp.shots} is not a positive shot count")
    if out.n_replicates_target < 1:
        v.append(f"n_replicates_target={out.n_replicates_target} is not a positive target")

    for label, value in (("raw_error_kcal", out.raw_error_kcal),
                         ("mitigated_error_kcal", out.mitigated_error_kcal),
                         ("q95_kcal", out.q95_kcal)):
        if value is not None and value < 0:
            v.append(f"{label}={value} is negative -- these are error magnitudes")
    for i, r in enumerate(out.replicate_errors_kcal):
        if r < 0:
            v.append(f"replicate_errors_kcal[{i}]={r} is negative -- these are error magnitudes")

    # Counted structurally, not by value: the assertion is about whether
    # draws were MADE, and a draw whose value is withheld (a blinded
    # challenge) or not yet transcribed is still a draw. Counting values
    # here would make redaction look like a malformed record, which is a
    # different finding entirely.
    if ctl.reproducibility_checked and len(out.replicates) < 2:
        v.append(
            f"reproducibility_checked=True but only {len(out.replicates)} "
            "replicate(s) recorded -- reproducibility cannot be asserted without the data"
        )

    if out.q95_kcal is not None and out.mitigated_error_kcal is not None:
        if out.q95_kcal < out.mitigated_error_kcal:
            v.append(
                f"q95_kcal={out.q95_kcal} is below mitigated_error_kcal="
                f"{out.mitigated_error_kcal} -- a 95% uncertainty envelope cannot be "
                "tighter than the point error it envelopes"
            )

    if out.n_trials is not None and out.n_outlier_trials is not None:
        if out.n_outlier_trials > out.n_trials:
            v.append(f"n_outlier_trials={out.n_outlier_trials} exceeds n_trials={out.n_trials}")
        if out.n_trials < 0 or out.n_outlier_trials < 0:
            v.append("trial counts cannot be negative")

    quantiles = [("q50_kcal", out.q50_kcal), ("q95_kcal", out.q95_kcal),
                 ("q99_kcal", out.q99_kcal)]
    present = [(n, q) for n, q in quantiles if q is not None]
    for (n_lo, lo), (n_hi, hi) in zip(present, present[1:]):
        if hi < lo:
            v.append(f"{n_hi}={hi} is below {n_lo}={lo} -- quantiles must be non-decreasing")

    reps = out.replicate_errors_kcal
    if len(reps) >= 2 and out.mitigated_error_kcal is not None:
        m = mean(reps)
        tol = max(pstdev(reps), _HEADLINE_REL_TOL * abs(m))
        if abs(out.mitigated_error_kcal - m) > tol:
            v.append(
                f"mitigated_error_kcal={out.mitigated_error_kcal} is not consistent with "
                f"its own {len(reps)} replicates (mean={m:.6f}, tolerance={tol:.6f}) -- "
                "the headline number is not what the replicates measured"
            )

    return v
