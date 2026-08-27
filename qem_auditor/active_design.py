"""When more data cannot help, and what to run instead.

The most expensive kind of experiment is one that cannot answer its
question no matter how many shots it gets. This project hit exactly that:
after iterations of trying to pin down a calibration parameter by
collecting more samples from the same circuit, a Fisher analysis showed
the parameter combination was structurally unidentifiable -- the
experiment's design, not its sample size, was the binding constraint.
Every additional sample was buying nothing.

So this module answers two questions the planner could not:

1. **Is this parameter learnable from this experiment at all?** The
   Fisher information F = J^T Sigma^-1 J says how much the data
   constrains each direction in parameter space. A near-zero eigenvalue
   means a direction the experiment is blind to, and blindness is not
   cured by repetition.

2. **What experiment would see it?** Given a weak direction v, rank
   candidate experiments by how much information they add along v per
   unit cost -- or by the D-optimal criterion when no single direction
   dominates. That is active experimental design: not ranking experiments
   someone proposed, but deriving the one the current evidence needs.

Stdlib only: symmetric eigendecomposition by cyclic Jacobi rotation,
which is exact enough and well-behaved on the small (5-20 parameter)
matrices this involves.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

Matrix = list[list[float]]
Vector = list[float]


class DesignError(ValueError):
    """A design question that cannot be answered as asked."""


# --------------------------------------------------------------------------
# Small dense linear algebra, stdlib only
# --------------------------------------------------------------------------

def _check_square(m: Matrix, name: str = "matrix") -> int:
    n = len(m)
    if n == 0:
        raise DesignError(f"{name} is empty")
    for row in m:
        if len(row) != n:
            raise DesignError(f"{name} is not square: {n} rows, a row of {len(row)}")
    return n


def is_symmetric(m: Matrix, tol: float = 1e-9) -> bool:
    n = _check_square(m)
    return all(abs(m[i][j] - m[j][i]) <= tol * max(1.0, abs(m[i][j]))
               for i in range(n) for j in range(i + 1, n))


def jacobi_eigen(matrix: Matrix, max_sweeps: int = 100,
                 tol: float = 1e-12) -> tuple[Vector, Matrix]:
    """Eigenvalues and eigenvectors of a real symmetric matrix.

    Cyclic Jacobi: repeatedly zero the largest off-diagonal element by a
    plane rotation. Chosen over anything faster because it is short enough
    to audit by eye, numerically well-behaved on the small matrices here,
    and needs no dependency -- which matters for a module whose whole job
    is telling someone their experiment cannot answer their question.

    Returns (eigenvalues, eigenvectors) with eigenvectors as COLUMNS,
    sorted ascending by eigenvalue so the weakest direction is first.
    """
    n = _check_square(matrix, "Fisher matrix")
    if not is_symmetric(matrix):
        raise DesignError(
            "Fisher information must be symmetric; got a matrix that is not. "
            "F = J^T Sigma^-1 J is symmetric by construction, so an asymmetric "
            "input usually means the Jacobian or the covariance is wrong.")
    a = [row[:] for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for _ in range(max_sweeps):
        off = math.sqrt(sum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j))
        if off <= tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) <= tol:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p], a[k][q] = c * akp - s * akq, s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k], a[q][k] = c * apk - s * aqk, s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p], v[k][q] = c * vkp - s * vkq, s * vkp + c * vkq

    eigenvalues = [a[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: eigenvalues[i])
    return ([eigenvalues[i] for i in order],
            [[v[r][i] for i in order] for r in range(n)])


def log_det(matrix: Matrix) -> float:
    """log det of a symmetric positive-(semi)definite matrix.

    Returns -inf for a singular matrix rather than raising: a design with
    zero information in some direction has genuinely infinite uncertainty
    there, and -inf is the honest value for the D-optimal objective.
    """
    eigenvalues, _ = jacobi_eigen(matrix)
    total = 0.0
    for lam in eigenvalues:
        if lam <= 0:
            return float("-inf")
        total += math.log(lam)
    return total


def quadratic_form(matrix: Matrix, vector: Vector) -> float:
    """v^T M v."""
    n = _check_square(matrix)
    if len(vector) != n:
        raise DesignError(f"vector of length {len(vector)} against a {n}x{n} matrix")
    return sum(vector[i] * matrix[i][j] * vector[j] for i in range(n) for j in range(n))


def add(a: Matrix, b: Matrix) -> Matrix:
    n = _check_square(a)
    if _check_square(b) != n:
        raise DesignError("cannot add matrices of different sizes")
    return [[a[i][j] + b[i][j] for j in range(n)] for i in range(n)]


# --------------------------------------------------------------------------
# Fisher information
# --------------------------------------------------------------------------

def fisher_information(jacobian: Sequence[Sequence[float]],
                       sigma: Sequence[float]) -> Matrix:
    """F = J^T Sigma^-1 J, for independent measurements.

    `jacobian[i][k]` is d(observable_i)/d(parameter_k); `sigma[i]` is the
    standard deviation of measurement i. Diagonal Sigma only -- correlated
    measurements need the full inverse covariance, which this deliberately
    does not fake.
    """
    if not jacobian:
        raise DesignError("an empty Jacobian carries no information")
    n_obs = len(jacobian)
    n_par = len(jacobian[0])
    if any(len(row) != n_par for row in jacobian):
        raise DesignError("the Jacobian is ragged: every row needs one entry per parameter")
    if len(sigma) != n_obs:
        raise DesignError(
            f"{len(sigma)} sigmas for {n_obs} measurements -- one uncertainty per "
            f"measurement is required")
    if any(s <= 0 for s in sigma):
        raise DesignError("every measurement uncertainty must be positive")

    f = [[0.0] * n_par for _ in range(n_par)]
    for i in range(n_obs):
        w = 1.0 / (sigma[i] ** 2)
        for a in range(n_par):
            ja = jacobian[i][a]
            if ja == 0.0:
                continue
            for b in range(n_par):
                f[a][b] += w * ja * jacobian[i][b]
    return f


@dataclass
class Identifiability:
    """Whether the parameters can be learned from this experiment at all."""

    eigenvalues: Vector
    eigenvectors: Matrix
    parameter_names: list[str] = field(default_factory=list)
    tolerance: float = 1e-8

    @property
    def lambda_min(self) -> float:
        return self.eigenvalues[0]

    @property
    def lambda_max(self) -> float:
        return self.eigenvalues[-1]

    @property
    def condition_number(self) -> float:
        if self.lambda_min <= 0:
            return float("inf")
        return self.lambda_max / self.lambda_min

    @property
    def is_identifiable(self) -> bool:
        return self.lambda_min > self.tolerance

    def weak_direction(self) -> Vector:
        """The direction the experiment is least able to see."""
        return [row[0] for row in self.eigenvectors]

    def describe_weak_direction(self) -> str:
        v = self.weak_direction()
        names = self.parameter_names or [f"p{i}" for i in range(len(v))]
        terms = sorted(zip(names, v), key=lambda nv: -abs(nv[1]))
        lead = ", ".join(f"{c:+.3f}*{n}" for n, c in terms[:3])
        return lead

    def verdict(self) -> str:
        """What this means for the next experiment, in plain terms."""
        if self.is_identifiable:
            return (f"identifiable: lambda_min={self.lambda_min:.4g}, "
                    f"condition number {self.condition_number:.3g}. More samples from "
                    f"this design will tighten every parameter.")
        return (f"NOT identifiable: lambda_min={self.lambda_min:.4g} is at the numerical "
                f"floor along ({self.describe_weak_direction()}). More samples from this "
                f"circuit will not solve this -- the experiment is blind to that "
                f"direction, and repetition does not cure blindness. Change the design.")


def identifiability(fisher: Matrix, parameter_names: Sequence[str] | None = None,
                    tolerance: float = 1e-8) -> Identifiability:
    eigenvalues, eigenvectors = jacobi_eigen(fisher)
    return Identifiability(eigenvalues, eigenvectors,
                           list(parameter_names or []), tolerance)


# --------------------------------------------------------------------------
# Active design: derive the experiment the evidence needs
# --------------------------------------------------------------------------

@dataclass
class DesignCandidate:
    """A candidate experiment, described by the information it would add."""

    candidate_id: str
    fisher_contribution: Matrix
    cost_usd: float = 0.0
    description: str = ""


@dataclass
class DesignChoice:
    candidate: DesignCandidate
    score: float
    criterion: str
    detail: str = ""


def rank_for_direction(current: Matrix, candidates: Sequence[DesignCandidate],
                       direction: Vector) -> list[DesignChoice]:
    """Rank by information gained along a specific weak direction per dollar:

        e* = argmax_e  (v^T F_e v) / C_e

    Use this when identifiability analysis has found ONE direction the
    experiment is blind to. It is much sharper than a global criterion,
    because it optimizes for the thing actually blocking progress rather
    than for average information.
    """
    if not candidates:
        return []
    out = []
    for c in candidates:
        gain = quadratic_form(c.fisher_contribution, direction)
        score = gain if c.cost_usd <= 0 else gain / c.cost_usd
        out.append(DesignChoice(
            c, score, "directional",
            f"adds {gain:.4g} along the weak direction"
            + (f" at ${c.cost_usd:,.2f} ({score:.4g}/$)" if c.cost_usd > 0 else " for free")))
    out.sort(key=lambda d: -d.score)
    return out


def rank_d_optimal(current: Matrix, candidates: Sequence[DesignCandidate],
                   cost_weight: float = 0.0) -> list[DesignChoice]:
    """Rank by the D-optimal criterion:

        argmax_e  log det(F + F_e) - lambda * C_e

    Maximizing log det shrinks the volume of the joint confidence region,
    which is the right objective when no single direction dominates. Note
    it will happily ignore a blind direction if the other directions gain
    enough -- so run identifiability first, and prefer `rank_for_direction`
    when it finds one.
    """
    if not candidates:
        return []
    base = log_det(current)
    out = []
    for c in candidates:
        combined = log_det(add(current, c.fisher_contribution))
        gain = combined - base if math.isfinite(base) else combined
        score = gain - cost_weight * c.cost_usd
        detail = (f"log det {base:.4g} -> {combined:.4g}"
                  if math.isfinite(base) else
                  f"log det -inf -> {combined:.4g} (the candidate makes a singular "
                  f"design identifiable)")
        out.append(DesignChoice(c, score, "D-optimal", detail))
    out.sort(key=lambda d: -d.score)
    return out


def recommend(current: Matrix, candidates: Sequence[DesignCandidate],
              parameter_names: Sequence[str] | None = None,
              cost_weight: float = 0.0,
              tolerance: float = 1e-8) -> tuple[Identifiability, list[DesignChoice]]:
    """Diagnose identifiability, then rank candidates by the criterion that
    matches what is actually wrong.

    A blind direction gets the directional criterion; an
    already-identifiable design gets D-optimal. Choosing the criterion for
    the situation is most of the value -- D-optimal on a singular design
    can recommend an experiment that improves the well-determined
    parameters and leaves the blind direction blind.
    """
    ident = identifiability(current, parameter_names, tolerance)
    if not ident.is_identifiable:
        return ident, rank_for_direction(current, candidates, ident.weak_direction())
    return ident, rank_d_optimal(current, candidates, cost_weight)
