"""The door for people who already ran the experiment.

Every entry point here needed a circuit or a written record. Someone with
counts from a real device -- the person this project is most for -- had
nothing to point at them. This is that door, and it deliberately needs no
qiskit: counts are dictionaries, so a hardware post-mortem runs in the
dependency-free install.

The first thing it computes is the one nobody computes for themselves:
**the shot-noise floor of their own estimate**. A quoted uncertainty
below that floor is impossible, not merely optimistic, and it can be
checked from the counts alone with no model, no method, and no
assumption about the device. The source project quoted 0.115 kcal/mol
from a single submission and later disowned it; a floor computed at
submission time would have said so on the day.

The variance is exact rather than a rule of thumb. Terms sharing a
measurement setting are correlated -- treating them as independent
understates the floor, which is the wrong direction to be wrong in for a
number whose job is to refuse. So for each setting the per-shot value of
its whole contribution is formed from the counts, and its variance taken
directly; settings are independent of each other and add.

What it can then run depends on what arrived. With calibration counts it
can undo readout error. With counts at two or more noise scales it can
extrapolate. It says which of those it could do, and names exactly what
is missing for the rest, because "we could not run ZNE" is useless next
to "submit counts at fold 3 and 5 and we can".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .estimation import EstimationError, term_bases


def _shots(counts: dict) -> float:
    total = sum(counts.values())
    if total <= 0:
        raise EstimationError("a measurement with no shots cannot be analysed")
    return float(total)


def _parity_of(bits: str, qubits) -> int:
    return -1 if sum(int(bits[q]) for q in qubits) % 2 else 1


def per_shot_moments(counts: dict, terms) -> tuple:
    """Mean and variance of one setting's whole contribution, per shot.

    `terms` is a sequence of (coefficient, qubit indices). Their parities
    are read off the SAME bitstring, so their fluctuations are correlated
    and the variance has to be taken on the summed value rather than
    added up term by term. Getting that wrong understates the floor.
    """
    total = _shots(counts)
    mean = 0.0
    mean_square = 0.0
    for bits, n in counts.items():
        value = sum(coefficient * _parity_of(bits, qubits)
                    for coefficient, qubits in terms)
        weight = n / total
        mean += weight * value
        mean_square += weight * value * value
    return mean, max(0.0, mean_square - mean * mean)


def _route(measurements: dict, observable) -> tuple:
    """Assign each term to a setting that can actually measure it.

    Matching by exact label would be wrong and expensive: a ZI term is
    measurable in a ZZ setting, because both read qubit 1 in the Z basis
    and ZI simply ignores qubit 0. Demanding its own setting per term
    would make people collect several times the data they need, and it is
    not how anyone runs an experiment -- one Z-basis measurement yields
    ZZ, ZI and IZ at once.

    A term is measurable in a setting when, on every qubit, the term asks
    for the identity or for exactly the basis that setting reads. Terms
    landing in the same setting are then genuinely correlated, which is
    why the variance is taken on their summed per-shot value.
    """
    grouped: dict = {name: [] for name in measurements}
    constant = 0.0
    setting_bases = {name: term_bases(name) for name in measurements}

    for label, coefficient in observable:
        bases = term_bases(label)
        qubits = tuple(q for q, basis in enumerate(bases) if basis != "I")
        if not qubits:
            constant += float(coefficient)
            continue
        home = None
        for name, available in setting_bases.items():
            if len(available) == len(bases) and all(
                    basis == "I" or basis == available[q]
                    for q, basis in enumerate(bases)):
                home = name
                break
        if home is None:
            raise EstimationError(
                f"term {label!r} cannot be measured by any supplied setting "
                f"({sorted(measurements)}). Add a setting that reads every "
                "non-identity qubit of that term in the basis it asks for, or "
                "drop the term -- a silently dropped term reports a different "
                "observable under the name of the one you asked for.")
        grouped[home].append((float(coefficient), qubits))
    return grouped, constant


@dataclass(frozen=True)
class ShotNoise:
    """The precision the data can support, before any method is applied."""

    estimate: float
    variance: float
    shots: int
    settings: int

    @property
    def sigma(self) -> float:
        return self.variance ** 0.5

    def floor(self, confidence: float = 0.95) -> float:
        """Half-width of the tightest honest interval, in the same units."""
        from .power import normal_quantile
        return normal_quantile(1 - (1 - confidence) / 2) * self.sigma

    def shots_for(self, target: float) -> Optional[int]:
        """Shots needed to reach `target` half-width. None if already there."""
        if target <= 0:
            return None
        if self.floor() <= target:
            return None
        return int(self.shots * (self.floor() / target) ** 2) + 1

    def describe(self, confidence: float = 0.95) -> str:
        return (f"  estimate {self.estimate:.6g} +- {self.floor(confidence):.4g} "
                f"({confidence:.0%}, {self.shots} shots over {self.settings} "
                f"setting{'s' if self.settings != 1 else ''})")


def shot_noise(measurements: dict, observable) -> ShotNoise:
    """The estimate and its shot-noise floor, from counts alone.

    `measurements` maps a setting name to its counts table.
    `observable` is a sequence of (pauli label, coefficient). Each term is
    routed to the setting whose name matches its label; a term with no
    matching setting is refused by name rather than dropped, because a
    silently dropped term is a different observable reported as the one
    you asked for.
    """
    grouped, constant = _route(measurements, observable)

    estimate = constant
    variance = 0.0
    shots = 0
    used = 0
    for name, terms in grouped.items():
        if not terms:
            continue
        counts = measurements[name]
        mean, per_shot_variance = per_shot_moments(counts, terms)
        n = _shots(counts)
        estimate += mean
        variance += per_shot_variance / n
        shots += int(n)
        used += 1
    return ShotNoise(estimate=estimate, variance=variance, shots=shots,
                     settings=used)


@dataclass(frozen=True)
class MethodAvailability:
    """What could be run on this data, and what is missing for the rest."""

    available: tuple
    missing: tuple

    def describe(self) -> str:
        lines = []
        if self.available:
            lines.append("  can run on this data: " + ", ".join(self.available))
        else:
            lines.append("  nothing can be run on this data as submitted")
        for method, need in self.missing:
            lines.append(f"  {method}: needs {need}")
        return "\n".join(lines)


def available_methods(measurements: dict, calibration: Optional[dict] = None,
                      folds: Optional[dict] = None) -> MethodAvailability:
    """Which methods this data supports, and what each missing one needs.

    Names what to submit rather than only what failed. "We could not run
    ZNE" is useless beside "submit counts at fold 3 and 5 and we can".
    """
    available = ["unmitigated"]
    missing = []

    if calibration:
        available.append("REM (readout)")
    else:
        missing.append(("REM (readout)",
                        "calibration counts: prepare all-zeros and all-ones and "
                        "measure each, as `calibration.prepared_0` and "
                        "`calibration.prepared_1`"))

    scales = sorted(folds) if folds else []
    if len(scales) >= 2:
        available.append("ZNE")
        if calibration:
            available.append("REM + ZNE")
    else:
        missing.append(("ZNE",
                        "counts at two or more noise scales, as `folds` keyed by "
                        "fold factor -- 1, 3 and 5 is the usual set. Each scale "
                        "carries its OWN full set of measurement settings, the "
                        "same shape as `measurements`, since the observable "
                        "spans them"))
    return MethodAvailability(tuple(available), tuple(missing))


@dataclass(frozen=True)
class MitigatedEstimate:
    """One method's answer on this data, with its own shot-noise floor."""

    method: str
    estimate: float
    floor: Optional[float] = None
    note: str = ""

    def describe(self) -> str:
        bar = f" +- {self.floor:.4g}" if self.floor is not None else ""
        tail = f"   ({self.note})" if self.note else ""
        return f"    {self.method:<16} {self.estimate:.6g}{bar}{tail}"


@dataclass(frozen=True)
class ResultsReport:
    """What the submitted data supports, and what it cannot."""

    noise: ShotNoise
    availability: MethodAvailability
    claimed_uncertainty: Optional[float] = None
    confidence: float = 0.95
    notes: tuple = field(default_factory=tuple)
    estimates: tuple = field(default_factory=tuple)

    @property
    def claim_is_impossible(self) -> bool:
        """A quoted uncertainty below the shot-noise floor is not optimistic,
        it is unattainable: no analysis of this data can be that precise."""
        return (self.claimed_uncertainty is not None
                and self.claimed_uncertainty < self.noise.floor(self.confidence))

    def format_report(self) -> str:
        lines = ["  from the counts alone, before any method:",
                 self.noise.describe(self.confidence)]
        if self.claimed_uncertainty is not None:
            floor = self.noise.floor(self.confidence)
            if self.claim_is_impossible:
                needed = self.noise.shots_for(self.claimed_uncertainty)
                lines.append(
                    f"  -> the quoted +-{self.claimed_uncertainty:.4g} is BELOW "
                    f"the shot-noise floor of +-{floor:.4g}.")
                lines.append(
                    "     No analysis of this data reaches that precision. "
                    f"About {needed} shots would.")
            else:
                lines.append(f"  -> the quoted +-{self.claimed_uncertainty:.4g} "
                             f"clears the shot-noise floor of +-{floor:.4g}")
        if self.estimates:
            lines.append("  what each available method makes of it:")
            lines.extend(estimate.describe() for estimate in self.estimates)
        lines.append(self.availability.describe())
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)


def analyse(measurements: dict, observable, calibration: Optional[dict] = None,
            folds: Optional[dict] = None,
            claimed_uncertainty: Optional[float] = None,
            confidence: float = 0.95) -> ResultsReport:
    """Read someone's hardware results and say what they support."""
    noise = shot_noise(measurements, observable)
    availability = available_methods(measurements, calibration, folds)
    notes = []

    # A method listed as available has to actually be run. Advertising
    # "can run: REM" and then not running it would be the same unbacked
    # claim this package objects to everywhere else.
    estimates = [MitigatedEstimate("unmitigated", noise.estimate, noise.floor(confidence))]
    corrected = None
    if calibration:
        width = len(next(iter(next(iter(measurements.values())))))
        model = readout_model(calibration, width)
        rem = corrected_shot_noise(measurements, observable, model)
        estimates.append(MitigatedEstimate(
            "REM (tensored)", rem.estimate, rem.floor(confidence),
            f"assumes readout factorises; shot noise amplified "
            f"{rem.sigma / max(noise.sigma, 1e-30):.2f}x by the correction"))

    if folds and len(folds) >= 2:
        scales = sorted(float(scale) for scale in folds)
        raw = [shot_noise(folds[key], observable).estimate
               for key in sorted(folds, key=float)]
        estimates.append(MitigatedEstimate(
            "ZNE", extrapolate_to_zero(scales, raw),
            None, f"linear fit through {len(scales)} scales; no interval, because "
                  "an extrapolation's uncertainty is not its inputs'"))
    notes = []
    if noise.settings > 1:
        notes.append("settings are treated as independent, which they are when "
                     "measured in separate submissions; terms WITHIN a setting "
                     "are correlated and that is accounted for exactly")
    return ResultsReport(noise=noise, availability=availability,
                         claimed_uncertainty=claimed_uncertainty,
                         confidence=confidence, notes=tuple(notes),
                         estimates=tuple(estimates))


def _invert_2x2(matrix) -> tuple:
    """Inverse of [[a, b], [c, d]], refusing a singular readout model."""
    (a, b), (c, d) = matrix
    determinant = a * d - b * c
    if abs(determinant) < 1e-12:
        raise EstimationError(
            "the calibration says the two prepared states are indistinguishable "
            "on some qubit, so readout cannot be undone -- that is a dead qubit "
            "or a mislabelled calibration, not something to correct around")
    return ((d / determinant, -b / determinant),
            (-c / determinant, a / determinant))


def readout_model(calibration: dict, n_qubits: int) -> list:
    """Per-qubit 2x2 readout confusion, from two prepared states.

    Two circuits at any width, against 2**n for a full model. The price is
    the assumption that readout errors factorise across qubits, which
    discards crosstalk between neighbouring resonators -- so this is a
    DIFFERENT correction from a full one, with weaker guarantees, and it
    is named for what it is wherever it appears.
    """
    prepared = {}
    for state, key in ((0, "prepared_0"), (1, "prepared_1")):
        if key not in calibration:
            raise EstimationError(
                f"calibration is missing {key!r}: prepare the all-"
                f"{'zeros' if state == 0 else 'ones'} state, measure it, and "
                "submit those counts")
        counts = calibration[key]
        total = _shots(counts)
        ones = [0.0] * n_qubits
        for bits, n in counts.items():
            if len(bits) != n_qubits:
                raise EstimationError(
                    f"calibration bitstring {bits!r} is {len(bits)} bits but the "
                    f"measurements are {n_qubits}")
            for qubit, bit in enumerate(bits):
                if bit == "1":
                    ones[qubit] += n
        prepared[state] = [count / total for count in ones]

    return [((1 - prepared[0][q], 1 - prepared[1][q]),
             (prepared[0][q], prepared[1][q])) for q in range(n_qubits)]


def apply_readout_correction(counts: dict, model: list) -> dict:
    """Undo readout error one qubit at a time.

    The tensor product of n 2x2 inverses is a 2**n-square matrix; applying
    them one axis at a time is the same arithmetic at 2**n * n cost
    instead of 4**n, and needs no linear algebra library.
    """
    n_qubits = len(model)
    total = _shots(counts)
    distribution = {bits: n / total for bits, n in counts.items()}

    for qubit in range(n_qubits):
        inverse = _invert_2x2(model[qubit])
        updated: dict = {}
        seen = set()
        for bits in list(distribution) + [b for b in distribution]:
            if bits in seen:
                continue
            seen.add(bits)
            partner = bits[:qubit] + ("1" if bits[qubit] == "0" else "0") + bits[qubit + 1:]
            seen.add(partner)
            zero_bits = bits if bits[qubit] == "0" else partner
            one_bits = partner if bits[qubit] == "0" else bits
            p0 = distribution.get(zero_bits, 0.0)
            p1 = distribution.get(one_bits, 0.0)
            updated[zero_bits] = inverse[0][0] * p0 + inverse[0][1] * p1
            updated[one_bits] = inverse[1][0] * p0 + inverse[1][1] * p1
        distribution = updated

    # Inversion can push a probability below zero. Clipping is where this
    # stops being an identity and starts being an approximation, and the
    # honest place to say so is here rather than in a footnote.
    clipped = {bits: max(0.0, p) for bits, p in distribution.items()}
    mass = sum(clipped.values())
    if mass <= 0:
        raise EstimationError("readout correction left no probability mass")
    return {bits: p / mass * total for bits, p in clipped.items() if p > 0}


def extrapolate_to_zero(scales, values, order: int = 1) -> float:
    """Least-squares polynomial fit in the noise scale, evaluated at zero.

    Stdlib, by normal equations. `order` must leave at least one degree of
    freedom -- fitting a line through two points is exact and fine, but
    fitting a parabola through three is interpolation wearing a fit's
    clothes, and the residual it reports would be a fiction.
    """
    if len(scales) != len(values):
        raise EstimationError("each noise scale needs exactly one value")
    if len(scales) < order + 1:
        raise EstimationError(
            f"an order-{order} fit needs at least {order + 1} noise scales, "
            f"got {len(scales)}")
    size = order + 1
    matrix = [[sum(s ** (i + j) for s in scales) for j in range(size)]
              for i in range(size)]
    rhs = [sum(v * s ** i for s, v in zip(scales, values)) for i in range(size)]

    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(matrix[r][column]))
        if abs(matrix[pivot][column]) < 1e-15:
            raise EstimationError(
                "the noise scales are degenerate: extrapolation has nothing to "
                "extrapolate along")
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        rhs[column], rhs[pivot] = rhs[pivot], rhs[column]
        for row in range(size):
            if row == column:
                continue
            factor = matrix[row][column] / matrix[column][column]
            for c in range(column, size):
                matrix[row][c] -= factor * matrix[column][c]
            rhs[row] -= factor * rhs[column]
    # The value at zero is the constant coefficient.
    return rhs[0] / matrix[0][0]


def _weight_vector(n_qubits: int, terms) -> dict:
    """The observable's per-shot value on every bitstring of this width."""
    weights = {}
    for index in range(2 ** n_qubits):
        bits = format(index, f"0{n_qubits}b")
        weights[bits] = sum(coefficient * _parity_of(bits, qubits)
                            for coefficient, qubits in terms)
    return weights


def corrected_moments(counts: dict, terms, model: list) -> tuple:
    """Mean and per-shot variance of the READOUT-CORRECTED contribution.

    The correction is linear, so instead of correcting the counts and
    measuring the spread of the result -- which reports the corrected
    distribution's own spread and has nothing to do with the estimate's
    uncertainty -- the weights are pushed through the transpose of the
    inverse and the moments taken on the ORIGINAL counts.

    Both routes give the same estimate. Only this one gives the right
    error bar, and the difference is not cosmetic: correcting the counts
    first made a mitigated estimate report +- 0 in this module's own
    first draft. Mitigation AMPLIFIES shot noise. A pipeline that
    reported a tighter bar after mitigation would be the exact failure
    this package exists to catch, produced by the package itself.
    """
    n_qubits = len(model)
    weights = _weight_vector(n_qubits, terms)
    for qubit in range(n_qubits):
        inverse = _invert_2x2(model[qubit])
        updated = {}
        for bits, value in weights.items():
            partner = bits[:qubit] + ("1" if bits[qubit] == "0" else "0") + bits[qubit + 1:]
            zero_bits = bits if bits[qubit] == "0" else partner
            one_bits = partner if bits[qubit] == "0" else bits
            w0, w1 = weights[zero_bits], weights[one_bits]
            # transpose of the inverse: column index varies over the source
            updated[bits] = (inverse[0][0] * w0 + inverse[1][0] * w1
                             if bits[qubit] == "0" else
                             inverse[0][1] * w0 + inverse[1][1] * w1)
        weights = updated

    total = _shots(counts)
    mean = 0.0
    mean_square = 0.0
    for bits, n in counts.items():
        value = weights[bits]
        weight = n / total
        mean += weight * value
        mean_square += weight * value * value
    return mean, max(0.0, mean_square - mean * mean)


def corrected_shot_noise(measurements: dict, observable, model: list) -> ShotNoise:
    """`shot_noise`, but for the readout-corrected estimate.

    Same routing of terms to settings; the difference is that each
    setting's moments come from `corrected_moments`, which carries the
    correction's noise amplification instead of discarding it.
    """
    grouped, constant = _route(measurements, observable)

    estimate = constant
    variance = 0.0
    shots = 0
    used = 0
    for name, terms in grouped.items():
        if not terms:
            continue
        counts = measurements[name]
        mean, per_shot_variance = corrected_moments(counts, terms, model)
        n = _shots(counts)
        estimate += mean
        variance += per_shot_variance / n
        shots += int(n)
        used += 1
    return ShotNoise(estimate=estimate, variance=variance, shots=shots,
                     settings=used)
