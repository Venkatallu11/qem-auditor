"""The control experiment: is the effect the mechanism, or the apparatus?

Ported from the `falsify` capability in Venkat's `quantum-verifier`, whose
idea is better than anything this package had for the same job. Given a
circuit claiming an effect, build a SECOND circuit that removes the
entangling mechanism the claim depends on and keeps everything else --
qubit count, single-qubit gates, measurement structure -- identical. Run
both. Confounds that afflict both circuits equally, readout bias and SPAM
above all, cancel out of the difference.

Its real advantage over this package's existing attacks is that it works
in discovery mode. `data_sensitivity` scrambles the outcome labels and
asks whether a method's answer was ever a function of the data, which
needs no known answer either -- but it audits a METHOD. This audits a
CLAIM, and it does so without anyone knowing what the right answer was.

Two things are added here, both because the ported version reports
numbers without saying which of them are real:

**An effect size is not a finding until it clears its own noise.** The
original reports the isolated effect and interprets it in prose. At 4096
shots an isolated effect of 0.01 is indistinguishable from zero and 0.35
is not, and nothing in the report separated them.

**Total variation distance, in discovery mode, is badly biased upward.**
Two INDEPENDENT samples of the SAME distribution do not give TVD zero;
they give roughly sqrt(outcomes / shots). Measured, at 4096 shots:

    qubits   outcomes   median TVD comparing a distribution to ITSELF
         3          8   0.022
         8        256   0.139
        10       1024   0.278
        12       4096   0.523

So at twelve qubits a report reading "TVD 0.52, where 0 is identical and
1 is completely different" describes two identical distributions. The
scale is not wrong, it is just not anchored at zero, and the anchor moves
with width and shots. Here the null is measured by permutation -- pool
both runs' shots, re-split them at random, and see what TVD that gives --
and the answer is quoted against it rather than against zero.

Stdlib only. Counts in, verdict out: the caller runs the circuits, so
this works on real hardware results and not only on a simulator.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Optional

from .power import normal_quantile

#: Two-qubit entangling gates, across both vendors' native sets. Taken
#: from quantum-verifier, which collected them from circuits that
#: actually ran on IBM and IonQ hardware.
ENTANGLING_GATES = frozenset({
    "rzz", "cx", "cz", "ecr", "zz", "ms", "cnot", "swap", "cy", "ch",
    "cp", "crx", "cry", "crz", "iswap", "rxx", "ryy", "cu", "csx",
})


class ControlError(ValueError):
    """This comparison cannot isolate what it is being asked to isolate."""


def build_control(circuit: Any) -> tuple:
    """Strip every entangling gate, keep everything else exactly.

    Returns `(control_circuit, removed)`. The counts, the single-qubit
    gates and the measurement structure are preserved, so the only thing
    that differs between the two circuits is the mechanism under test.
    """
    from qiskit import QuantumCircuit

    control = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    removed = 0
    for instruction in circuit.data:
        if instruction.operation.name.lower() in ENTANGLING_GATES:
            removed += 1
            continue
        control.append(
            instruction.operation,
            [circuit.find_bit(q).index for q in instruction.qubits],
            [circuit.find_bit(c).index for c in instruction.clbits])
    if removed == 0:
        raise ControlError(
            "no entangling gate was removed, so the control is the circuit "
            "itself and the comparison isolates nothing. A claim that does "
            "not depend on entanglement needs a different control -- one "
            "that removes whatever mechanism it DOES depend on.")
    return control, removed


def _fraction(counts: dict, marked) -> tuple:
    total = sum(counts.values())
    if total <= 0:
        raise ControlError("a run with no shots cannot be compared")
    hits = sum(n for bits, n in counts.items() if bits in set(marked))
    return hits / total, total


@dataclass(frozen=True)
class IsolatedEffect:
    """What the mechanism contributes, and whether that is more than noise."""

    effect: float
    circuit_signal: float
    control_signal: float
    circuit_shots: int
    control_shots: int
    removed: int
    confidence: float = 0.95

    @property
    def standard_error(self) -> float:
        """Of the difference of two proportions."""
        p, q = self.circuit_signal, self.control_signal
        return (p * (1 - p) / self.circuit_shots
                + q * (1 - q) / self.control_shots) ** 0.5

    @property
    def half_width(self) -> float:
        return normal_quantile(1 - (1 - self.confidence) / 2) * self.standard_error

    @property
    def real(self) -> bool:
        """Does the isolated effect clear its own uncertainty?"""
        return abs(self.effect) > self.half_width

    @property
    def shots_for_a_signal(self) -> Optional[int]:
        """Shots per circuit that would resolve this effect, if it is real.

        None for an effect of exactly zero: no finite number of shots
        establishes that a mechanism contributes nothing.
        """
        if self.effect == 0:
            return None
        p, q = self.circuit_signal, self.control_signal
        z = normal_quantile(1 - (1 - self.confidence) / 2) + normal_quantile(0.80)
        variance = p * (1 - p) + q * (1 - q)
        return max(1, int((z / self.effect) ** 2 * variance) + 1)

    def describe(self) -> str:
        head = (f"  removed {self.removed} entangling gates\n"
                f"  circuit {self.circuit_signal:.4f} vs control "
                f"{self.control_signal:.4f}\n"
                f"  isolated effect {self.effect:+.4f} "
                f"+- {self.half_width:.4f}")
        if self.real:
            return head + "\n  -> the mechanism contributes; SPAM and readout bias " \
                          "affect both circuits and cancel here"
        needed = self.shots_for_a_signal
        tail = ("the effect is exactly zero, and no number of shots establishes "
                "that a mechanism contributes nothing"
                if needed is None else
                f"about {needed} shots per circuit would resolve it")
        return head + (f"\n  -> NOT distinguishable from zero; {tail}")


def isolate_effect(circuit_counts: dict, control_counts: dict, marked,
                   removed: int, confidence: float = 0.95) -> IsolatedEffect:
    """The confound-free effect size, with the bar it has to clear."""
    circuit_signal, circuit_shots = _fraction(circuit_counts, marked)
    control_signal, control_shots = _fraction(control_counts, marked)
    return IsolatedEffect(
        effect=circuit_signal - control_signal,
        circuit_signal=circuit_signal, control_signal=control_signal,
        circuit_shots=circuit_shots, control_shots=control_shots,
        removed=removed, confidence=confidence)


def total_variation(a: dict, b: dict) -> float:
    """TVD between two counts tables, each normalised to its own total."""
    total_a, total_b = sum(a.values()), sum(b.values())
    if total_a <= 0 or total_b <= 0:
        raise ControlError("a run with no shots cannot be compared")
    return 0.5 * sum(abs(a.get(k, 0) / total_a - b.get(k, 0) / total_b)
                     for k in set(a) | set(b))


@dataclass(frozen=True)
class DistributionShift:
    """How far the circuit's output is from its control's, against the null.

    `observed` alone is uninterpretable: at twelve qubits and 4096 shots,
    two draws from the SAME distribution sit around 0.52. `null_median`
    is what this comparison gives when nothing has changed, measured by
    permutation on these very counts, so it already accounts for this
    run's width and shot count rather than a rule of thumb.
    """

    observed: float
    null_median: float
    null_high: float
    replicates: int
    gainers: tuple

    @property
    def excess(self) -> float:
        return self.observed - self.null_median

    @property
    def real(self) -> bool:
        return self.observed > self.null_high

    def describe(self) -> str:
        lines = [f"  TVD {self.observed:.4f}",
                 f"  same-distribution null: {self.null_median:.4f} "
                 f"(95th percentile {self.null_high:.4f}, "
                 f"{self.replicates} permutations)"]
        if self.real:
            lines.append(f"  -> excess {self.excess:+.4f} over the null: the "
                         "entangling mechanism moves the distribution")
            if self.gainers:
                lines.append("  most boosted by entanglement: "
                             + ", ".join(self.gainers))
        else:
            lines.append("  -> inside what identical distributions give at this "
                         "width and shot count; no shift is established")
            lines.append("     Reading this TVD against zero would report noise "
                         "as a finding.")
        return "\n".join(lines)


def distribution_shift(circuit_counts: dict, control_counts: dict,
                       replicates: int = 200, seed: int = 0,
                       top: int = 5) -> DistributionShift:
    """Discovery mode: did the mechanism move the distribution at all?

    The null is built by permutation -- pool every shot from both runs,
    re-split into two groups of the original sizes, and measure TVD. That
    is what this comparison returns when the mechanism changed nothing,
    and it is the number the observed TVD has to beat.

    Candidate gainers are reported ONLY when the shift clears the null.
    Ranking outcomes by gain and printing the top five is a selection
    over thousands of noisy differences; the largest of them is large
    whether or not anything happened, so the list is withheld rather
    than offered as "candidate answers" when the shift is not real.
    """
    observed = total_variation(circuit_counts, control_counts)

    pool = []
    for table in (circuit_counts, control_counts):
        for bits, n in table.items():
            pool.extend([bits] * int(round(n)))
    size_a = int(round(sum(circuit_counts.values())))
    rng = random.Random(seed)

    nulls = []
    for _ in range(replicates):
        rng.shuffle(pool)
        left, right = {}, {}
        for i, bits in enumerate(pool):
            table = left if i < size_a else right
            table[bits] = table.get(bits, 0) + 1
        nulls.append(total_variation(left, right))
    nulls.sort()
    null_median = nulls[len(nulls) // 2]
    null_high = nulls[min(len(nulls) - 1, int(0.95 * len(nulls)))]

    gainers = ()
    if observed > null_high:
        total_c = sum(circuit_counts.values())
        total_k = sum(control_counts.values())
        gains = sorted(set(circuit_counts) | set(control_counts),
                       key=lambda k: (circuit_counts.get(k, 0) / total_c
                                      - control_counts.get(k, 0) / total_k),
                       reverse=True)
        gainers = tuple(gains[:top])

    return DistributionShift(observed=observed, null_median=null_median,
                             null_high=null_high, replicates=replicates,
                             gainers=gainers)
