"""Does the circuit compute what its author says it computes?

This package audits error-mitigation claims, and every one of those
claims presupposes something nobody had been checking: that the circuit
under test is the circuit the author described. The first outside circuit
brought here -- an 18-qubit phase oracle with a specification attached --
marked **one** of the 1097 basis states its own specification named, and
left its ancillas entangled on all 4096 inputs. No amount of readout
mitigation improves that number, and a mitigation report on such a
circuit is a precise answer to the wrong question.

So this runs first, and it is exact rather than statistical.

The leverage: an oracle, an arithmetic block, a reversible classical
subroutine -- the parts of a quantum program most likely to carry a
written specification -- are built from gates that only ever permute
computational basis states and multiply by a phase. X, CX, CCX, MCX, Z
and their multi-controlled forms are all of that kind. A circuit made of
them can be evaluated one basis state at a time on a classical machine,
tracking a bitstring and a sign, and never touching a state vector. That
is what makes 18 qubits cost 4096 cheap evaluations instead of 262144
amplitudes, and it is why this check is affordable enough to be
mandatory.

What is deliberately NOT here: any attempt to guess the specification.
The predicate is supplied by the person making the claim. This module
checks a stated claim against the circuit; it does not invent the claim,
because an auditor that writes both sides of the comparison is not
auditing anything.

Stdlib only, plus whatever circuit object the caller passes -- the gates
are read by name and qubit index, so any library whose circuit can be
walked works.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

#: Gates that permute basis states, mapped to their control count. A
#: multi-controlled form is matched by pattern, so `c12z` and `mcx`
#: arrive here without being enumerated.
_NEGATION = re.compile(r"^(x|cx|ccx|c\d*x|mcx|toffoli|not|cnot)$")
_PHASE = re.compile(r"^(z|cz|ccz|c\d*z|mcz)$")


class NotReversible(ValueError):
    """This circuit does more than permute basis states and add phases.

    Raised by name rather than skipped, because the alternative -- an
    exact-sounding verdict computed by ignoring the gates it could not
    handle -- is the failure mode this package exists to prevent.
    """


@dataclass(frozen=True)
class Discrepancy:
    """One way the circuit and the specification disagree."""

    kind: str
    count: int
    total: int
    detail: str

    @property
    def fraction(self) -> float:
        return self.count / self.total if self.total else 0.0

    def __str__(self) -> str:
        return (f"{self.kind}: {self.count} of {self.total} "
                f"({self.fraction:.1%}) -- {self.detail}")


@dataclass(frozen=True)
class OracleReport:
    """What the circuit does, against what it was said to do.

    `marked` and `expected` are the sets of inputs that receive a minus
    sign and that the specification says should. Everything else is
    derived from those two sets, so a reader can recompute any number
    here rather than take it on faith.
    """

    n_inputs: int
    marked: frozenset
    expected: frozenset
    dirty_ancillas: int
    altered_inputs: int
    discrepancies: tuple

    @property
    def false_negatives(self) -> frozenset:
        return self.expected - self.marked

    @property
    def false_positives(self) -> frozenset:
        return self.marked - self.expected

    @property
    def is_a_phase_oracle(self) -> bool:
        """A phase oracle must return the ancillas to zero and leave the
        input alone. One that does not is still a unitary, but it is an
        entangler: used inside amplitude amplification it destroys the
        interference the algorithm runs on, and the failure shows up as a
        search that never converges rather than as an error message."""
        return self.dirty_ancillas == 0 and self.altered_inputs == 0

    @property
    def matches_specification(self) -> bool:
        return self.is_a_phase_oracle and self.marked == self.expected

    @property
    def accuracy(self) -> float:
        """Fraction of inputs given the right phase. Reported alongside
        the counts and never instead of them: a specification marking
        1097 of 4096 states is matched to 73% accuracy by a circuit that
        does nothing at all, so this number alone flatters a failure."""
        wrong = len(self.false_negatives) + len(self.false_positives)
        return (self.n_inputs - wrong) / self.n_inputs if self.n_inputs else 0.0

    def format_report(self) -> str:
        lines = [f"  inputs checked:   {self.n_inputs} (exhaustive, exact)",
                 f"  specification:    {len(self.expected)} marked",
                 f"  circuit marks:    {len(self.marked)}"]
        if self.matches_specification:
            lines.append("  -> the circuit implements its specification")
            return "\n".join(lines)
        for discrepancy in self.discrepancies:
            lines.append(f"  {discrepancy}")
        lines.append("  -> the circuit does NOT implement its specification; "
                     "no mitigation method addresses this")
        return "\n".join(lines)


def _gate_sequence(circuit: Any) -> list:
    """Read a circuit down to (name, qubit indices).

    Written against qiskit's shape but deliberately duck-typed: anything
    exposing `.data` with `.operation.name` and `.qubits`, or plain
    `(name, indices)` pairs, walks through unchanged.
    """
    if isinstance(circuit, (list, tuple)):
        return [(str(name).lower(), list(qubits)) for name, qubits in circuit]
    sequence = []
    for instruction in circuit.data:
        name = instruction.operation.name.lower()
        indices = [circuit.find_bit(bit).index for bit in instruction.qubits]
        sequence.append((name, indices))
    return sequence


def evaluate(circuit: Any, bits: Sequence[int]) -> tuple:
    """Run one basis state through. Returns (output bits, sign).

    Refuses on the first gate that is not a permutation or a phase,
    naming it. A Hadamard here is not an error in the circuit -- it means
    the circuit is not the kind of object this check applies to, and the
    caller needs a different tool, which is worth saying out loud.
    """
    state = list(bits)
    sign = 1
    for name, qubits in _gate_sequence(circuit):
        if _NEGATION.match(name):
            if all(state[control] for control in qubits[:-1]):
                state[qubits[-1]] ^= 1
        elif _PHASE.match(name):
            if all(state[qubit] for qubit in qubits):
                sign = -sign
        elif name in ("barrier", "id", "delay"):
            continue
        else:
            raise NotReversible(
                f"gate {name!r} neither permutes basis states nor adds a "
                "phase, so this circuit cannot be checked exactly -- it "
                "needs simulation, not enumeration")
    return tuple(state), sign


def audit_oracle(circuit: Any, predicate: Callable, n_inputs: int,
                 encode: Callable, ancillas: Optional[Sequence[int]] = None,
                 inputs: Optional[Sequence] = None) -> OracleReport:
    """Check a phase oracle against the predicate it claims to implement.

    `predicate(value) -> bool` is the specification, supplied by whoever
    is making the claim. `encode(value) -> bits` places one input into
    the register. Both are the caller's, on purpose: the encoding of a
    coordinate into qubits is exactly the kind of convention that gets
    written down one way in a comment and implemented another, and
    guessing it here would let this check pass a circuit whose real
    defect is the guess.

    Exhaustive over `inputs` -- so the verdict is a proof over the input
    space, not a sample of it.
    """
    values = list(range(n_inputs)) if inputs is None else list(inputs)
    ancilla_set = list(ancillas) if ancillas is not None else []
    # Read the circuit once. Re-deriving it per input made the 4096-state
    # audit of a 111-gate oracle take 24 seconds for no reason: the
    # circuit does not change between inputs.
    program = _gate_sequence(circuit)

    marked, expected = set(), set()
    dirty = altered = 0
    for value in values:
        start = list(encode(value))
        out, sign = evaluate(program, start)
        if any(out[a] for a in ancilla_set):
            dirty += 1
        data = [i for i in range(len(start)) if i not in set(ancilla_set)]
        if [out[i] for i in data] != [start[i] for i in data]:
            altered += 1
        if sign == -1:
            marked.add(value)
        if predicate(value):
            expected.add(value)

    total = len(values)
    discrepancies = []
    if dirty:
        discrepancies.append(Discrepancy(
            "ancillas not restored", dirty, total,
            "the ancillas stay entangled with the input, so this is not a "
            "phase oracle and amplitude amplification will not work on it"))
    if altered:
        discrepancies.append(Discrepancy(
            "input register altered", altered, total,
            "the oracle must leave its input unchanged"))
    missing = expected - marked
    if missing:
        discrepancies.append(Discrepancy(
            "specified but unmarked", len(missing), total,
            "states the specification marks that the circuit leaves alone"))
    spurious = marked - expected
    if spurious:
        discrepancies.append(Discrepancy(
            "marked but not specified", len(spurious), total,
            "states the circuit marks that the specification does not"))

    return OracleReport(
        n_inputs=total,
        marked=frozenset(marked),
        expected=frozenset(expected),
        dirty_ancillas=dirty,
        altered_inputs=altered,
        discrepancies=tuple(discrepancies),
    )
