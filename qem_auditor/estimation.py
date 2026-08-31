"""Measuring a general observable, on somebody else's circuit.

The measurement layer this project grew up with assumed every term of the
Hamiltonian was all-Z or all-X, which is true of H2 in the encoding used
and of the Ising chain, and false of most things. Worse, it did not
CHECK: it took the set of bases a term needed and popped one, so an
observable with a term like XYZ got measured in an arbitrary single basis
and returned a number that was wrong rather than absent. Its own
docstring said that silently averaging would be wrong; the code did it
anyway.

That is the whole reason this module exists. An engine that sits between
people and quantum hardware has to work on the circuit somebody actually
brings, and the first circuit brought here that was not H2 or Ising would
have been quietly mismeasured.

So: general Pauli estimation. Terms are grouped into commuting sets that
can share one measurement circuit, each qubit is rotated into the basis
that term needs, and anything that cannot be measured is refused by name
rather than approximated.

Stdlib plus qiskit, since rotating a circuit into a measurement basis
needs a circuit.
"""
from __future__ import annotations

from typing import Any, Optional


class EstimationError(ValueError):
    """Something about this observable cannot be measured as asked."""


def term_bases(label: str) -> tuple:
    """Per-qubit measurement basis for one Pauli term, little-endian.

    Returns a tuple indexed by qubit number, holding 'X', 'Y', 'Z' or
    'I'. Qiskit labels read qubit n-1 first, which is reversed here so
    that position i is qubit i -- getting that backwards is the kind of
    silent index error this package exists to catch, and it is verified
    against a statevector in the tests.
    """
    if not label or any(c not in "IXYZ" for c in label):
        raise EstimationError(
            f"{label!r} is not a Pauli string over I, X, Y and Z")
    return tuple(reversed(label))


def compatible(a: tuple, b: tuple) -> bool:
    """Can these two terms be measured by the same circuit?

    Only if no qubit is asked for two different bases. 'I' imposes
    nothing, so it is compatible with everything -- which is what makes
    grouping worth doing at all.
    """
    if len(a) != len(b):
        raise EstimationError(
            f"terms span different numbers of qubits: {len(a)} and {len(b)}")
    return all(x == "I" or y == "I" or x == y for x, y in zip(a, b))


def merge(a: tuple, b: tuple) -> tuple:
    return tuple(x if x != "I" else y for x, y in zip(a, b))


def group_terms(labels) -> list:
    """Assign every term to a measurement setting, greedily.

    Greedy first-fit rather than an optimal grouping. The optimal version
    is a graph colouring and the difference is a handful of extra
    circuits; presenting a heuristic as optimal would be the kind of
    quiet overstatement this package objects to, so it is named here
    instead. Terms are sorted by how constrained they are, most first,
    because placing the fussiest terms while there is still room is what
    makes first-fit behave.
    """
    settings: list = []
    assignment: dict = {}
    ordered = sorted(set(labels),
                     key=lambda s: (-sum(1 for c in s if c != "I"), s))
    for label in ordered:
        bases = term_bases(label)
        if all(b == "I" for b in bases):
            assignment[label] = None          # the identity needs no circuit
            continue
        for index, setting in enumerate(settings):
            if compatible(setting, bases):
                settings[index] = merge(setting, bases)
                assignment[label] = index
                break
        else:
            settings.append(bases)
            assignment[label] = len(settings) - 1
    return [tuple(s) for s in settings], assignment


def rotate_into_basis(circuit: Any, setting: tuple) -> Any:
    """Append the rotations that turn a basis measurement into this one.

    X is measured by an H before the Z measurement; Y by Sdg then H. Z
    needs nothing, and neither does I -- a qubit no term cares about is
    still measured, and its outcome is simply never read.
    """
    from qiskit import QuantumCircuit

    if circuit.num_qubits != len(setting):
        raise EstimationError(
            f"the circuit has {circuit.num_qubits} qubits and the measurement "
            f"setting covers {len(setting)}")
    rotated = circuit.copy()
    for qubit, basis in enumerate(setting):
        if basis == "X":
            rotated.h(qubit)
        elif basis == "Y":
            rotated.sdg(qubit)
            rotated.h(qubit)
        elif basis not in ("Z", "I"):
            raise EstimationError(f"{basis!r} is not a measurement basis")
    rotated.measure_all()
    return rotated


def parity(counts: dict, qubits) -> float:
    """<Z...Z> on `qubits`, from bitstrings indexed by qubit number."""
    shots = sum(counts.values())
    if not shots:
        raise EstimationError(
            "no surviving shots: this estimator has no data to average")
    if not qubits:
        return 1.0
    return sum((-1 if sum(int(bits[q]) for q in qubits) % 2 else 1) * n
               for bits, n in counts.items()) / shots


def expectation(counts_by_setting: list, operator: Any,
                assignment: Optional[dict] = None) -> float:
    """<O> from one counts table per measurement setting."""
    labels = list(operator.paulis.to_labels())
    if assignment is None:
        _, assignment = group_terms(labels)
    total = 0.0
    for label, coefficient in zip(labels, operator.coeffs):
        index = assignment.get(label)
        if index is None:
            total += float(coefficient.real)     # the identity term
            continue
        bases = term_bases(label)
        qubits = [q for q, b in enumerate(bases) if b != "I"]
        total += float(coefficient.real) * parity(counts_by_setting[index], qubits)
    return total


def predicate_expectation(counts: dict, predicate: Any,
                          decode: Optional[Any] = None) -> float:
    """Probability that a shot satisfies `predicate`, from one Z-basis table.

    The observable that matters for an oracle, a Grover search or any
    algorithm whose answer is "did we land in the right set" is the
    indicator function of that set. It is diagonal, so a single Z-basis
    setting measures it exactly -- but it has no useful Pauli expansion:
    the 1097-state marked set of the first outside circuit brought here
    would need thousands of terms to write as a sum of Pauli Zs, and
    `expectation` would dutifully estimate every one of them.

    So this takes the classical function directly. One setting, any
    width, exact in the shot-noise limit. `decode` turns a bitstring
    into whatever the predicate wants to see -- a coordinate pair, an
    integer -- and defaults to the integer the bitstring spells,
    little-endian, matching `evaluate` in `qem_auditor.reversible`.

    Counts with negative weights are accepted, because readout
    mitigation produces them: inverting a confusion matrix can push a
    quasi-probability below zero, and clipping those to zero here would
    silently bias the result back toward the unmitigated value -- which
    would make mitigation look like it did less than it did.
    """
    shots = sum(counts.values())
    if shots == 0:
        raise EstimationError(
            "no surviving shots: this estimator has no data to average")
    if decode is None:
        def decode(bits):
            return int(bits[::-1], 2)
    hits = 0.0
    for bits, n in counts.items():
        if predicate(decode(bits)):
            hits += n
    return hits / shots
