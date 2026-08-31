"""A phase oracle over a 2D bitmap, and what a correct one costs.

Built when the first circuit from outside this project arrived: an
18-qubit phase oracle for a 64x64 logo, with a specification claiming
1097 marked pixels, a depth target of 726, and `"verified": true`.
`qem_auditor.reversible` found it marked one pixel, the wrong one, and
left its ancillas entangled on every input.

This module exists to answer the obvious follow-up -- what SHOULD it have
cost -- because that is the number that turns "your circuit is wrong"
into something actionable. It builds the same specification correctly and
reports the price, and the price is the finding: a depth target is easy
to hit if the function is not implemented.

The construction is deliberately the simplest thing that is obviously
right, not the cleverest thing that might be:

  * The marked set is cut into DISJOINT rectangles, each rectangle into
    prefix-aligned cubes -- a cube being a set of bits fixed and the rest
    free, which is exactly what a multi-controlled gate tests.
  * Because the cubes are disjoint, at most one matches any input, so the
    phases cannot double-count and the oracle is a plain sequence of
    multi-controlled Z gates.
  * That leaves NO ancillas, which removes the entire failure mode the
    uploaded circuit died of. There is nothing to uncompute, so there is
    no uncompute to get wrong.

Correctness here is checked exhaustively over all 4096 inputs by the
same module that audits anyone else's, and by the same call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZGate

#: The four shapes the uploaded specification named, transcribed exactly.
#: Kept as the SPEC, separate from any circuit, so the two can disagree.
LOGO_SHAPES = (
    ("Square", lambda x, y: 2 <= x <= 26 and 29 <= y <= 53),
    ("Bar", lambda x, y: 26 <= x <= 49 and 39 <= y <= 43),
    ("Disk1", lambda x, y: (x - 55) ** 2 + (y - 41) ** 2 <= 42),
    ("Disk2", lambda x, y: (x - 40) ** 2 + (y - 19) ** 2 <= 72),
)

GRID = 64
COORDINATE_BITS = 6


def logo_predicate(x: int, y: int) -> bool:
    """The specification: union of the four shapes."""
    return any(shape(x, y) for _, shape in LOGO_SHAPES)


def marked_pixels(predicate: Callable = logo_predicate) -> frozenset:
    return frozenset((x, y) for x in range(GRID) for y in range(GRID)
                     if predicate(x, y))


def disjoint_rectangles(predicate: Callable = logo_predicate) -> list:
    """Cut the marked set into non-overlapping rectangles.

    Greedy and not minimal. Minimality is not worth pursuing here: a
    smaller cover would make the circuit shallower without making it more
    correct, and this module's job is to establish what correct costs at
    all, honestly, rather than to compete on depth.
    """
    remaining = set(marked_pixels(predicate))
    rectangles = []
    while remaining:
        x0, y0 = min(remaining)
        y1 = y0
        while (x0, y1 + 1) in remaining:
            y1 += 1
        x1 = x0
        while all((x1 + 1, y) in remaining for y in range(y0, y1 + 1)):
            x1 += 1
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                remaining.discard((x, y))
        rectangles.append((x0, x1, y0, y1))
    return rectangles


def _prefix_blocks(low: int, high: int) -> list:
    """Split [low, high] into power-of-two aligned blocks (value, size).

    A block whose size is 2**k has its low k bits free and the rest
    fixed, which is precisely a multi-controlled gate's control pattern.
    """
    blocks = []
    while low <= high:
        size = 1
        while low % (size * 2) == 0 and low + size * 2 - 1 <= high:
            size *= 2
        blocks.append((low, size))
        low += size
    return blocks


@dataclass(frozen=True)
class Cube:
    """A set of inputs with some bits fixed and the rest free."""

    x_value: int
    x_size: int
    y_value: int
    y_size: int

    def controls(self) -> list:
        """(qubit, required bit) pairs. x is little-endian on q[0:6],
        y little-endian on q[6:12] -- stated here once and used by both
        the builder and the encoder so they cannot drift apart."""
        pairs = []
        for bit in range(COORDINATE_BITS):
            if self.x_size >> bit == 0 or (1 << bit) >= self.x_size:
                pairs.append((bit, (self.x_value >> bit) & 1))
        for bit in range(COORDINATE_BITS):
            if self.y_size >> bit == 0 or (1 << bit) >= self.y_size:
                pairs.append((COORDINATE_BITS + bit, (self.y_value >> bit) & 1))
        return pairs

    def contains(self, x: int, y: int) -> bool:
        return (self.x_value <= x < self.x_value + self.x_size
                and self.y_value <= y < self.y_value + self.y_size)


def cube_cover(predicate: Callable = logo_predicate) -> list:
    """Disjoint cubes covering the marked set exactly."""
    cubes = []
    for x0, x1, y0, y1 in disjoint_rectangles(predicate):
        for x_value, x_size in _prefix_blocks(x0, x1):
            for y_value, y_size in _prefix_blocks(y0, y1):
                cubes.append(Cube(x_value, x_size, y_value, y_size))
    return cubes


def encode(value: int) -> list:
    """Input index -> qubit bits. value = x * 64 + y."""
    x, y = divmod(value, GRID)
    bits = [0] * (2 * COORDINATE_BITS)
    for bit in range(COORDINATE_BITS):
        bits[bit] = (x >> bit) & 1
        bits[COORDINATE_BITS + bit] = (y >> bit) & 1
    return bits


def build_oracle(predicate: Callable = logo_predicate,
                 cubes: Optional[list] = None) -> QuantumCircuit:
    """The phase oracle: one multi-controlled Z per disjoint cube.

    No ancillas, so `|x,y> -> (-1)^L(x,y)|x,y>` holds by construction
    rather than by an uncompute that has to be right.
    """
    cubes = cube_cover(predicate) if cubes is None else cubes
    width = 2 * COORDINATE_BITS
    circuit = QuantumCircuit(width)
    for cube in cubes:
        pairs = cube.controls()
        zeros = [qubit for qubit, bit in pairs if bit == 0]
        qubits = [qubit for qubit, _ in pairs]
        if zeros:
            circuit.x(zeros)
        if len(qubits) == 1:
            circuit.z(qubits[0])
        else:
            circuit.append(ZGate().control(len(qubits) - 1), qubits)
        if zeros:
            circuit.x(zeros)
    return circuit
