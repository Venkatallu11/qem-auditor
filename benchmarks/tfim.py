"""A second physical system, so the findings stop resting on one molecule.

Every quantitative claim this project has made -- that readout error
dominates and defeats ZNE, that ZNE and REM change places between noise
models, that CDR is the robust one -- was measured on H2 in STO-3G on two
qubits with two CX gates. That is one system. Findings from one system
are a hypothesis about the world, and this package is not entitled to
state them as more than that without checking.

The transverse-field Ising model is the right second system for three
reasons.

Its Hamiltonian is CONSTRUCTED here from a formula rather than
transcribed from a paper. Every number is generated and the exact ground
state comes from diagonalising the operator, so there is nothing to get
wrong in copying and nothing to take on trust -- which matters, because
the papers are unreachable from this sandbox and a transcribed number
would be exactly the unverified claim this package exists to reject.

Its DEPTH IS A KNOB. H2 has two CX gates, which is why its error budget
is readout-dominated: readout error is charged once per measured qubit
however few gates ran. A Trotterised evolution has as many two-qubit
gates as you ask for, so the same code can produce a gate-dominated
budget and a readout-dominated one. That turns "is our finding general?"
from an opinion into a measurement -- and it tests the mechanism claim
directly, since the claim is precisely that some errors scale with gate
count and others do not.

And it is the model family that real hardware experiments actually run,
including the 127-qubit one whose classical simulability was argued over
for a year.

    H = -J sum_i Z_i Z_{i+1} - h sum_i X_i
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector


def hamiltonian(n_spins: int, coupling: float = 1.0,
                field: float = 0.5, periodic: bool = False) -> SparsePauliOp:
    """The TFIM operator, built term by term.

    `periodic` closes the chain. Left off by default because an open
    chain maps to a line on a heavy-hex lattice without routing, and a
    placement that silently inserts swaps is a different circuit.
    """
    if n_spins < 2:
        raise ValueError("a spin chain needs at least two spins")
    terms = []
    for i in range(n_spins - 1 + (1 if periodic else 0)):
        j = (i + 1) % n_spins
        label = ["I"] * n_spins
        label[n_spins - 1 - i] = "Z"
        label[n_spins - 1 - j] = "Z"
        terms.append(("".join(label), -coupling))
    for i in range(n_spins):
        label = ["I"] * n_spins
        label[n_spins - 1 - i] = "X"
        terms.append(("".join(label), -field))
    return SparsePauliOp.from_list(terms)


def exact_ground_energy(operator: SparsePauliOp) -> float:
    return float(np.linalg.eigvalsh(operator.to_matrix())[0])


def trotter_circuit(n_spins: int, steps: int, dt: float = 0.3,
                    coupling: float = 1.0, field: float = 0.5) -> QuantumCircuit:
    """One Trotterised evolution, `steps` deep.

    Two CX per bond per step, so the two-qubit gate count is
    2 * (n_spins - 1) * steps and depth is the knob this file exists to
    turn. Starts from |+>^n, which has support across the whole
    computational basis -- a start state concentrated on one bitstring
    would make readout error look artificially cheap.
    """
    if steps < 1:
        raise ValueError("an evolution of zero steps is not an evolution")
    qc = QuantumCircuit(n_spins)
    qc.h(range(n_spins))
    for _ in range(steps):
        for i in range(n_spins - 1):
            qc.cx(i, i + 1)
            qc.rz(-2.0 * coupling * dt, i + 1)
            qc.cx(i, i + 1)
        for i in range(n_spins):
            qc.rx(-2.0 * field * dt, i)
    return qc


def exact_expectation(circuit: QuantumCircuit, operator: SparsePauliOp) -> float:
    """What the circuit actually prepares, exactly.

    The target for error measurement is this, not the ground energy: the
    circuit prepares an evolved state, and grading it against the ground
    state would charge the mitigation for the ansatz's distance from it.
    """
    return float(Statevector(circuit).expectation_value(operator).real)


def gate_counts(circuit: QuantumCircuit) -> tuple:
    two = sum(1 for inst in circuit.data if len(inst.qubits) >= 2)
    one = sum(1 for inst in circuit.data if len(inst.qubits) == 1)
    return two, one


def near_clifford_variants(n_spins: int, steps: int, dt: float,
                           coupling: float, field: float,
                           n_variants: int = 5, seed: int = 7) -> list:
    """Training circuits for CDR: the same circuit with most of its
    rotations snapped to Clifford angles.

    The first attempt at this replaced dt wholesale with multiples of
    pi/2, which produces circuits of the right depth and, at this step
    count, all with the SAME exact value of -2.0. A regression whose
    targets are all identical has slope zero and returns its intercept
    whatever it is shown -- so CDR became a constant, and that constant
    happened to sit closer to the answer than any real method managed. It
    won the comparison while not reading the data at all.

    The scramble attack caught it at a sensitivity of 0.000, which is
    what that check is for, and it was nearly dismissed as a false
    positive on a method known to be legitimate.

    Snapping a FRACTION of the rotations keeps the depth and the gate
    structure of the target while giving the training targets a spread to
    regress against, which is also what the real construction does.

    One honest caveat about scale: at four qubits the exact value of any
    of these is computable regardless of how Clifford they are, so
    nothing here is constrained by simulability. At a size where CDR
    would actually be needed, the training circuits have to be Clifford
    enough to simulate, and that is a real limit this does not feel.
    """
    import random

    rng = random.Random(seed)
    clifford = (0.0, np.pi / 2, np.pi, -np.pi / 2)
    variants, seen = [], set()
    operator = hamiltonian(n_spins, coupling, field)
    for _ in range(n_variants * 20):
        if len(variants) >= n_variants:
            break
        base = trotter_circuit(n_spins, steps, dt, coupling, field)
        trained = QuantumCircuit(n_spins)
        for instruction in base.data:
            operation = instruction.operation
            if operation.name in ("rz", "rx") and rng.random() < 0.8:
                snapped = operation.copy()
                snapped.params = [rng.choice(clifford)]
                trained.append(snapped, instruction.qubits, instruction.clbits)
            else:
                trained.append(operation, instruction.qubits, instruction.clbits)
        value = exact_expectation(trained, operator)
        if round(value, 6) in seen:
            continue
        seen.add(round(value, 6))
        variants.append((trained, value))
    if len({round(v, 6) for _, v in variants}) < 2:
        raise ValueError(
            "every training circuit has the same exact value; a regression "
            "through them is a constant, not a fit")
    return variants


def system(n_spins: int = 4, steps: int = 2, dt: float = 0.3,
           coupling: float = 1.0, field: float = 0.5):
    """The TFIM as something the nine methods can be run against.

    `unit_scale` is 1.0: this is a spin model, so its errors are in units
    of the coupling and calling them kcal/mol would be borrowing a
    chemistry unit for a system that has none.

    Clifford training circuits come from `near_clifford_variants`, which
    snaps most of the rotations to Clifford angles while keeping the
    depth and structure of the target. Replacing the timestep wholesale
    was tried first and produced a degenerate training set; the docstring
    there records what that cost and what caught it.

    No symmetry is declared. The evolved state has support across the
    whole computational basis, so there is no Z-basis sector to
    post-select on, and saying so is better than inventing a filter.
    """
    from .methods import System

    operator = hamiltonian(n_spins, coupling, field)
    circuit = trotter_circuit(n_spins, steps, dt, coupling, field)
    variants = near_clifford_variants(n_spins, steps, dt, coupling, field)
    return System(
        name=f"TFIM {n_spins} spins, {steps} Trotter steps",
        circuit=circuit,
        observable=operator,
        exact=exact_expectation(circuit, operator),
        unit_scale=1.0,
        clifford_variants=tuple(variants),
        physical_z_strings=None,
    )
