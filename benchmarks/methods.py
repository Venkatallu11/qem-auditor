"""Reference implementations of error mitigation methods, to be audited.

The auditor does not ship the thing it audits -- `qem_auditor` imports
nothing from here, exactly as it imports nothing from `benchmarks.suite`.
These live on the benchmark side because their job is to be judged.

Every method has the same signature and the same access to the device:

    method(circuit, backend, shots, seed) -> energy in Hartree

so none of them can quietly buy accuracy with a bigger budget or a
privilege the others lack. Where a method genuinely needs extra circuits
-- readout calibration, Clifford training data, folded copies -- it pays
for them out of its own shot count where that is meaningful, and the cost
is reported either way.

The list deliberately includes two methods that should NOT survive an
audit: a dressed-up identity that does nothing, and one that peeks at the
exact answer. A benchmark of mitigation methods containing only real
ones tests whether the auditor can rank; including these tests whether it
can REFUSE, which is the harder and more useful property.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector

from qem_auditor.estimation import (expectation, group_terms,
                                    rotate_into_basis)

# ---------------------------------------------------------------------------
# The problem, shared with examples/live_h2_audit.py
# ---------------------------------------------------------------------------
HARTREE_TO_KCAL = 627.5094740631

H2 = SparsePauliOp.from_list([
    ("II", -1.052373245772859),
    ("IZ",  0.39793742484318045),
    ("ZI", -0.39793742484318045),
    ("ZZ", -0.01128010425623538),
    ("XX",  0.18093119978423156),
])
FCI = float(np.linalg.eigvalsh(H2.to_matrix())[0])
THETA = 0.223536983

#: The ground state has zero weight on |00> and |11>: it lives entirely in
#: the one-excitation subspace. Any Z-basis shot outside it is provably an
#: error, which is what symmetry verification exploits.
PHYSICAL_Z_STRINGS = ("01", "10")


def ansatz(theta: float = THETA) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.rx(np.pi / 2, 0)
    qc.h(1)
    qc.cx(0, 1)
    qc.rz(theta, 1)
    qc.cx(0, 1)
    qc.rx(-np.pi / 2, 0)
    qc.h(1)
    return qc


def fold_cx(circuit: QuantumCircuit, factor: int) -> QuantumCircuit:
    if factor % 2 == 0:
        raise ValueError("an even fold factor changes the unitary")
    folded = QuantumCircuit(circuit.num_qubits)
    for instruction in circuit.data:
        for _ in range(factor if instruction.operation.name == "cx" else 1):
            folded.append(instruction.operation, instruction.qubits, instruction.clbits)
    return folded


# ---------------------------------------------------------------------------
# Raw access to the device: counts in the two bases the Hamiltonian needs
# ---------------------------------------------------------------------------
def _normalise(counts: dict) -> dict:
    """Bitstrings indexed by qubit number, little-endian, spaces stripped."""
    return {b.replace(" ", "")[::-1]: n for b, n in counts.items()}


def basis_counts(circuit: QuantumCircuit, backend, shots: int, seed: int,
                 operator=None) -> list:
    """One counts table per measurement setting the operator needs.

    This used to run exactly two circuits, a Z basis and an X basis,
    because that is what H2 and the Ising chain happen to need. It did
    not check: an operator with a term like XYZ had its bases popped
    arbitrarily from a set and was measured in one of them, returning a
    number that was wrong rather than absent, while the docstring said
    silently averaging would be wrong.

    `qem_auditor.estimation` groups the terms into commuting settings and
    rotates each qubit into the basis its term needs, so any Pauli
    observable works and anything unmeasurable is refused by name.
    """
    operator = H2 if operator is None else operator
    settings, _ = group_terms(operator.paulis.to_labels())
    tables = []
    for index, setting in enumerate(settings):
        rotated = rotate_into_basis(circuit, setting)
        result = backend.run(transpile(rotated, backend, optimization_level=0),
                             shots=shots,
                             seed_simulator=seed + index).result()
        tables.append(_normalise(result.get_counts()))
    return tables


def energy_from_counts(tables, operator=None) -> float:
    """<O> from one counts table per measurement setting."""
    operator = H2 if operator is None else operator
    _, assignment = group_terms(operator.paulis.to_labels())
    return expectation(tables, operator, assignment)


def error_kcal(energy: float) -> float:
    # float() deliberately: several methods return numpy scalars, and a
    # numpy value flowing into a record is how a measured False became
    # "not recorded" once already in this project.
    return float(abs(float(energy) - FCI) * HARTREE_TO_KCAL)


# ---------------------------------------------------------------------------
# What a method is run against
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class System:
    """A circuit, an observable, and the answer nobody gets to see.

    Introduced when the findings needed a second physical system. Every
    method below was written against H2 and read `ansatz()` and the H2
    Hamiltonian directly, which made "does this generalise?" unanswerable
    without duplicating all nine. They now take the system from the
    sampler, so the same code runs on anything.

    `clifford_variants` is how CDR gets training circuits whose exact
    answer is classically computable. It is system-specific -- for a
    variational ansatz it is the angles at which the circuit becomes
    Clifford -- so the system supplies them rather than the method
    guessing.

    `physical_z_strings` declares a symmetry checkable in the Z basis, or
    None when the state has none. None is the honest default: most states
    do not, and a post-selection filter without a symmetry behind it is
    just discarding data that disagrees.
    """

    name: str
    circuit: Any
    observable: Any
    exact: float
    unit_scale: float = 1.0
    clifford_variants: tuple = ()
    physical_z_strings: Optional[tuple] = None

    def error(self, value: float) -> float:
        """Distance from the truth, in the system's own reported unit."""
        return float(abs(float(value) - self.exact) * self.unit_scale)


#: Angles at which a single-qubit rotation becomes Clifford.
CLIFFORD_ANGLES_GENERIC = (0.0, np.pi / 2, np.pi, -np.pi / 2)


def near_clifford_training(circuit, observable, n_variants: int = 5,
                           snap_fraction: float = 0.8, seed: int = 7) -> tuple:
    """Training circuits for CDR, from any parameterised circuit.

    Snaps a fraction of the rotation angles to Clifford values, keeping
    the circuit's depth and two-qubit structure, and computes each
    variant's exact value. That is the standard construction and it needs
    to know nothing about the physics -- which is what lets CDR run on a
    circuit this package has never seen instead of refusing it.

    Variants with duplicate exact values are dropped. A training set
    whose targets are all equal gives a regression of slope zero, which
    returns its intercept whatever it is shown -- a constant wearing a
    method's name. That happened once here, and the constant beat every
    real method until the scramble attack caught it.

    The honest limit: the exact value is computed by statevector, so this
    generates training data only where the circuit is simulable. At a
    size where CDR is actually needed, the variants have to be Clifford
    enough to simulate by other means, and that constraint is real and
    not felt here.
    """
    import random

    from qiskit.quantum_info import Statevector

    rng = random.Random(seed)
    variants, seen = [], set()
    for attempt in range(n_variants * 40):
        if len(variants) >= n_variants:
            break
        # Vary how much is snapped across variants. A fixed fraction
        # produces near-identical circuits on a circuit with few
        # rotations, and their exact values collide -- which is how a
        # training set of five became a training set of one here.
        fraction = snap_fraction * (0.4 + 0.6 * rng.random())
        trained = circuit.copy_empty_like()
        for instruction in circuit.data:
            operation = instruction.operation
            if (operation.name in ("rx", "ry", "rz", "p", "u1")
                    and len(operation.params) == 1
                    and rng.random() < fraction):
                snapped = operation.copy()
                snapped.params = [rng.choice(CLIFFORD_ANGLES_GENERIC)]
                trained.append(snapped, instruction.qubits, instruction.clbits)
            else:
                trained.append(operation, instruction.qubits, instruction.clbits)
        value = float(Statevector(trained).expectation_value(observable).real)
        # Rounded coarsely on purpose: two targets differing in the ninth
        # decimal are one point as far as a regression is concerned, and
        # counting them as two is how a degenerate set passes for a fit.
        if round(value, 4) in seen:
            continue
        seen.add(round(value, 4))
        variants.append((trained, value))
    return tuple(variants)


def system_from_circuit(circuit, observable, name: str = "",
                        unit_scale: float = 1.0,
                        physical_z_strings=None) -> System:
    """A System from somebody else's circuit and observable.

    Computes the exact value and generates CDR training data, so a
    circuit this package has never seen can go through every method
    rather than through the four that happen not to need them.

    Requires the circuit to be simulable, because measuring a method's
    ERROR needs the answer. Applying a method to get a number needs no
    such thing, and that distinction is the difference between
    benchmarking mitigation and using it.
    """
    from qiskit.quantum_info import Statevector

    return System(
        name=name or getattr(circuit, "name", "") or "user circuit",
        circuit=circuit,
        observable=observable,
        exact=float(Statevector(circuit).expectation_value(observable).real),
        unit_scale=unit_scale,
        clifford_variants=near_clifford_training(circuit, observable),
        physical_z_strings=physical_z_strings,
    )


def h2_system() -> System:
    return System(
        name="H2/STO-3G", circuit=ansatz(), observable=H2, exact=FCI,
        unit_scale=HARTREE_TO_KCAL,
        clifford_variants=tuple((ansatz(a), _exact_energy(a))
                                for a in CLIFFORD_ANGLES),
        physical_z_strings=PHYSICAL_Z_STRINGS)


# ---------------------------------------------------------------------------
# The sampler a method is handed
# ---------------------------------------------------------------------------
class Sampler:
    """What a method is allowed to touch: circuits in, counts out.

    Methods never hold the backend, the exact answer, or each other's
    results. That is what makes the shootout below a comparison rather
    than a collection of anecdotes -- and what makes it possible to hand
    any method a SCRAMBLED sampler and see whether its answer was ever a
    function of the data.
    """

    def __init__(self, backend, shots: int, seed: int,
                 system: "System" = None) -> None:
        self.backend = backend
        self.shots = shots
        self.seed = seed
        self.system = system if system is not None else h2_system()
        self.circuits_run = 0
        self.shots_used = 0

    def measure_in(self, circuit, setting, shots=None, seed_offset=0,
                   calibration: bool = False) -> dict:
        """One counts table in an explicitly named basis.

        Readout calibration needs the computational basis whatever the
        observable happens to need, so it cannot go through the normal
        path -- which returns the settings the OBSERVABLE requires and
        would have handed the calibration whichever basis came first.
        """
        shots = shots or self.shots
        rotated = rotate_into_basis(circuit, setting)
        self.circuits_run += 1
        self.shots_used += shots
        result = self.backend.run(
            transpile(rotated, self.backend, optimization_level=0),
            shots=shots, seed_simulator=self.seed + seed_offset).result()
        table = _normalise(result.get_counts())
        if calibration or not getattr(self, "_scrambles", False):
            return table
        rng = np.random.default_rng(self.seed + seed_offset)
        values = list(table.values())
        rng.shuffle(values)
        return dict(zip(list(table), values))

    def energy(self, tables) -> float:
        return energy_from_counts(tables, self.system.observable)

    @property
    def circuit(self):
        return self.system.circuit

    def __call__(self, circuit: QuantumCircuit, shots: int = None,
                 seed_offset: int = 0, calibration: bool = False) -> list:
        """`calibration=True` marks a measurement taken to CALIBRATE the
        method rather than to estimate the answer.

        The distinction is the method's to declare and nothing else can
        infer it: a folded copy of the experiment is still the
        experiment, while a near-Clifford training circuit of identical
        shape is not. Guessing by comparing circuit objects was tried and
        classified every one of ZNE's folded measurements as calibration,
        which made ZNE look perfectly data-independent.
        """
        shots = shots or self.shots
        tables = basis_counts(circuit, self.backend, shots,
                              self.seed + seed_offset,
                              self.system.observable)
        # Charged per circuit actually submitted, which is one per
        # measurement setting -- an operator needing four settings costs
        # four, and pretending otherwise would understate every method's
        # cost on anything but H2.
        self.circuits_run += len(tables)
        self.shots_used += len(tables) * shots
        return tables


class ScrambledSampler(Sampler):
    """Every returned table has its outcome labels randomly permuted.

    Note the "every". For a method that calibrates -- CDR against
    near-Clifford circuits, REM against preparation circuits -- this
    scrambles the calibration data as well as the experiment's, and the
    method re-fits to the garbage. Measured on a hardware-efficient
    ansatz, CDR's fitted slope flipped from +1.24 to -0.32 while its
    target also flipped sign, and the two distortions partly cancelled:
    the method scored 0.390 and looked like it was not reading the data,
    when what had happened is that it read scrambled data twice and
    compensated.

    `TargetScrambledSampler` is the variant that isolates the question
    actually being asked, and is the one to use on any method with a
    calibration step.

    The physics is destroyed and the shot statistics are untouched, so a
    method whose answer survives this was never reading the data. This is
    `T_label` from the failure grammar, pointed at a mitigation method
    instead of at a fitting routine.
    """

    def __call__(self, circuit, shots=None, seed_offset=0,
                 calibration: bool = False) -> list:
        tables = super().__call__(circuit, shots, seed_offset, calibration)
        rng = np.random.default_rng(self.seed + seed_offset)
        scrambled = []
        for table in tables:
            labels = list(table)
            values = list(table.values())
            rng.shuffle(values)
            scrambled.append(dict(zip(labels, values)))
        return scrambled


# ---------------------------------------------------------------------------
# The methods
# ---------------------------------------------------------------------------
def unmitigated(sampler: Sampler) -> float:
    """The baseline every other method has to beat."""
    return sampler.energy(sampler(sampler.circuit))


#: Full readout calibration needs one circuit per basis state, so its
#: cost is 2**n. Beyond this it is refused rather than run: an honest
#: implementation of a method nobody would use at that size is worse than
#: saying so, and the tensored and M3 approximations that real work uses
#: at scale are a different method with different assumptions, not this
#: one made faster.
MAX_REM_QUBITS = 6


def _confusion_matrix(sampler: Sampler, shots: int) -> np.ndarray:
    """Measured readout confusion, from one preparation circuit per basis
    state.

    M[measured, prepared]. Paid for out of the method's own shot budget,
    which is why REM is not free -- and the number of those circuits is
    2**n, which is why it does not stay affordable.

    This was fixed at 4x4 until a second physical system arrived and
    crashed on it. Being hardcoded to the width of the only system it had
    ever been run on is exactly the defect a second system exists to find.
    """
    n_qubits = sampler.circuit.num_qubits
    if n_qubits > MAX_REM_QUBITS:
        raise ValueError(
            f"full readout calibration needs 2**{n_qubits} = "
            f"{2 ** n_qubits} circuits. Past {MAX_REM_QUBITS} qubits that is "
            "not a method anyone runs; use a tensored or M3 estimator, which "
            "is a different method with different assumptions rather than "
            "this one made cheaper.")
    size = 2 ** n_qubits
    matrix = np.zeros((size, size))
    for column in range(size):
        bits = format(column, f"0{n_qubits}b")
        prep = QuantumCircuit(n_qubits)
        for qubit, bit in enumerate(bits[::-1]):
            if bit == "1":
                prep.x(qubit)
        table = sampler.measure_in(prep, ("Z",) * n_qubits, shots=shots,
                                   seed_offset=100 + column, calibration=True)
        total = sum(table.values())
        for measured, n in table.items():
            matrix[int(measured[::-1], 2), column] += n / total
    return matrix


def _apply_readout_correction(table: dict, inverse: np.ndarray) -> dict:
    total = sum(table.values())
    size = inverse.shape[0]
    observed = np.zeros(size)
    for bits, n in table.items():
        observed[int(bits[::-1], 2)] = n / total
    corrected = inverse @ observed
    # Unconstrained inversion can leave small negative probabilities. They
    # are clipped and renormalised rather than passed on, and this is
    # where REM starts to be an approximation rather than an identity.
    corrected = np.clip(corrected, 0.0, None)
    corrected /= corrected.sum()
    width = size.bit_length() - 1
    return {format(i, f"0{width}b")[::-1]: corrected[i] * total
            for i in range(size)}


def readout_mitigation(sampler: Sampler, calibration_shots: int = 8_000) -> float:
    """REM: measure the readout confusion matrix and invert it.

    Included because the ZNE audit found that readout error is what
    defeats zero-noise extrapolation: it does not scale with the number
    of gates, so no extrapolation in the fold factor can reach it. This
    is the method that attacks it directly.
    """
    inverse = np.linalg.pinv(_confusion_matrix(sampler, calibration_shots))
    tables = sampler(sampler.circuit)
    return sampler.energy(
        [_apply_readout_correction(t, inverse) for t in tables])


def zne(sampler: Sampler, folds=(1, 3, 5), order: int = 1) -> float:
    """Zero-noise extrapolation by unitary folding."""
    values = [sampler.energy(sampler(fold_cx(sampler.circuit, f), seed_offset=10 * i))
              for i, f in enumerate(folds)]
    return float(np.polyval(np.polyfit(folds, values, order), 0.0))


def rem_then_zne(sampler: Sampler, folds=(1, 3, 5), order: int = 1,
                 calibration_shots: int = 8_000) -> float:
    """Readout mitigation first, then extrapolate what is left.

    The composition the ZNE finding implies: remove the error folding
    cannot reach, then extrapolate the error it can.
    """
    inverse = np.linalg.pinv(_confusion_matrix(sampler, calibration_shots))
    values = []
    for i, factor in enumerate(folds):
        tables = sampler(fold_cx(sampler.circuit, factor), seed_offset=10 * i)
        values.append(sampler.energy(
            [_apply_readout_correction(t, inverse) for t in tables]))
    return float(np.polyval(np.polyfit(folds, values, order), 0.0))


def symmetry_verification(sampler: Sampler) -> float:
    """Post-select on the subspace the state is known to live in.

    The H2 ground state has zero weight on |00> and |11>, so a Z-basis
    shot outside {01, 10} is provably an error and can be discarded.

    The honest limitation, stated rather than hidden: this symmetry is
    only available in the Z basis. The XX term is measured after a basis
    rotation under which the one-excitation subspace is not preserved, so
    those shots are kept as they are. A method that mitigates three of a
    Hamiltonian's five terms should not be described as mitigating the
    Hamiltonian.
    """
    physical = sampler.system.physical_z_strings
    if not physical:
        raise ValueError(
            f"{sampler.system.name} declares no symmetry checkable in the Z "
            "basis. Post-selection without a symmetry behind it is discarding "
            "the data that happens to disagree.")
    settings, _ = group_terms(sampler.system.observable.paulis.to_labels())
    tables = sampler(sampler.circuit)
    # Only settings measured entirely in Z carry the symmetry. A setting
    # that rotated any qubit into X or Y is no longer in the basis the
    # symmetry is stated in, and filtering it would discard shots for a
    # reason that does not apply to them.
    kept = []
    filtered_any = False
    for setting, table in zip(settings, tables):
        if all(basis in ("I", "Z") for basis in setting):
            survivors = {b: n for b, n in table.items() if b in physical}
            if not survivors:
                raise ValueError(
                    "post-selection discarded every shot in a setting the "
                    "symmetry applies to")
            kept.append(survivors)
            filtered_any = True
        else:
            kept.append(table)
    if not filtered_any:
        raise ValueError(
            f"{sampler.system.name}: no measurement setting is entirely in the "
            "Z basis, so the declared symmetry is never visible and this method "
            "would discard nothing while claiming to mitigate")
    return sampler.energy(kept)


def _exact_energy(theta: float) -> float:
    return float(Statevector(ansatz(theta)).expectation_value(H2).real)


#: Angles at which the Trotter ansatz becomes Clifford, so the exact
#: answer is classically computable at any system size. That is what makes
#: CDR a real method rather than a way of smuggling in the answer.
CLIFFORD_ANGLES = tuple(k * np.pi / 2 for k in (-2, -1, 0, 1, 2))


def cdr(sampler: Sampler, angles=None) -> float:
    """Clifford data regression.

    Runs near-Clifford versions of the circuit, where the exact answer is
    classically computable, learns the map from noisy value to exact
    value, and applies it to the real circuit's noisy value.

    Unlike ZNE this makes no assumption about how the error scales with
    gate count, so readout error is inside what it learns rather than
    outside what it can reach. Unlike PEC it needs no noise model. What
    it does assume is that the map learned at Clifford angles transfers
    to the angle actually used -- an assumption about the CIRCUIT, traded
    for the assumption about the NOISE.
    """
    variants = angles if angles is not None else sampler.system.clifford_variants
    if len(variants) < 2:
        raise ValueError(
            f"{sampler.system.name} supplies {len(variants)} training circuits; "
            "a regression needs two points to be a line rather than a guess "
            "through one.")
    noisy, exact = [], []
    for i, (circuit, truth) in enumerate(variants):
        noisy.append(sampler.energy(
            sampler(circuit, seed_offset=200 + 10 * i, calibration=True)))
        exact.append(truth)
    slope, intercept = np.polyfit(noisy, exact, 1)
    return float(slope * sampler.energy(sampler(sampler.circuit)) + intercept)


def pec_model_inversion(sampler: Sampler, assumed_gate_error: float = 0.02,
                        n_two_qubit_gates: int = 2) -> float:
    """Noise-model inversion, the cheap end of the PEC family.

    Full probabilistic error cancellation samples a quasi-probability
    decomposition of the inverse noise channel. This implements the same
    idea at the expectation-value level under a global depolarizing
    assumption: if the circuit's noise depolarizes with fidelity F, then

        <O>_noisy = F <O>_ideal + (1 - F) tr(O)/d

    and <O>_ideal follows by rearranging. It is included precisely
    BECAUSE its correctness rests entirely on the assumed noise being the
    real noise. A method that is confidently wrong when its model is
    wrong is the case the auditor's CALIBRATION_MISMATCH diagnosis exists
    for, and it is not much use having that diagnosis with nothing in the
    suite that triggers it honestly.
    """
    fidelity = (1.0 - assumed_gate_error) ** n_two_qubit_gates
    operator = sampler.system.observable
    labels = list(operator.paulis.to_labels())
    identity = "I" * operator.num_qubits
    # tr(O)/d is the identity term's coefficient, and zero when the
    # operator has no identity term -- true of the Ising Hamiltonian, and
    # silently H2's first coefficient before this was generalised.
    identity_coefficient = (float(operator.coeffs[labels.index(identity)].real)
                            if identity in labels else 0.0)
    noisy = sampler.energy(sampler(sampler.circuit))
    return (noisy - (1.0 - fidelity) * identity_coefficient) / fidelity


def dressed_identity(sampler: Sampler, folds=(1, 3, 5)) -> float:
    """A method that runs everything a real one runs and returns the raw
    value anyway.

    Not a strawman: pipelines really do end up here, through an
    extrapolation whose coefficients collapse to (1, 0, 0), a correction
    applied to a copy that is then discarded, or a flag that silently
    turned the correction off. It costs the same as ZNE, produces plots
    that look like ZNE's, and mitigates nothing. Any benchmark of
    mitigation methods should contain one, and the auditor should say so.
    """
    for i, factor in enumerate(folds):
        sampler(fold_cx(sampler.circuit, factor), seed_offset=10 * i)
    return sampler.energy(sampler(sampler.circuit))


def oracle_peek(sampler: Sampler, blend: float = 0.98) -> float:
    """Fraud, on purpose: it looks at the answer.

    Runs the same circuits as everyone else, then returns something very
    close to the exact energy. On accuracy it wins every comparison in
    this file by orders of magnitude, which is the point -- a leaderboard
    ranked on error alone crowns it.

    What catches it is that its answer is barely a function of the data.
    Hand it a ScrambledSampler and it returns the same number. That is
    the `T_label` attack from the failure grammar, and it is the only
    thing here that separates this from a genuinely excellent method.
    """
    noisy = sampler.energy(sampler(sampler.circuit))
    return blend * sampler.system.exact + (1.0 - blend) * noisy


METHODS = {
    "unmitigated": unmitigated,
    "REM (readout)": readout_mitigation,
    "ZNE (fold 1,3,5)": zne,
    "REM + ZNE": rem_then_zne,
    "symmetry verification": symmetry_verification,
    "CDR (Clifford regression)": cdr,
    "PEC (model inversion)": pec_model_inversion,
    "dressed identity": dressed_identity,
    "oracle peek (fraud)": oracle_peek,
}


class TargetScrambledSampler(Sampler):
    """Scrambles the experiment's measurement, leaving calibration alone.

    The question the attack asks is whether the reported answer depends
    on the data it reports on. A calibration measurement is a separate
    and legitimate input, so scrambling it too asks a different and less
    useful question -- and lets a calibrated method absorb the distortion
    in its fit, scoring as though it were not reading anything.

    "The experiment's measurement" is the one taken on the system's own
    circuit. Everything else -- folded copies, preparation circuits,
    near-Clifford training -- passes through untouched.
    """

    def __call__(self, circuit, shots=None, seed_offset=0,
                 calibration: bool = False) -> list:
        tables = Sampler.__call__(self, circuit, shots, seed_offset, calibration)
        if calibration:
            return tables
        rng = np.random.default_rng(self.seed + seed_offset)
        scrambled = []
        for table in tables:
            labels = list(table)
            values = list(table.values())
            rng.shuffle(values)
            scrambled.append(dict(zip(labels, values)))
        return scrambled


def scramble_shift(method, backend, shots: int, seeds,
                   sampler_class=None, system=None) -> float:
    """How far a method's answer moves when the outcome labels are
    scrambled, in kcal/mol."""
    sampler_class = sampler_class or TargetScrambledSampler
    honest = statistics.median(
        [method(Sampler(backend, shots, s, system)) for s in seeds])
    scrambled = statistics.median(
        [method(sampler_class(backend, shots, s, system)) for s in seeds])
    # Scaled by the system's own unit, not H2's. Defaulting to kcal/mol
    # made every ratio right and every absolute figure wrong on anything
    # that is not a molecule.
    scale = (system or h2_system()).unit_scale
    return abs(scrambled - honest) * scale


def data_sensitivity(method, backend, shots: int, seeds,
                     reference: float = None, system=None) -> float:
    """How much of the method's answer is a function of the data?

    Scrambling the outcome labels destroys the physics and leaves the shot
    statistics alone, so every honest method's answer must move. The
    question is how far, and the only meaningful scale is how far the RAW
    estimate moved on the same scrambled data.

    Returns the ratio. An honest method sits near 1: it is as disturbed as
    the data it reads. A method that peeks at the answer sits near 0
    however good its accuracy looks, because its output was never really a
    function of its input.

    An absolute threshold cannot do this job -- the shift is hundreds of
    kcal/mol for everything -- which is why this is a ratio.

    `reference` is the unmitigated method's shift under the same backend
    and seeds. Pass it in when scoring several methods against one
    backend; recomputing it per method is the same measurement repeated.
    """
    if reference is None:
        reference = scramble_shift(unmitigated, backend, shots, seeds,
                                   system=system)
    if reference <= 0:
        raise ValueError("the baseline did not move under scrambling; "
                         "there is no scale to measure against")
    return scramble_shift(method, backend, shots, seeds,
                          system=system) / reference


def is_deterministic(method, backend, shots: int, seed: int) -> bool:
    """Does the identical computation on identical inputs give an
    identical result?

    Cheap, and one of the few controls the auditor can settle outright.
    A method that fails this has an unseeded source of randomness in it,
    and every other number it reports inherits that.
    """
    first = method(Sampler(backend, shots, seed))
    second = method(Sampler(backend, shots, seed))
    return first == second


#: Methods that fit something to data, and therefore owe a held-out
#: check. The rest have nothing fitted to hold data out of -- which is a
#: different situation from having skipped the check, and the shootout
#: records it as such rather than leaving the control unrun.
FITTING_METHODS = ("ZNE (fold 1,3,5)", "REM + ZNE", "CDR (Clifford regression)")


def heldout_ok(name: str, sampler_factory, tolerance_kcal: float,
               calibration_shots: int = 8_000) -> bool:
    """Predict a point that was withheld from the fit, in the direction
    production uses.

    For the extrapolators that means holding out the LOWEST fold and
    fitting only the ones above it, so the prediction is an extrapolation
    like the one production makes. For CDR it means holding out one
    Clifford angle and predicting it from the others.

    The check runs the method's OWN pipeline. Validating REM + ZNE with
    plain ZNE would test a method nobody proposed and condemn one for a
    failure that is not its own -- which is what this did on its first
    version, caught by REM + ZNE landing INVALID while being the most
    accurate honest method in the shootout.
    """
    sampler = sampler_factory()
    if name == "CDR (Clifford regression)":
        variants = list(sampler.system.clifford_variants)
        held, fitted = variants[0], variants[1:]
        noisy = [sampler.energy(sampler(c, seed_offset=300 + 10 * i))
                 for i, (c, _) in enumerate(fitted)]
        slope, intercept = np.polyfit(noisy, [t for _, t in fitted], 1)
        predicted = slope * sampler.energy(
            sampler(held[0], seed_offset=400)) + intercept
        error = abs(predicted - held[1]) * sampler.system.unit_scale
    else:
        # The same readout correction the method itself applies, or none
        # if it applies none.
        if name == "REM + ZNE":
            inverse = np.linalg.pinv(_confusion_matrix(sampler, calibration_shots))

            def process(tables):
                return [_apply_readout_correction(t, inverse)
                        for t in tables]
        else:
            def process(tables):
                return tables

        folds = [1, 3, 5]
        measured = {f: sampler.energy(process(
            sampler(fold_cx(sampler.circuit, f), seed_offset=500 + 10 * i)))
            for i, f in enumerate(folds)}
        fitted = folds[1:]
        predicted = np.polyval(
            np.polyfit(fitted, [measured[f] for f in fitted], 1), folds[0])
        error = abs(predicted - measured[folds[0]]) * sampler.system.unit_scale
    return bool(error <= tolerance_kcal)
