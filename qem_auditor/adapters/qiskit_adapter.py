"""Executes controls against real Qiskit circuits.

Three of the auditor's controls are mechanizable, and they happen to be
three of the cheapest and most decisive in the whole failure library:

- **unitary equivalence** -- does the circuit that would be SUBMITTED
  implement the intended unitary? Pure linear algebra, no simulation, no
  shots. This is the check that would have caught the abstract-gate
  folding failure before a single job was paid for.

- **ideal control** -- run the claimant's own mitigation pipeline against a
  noiseless model. If mitigation makes a zero-noise case worse, the method
  is broken independent of hardware. This is the check that caught the
  513x blowup, and the shot noise is the point: the blowup came from
  statistical noise alone, on a model with nothing to correct.

- **determinism** -- run the identical computation twice and diff. Invisible
  to any single run, and it caught two separate real bugs in the H4
  project (hash-order nondeterminism tipping a nonconvex solver into
  different local optima; hash()-derived bootstrap seeds).

The remaining controls -- target leakage, adversarial design, free-parameter
floors -- are procedural or domain-specific and still depend on honest
reporting. That is stated plainly by `independent_verification_gate`
rather than papered over.

Requires `qiskit`. Import is local to this module, so the core auditor
keeps its zero-dependency property.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from .base import ControlMeasurement, MeasurementError

# An expectation oracle: given a circuit and an observable, return <O>.
# The claimant's mitigation pipeline is handed one of these and never
# knows whether it is talking to a noiseless model or a noisy one -- which
# is exactly what makes the ideal control a fair test of the pipeline as
# it actually runs.
ExpectationFn = Callable[[Any, Any], float]
Mitigator = Callable[[ExpectationFn], float]


def _require_qiskit():
    try:
        from qiskit.quantum_info import Operator, SparsePauliOp, Statevector  # noqa: F401
    except ImportError as e:  # pragma: no cover - environment dependent
        raise MeasurementError(
            "the Qiskit adapter needs qiskit installed: pip install qiskit"
        ) from e
    from qiskit.quantum_info import Operator, SparsePauliOp, Statevector

    return Operator, SparsePauliOp, Statevector


class QiskitAdapter:
    """Runs the mechanizable controls against Qiskit circuits."""

    name = "qiskit"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        _require_qiskit()

    # -- unitary equivalence -------------------------------------------

    def measure_unitary_equivalence(self, intended, submitted) -> ControlMeasurement:
        """Does the submitted circuit implement the intended unitary?

        Compared up to global phase, which is unobservable. Anything else
        -- a cancelled fold pair, a transpiler pass that collapsed a gate
        sequence, a rewritten basis that is not actually equivalent -- shows
        up here.
        """
        Operator, _, _ = _require_qiskit()
        if intended.num_qubits != submitted.num_qubits:
            return ControlMeasurement(
                "unitary_equivalence", False,
                f"qubit count differs: intended {intended.num_qubits}, "
                f"submitted {submitted.num_qubits}",
                {"intended_qubits": intended.num_qubits,
                 "submitted_qubits": submitted.num_qubits})
        try:
            a, b = Operator(intended), Operator(submitted)
        except Exception as e:
            raise MeasurementError(
                f"could not build the unitary for comparison: {e}. Circuits with "
                f"measurements or resets are not unitary; compare the pre-measurement "
                f"circuits instead."
            ) from e

        equivalent = a.equiv(b)
        intended_ops = sum(intended.count_ops().values())
        submitted_ops = sum(submitted.count_ops().values())
        evidence = {
            "equivalent_up_to_global_phase": bool(equivalent),
            "intended_gate_count": intended_ops,
            "submitted_gate_count": submitted_ops,
            "intended_ops": dict(intended.count_ops()),
            "submitted_ops": dict(submitted.count_ops()),
        }
        if equivalent:
            # Equivalent unitaries with very different gate counts is the
            # fold-cancellation signature: the noise-amplified arm was
            # optimized back down to the unamplified one. Same maths, and
            # emphatically not the intended experiment.
            note = ""
            if submitted_ops < intended_ops:
                note = (f" -- but the submitted circuit has {intended_ops - submitted_ops} "
                        f"FEWER gates ({intended_ops} -> {submitted_ops}). If those gates "
                        f"were inserted deliberately to amplify noise, they did not "
                        f"survive to execution")
            return ControlMeasurement(
                "unitary_equivalence", True,
                f"submitted circuit implements the intended unitary "
                f"(up to global phase){note}", evidence)
        return ControlMeasurement(
            "unitary_equivalence", False,
            f"the submitted circuit does NOT implement the intended unitary "
            f"({intended_ops} gates intended, {submitted_ops} submitted)", evidence)

    def measure_fold_survival(self, base, submitted) -> ControlMeasurement:
        """Did deliberately-inserted noise-amplifying gates survive to execution?

        Unitary equivalence alone cannot answer this, and assuming it can
        is exactly how the abstract-gate folding failure happened. A ZNE
        fold pair G.G^-1 is *supposed* to leave the unitary unchanged --
        that is the whole design. So an equivalence check passes happily
        while the transpiler quietly removes the pairs, leaving a
        "noise-amplified" arm that has the same noise as the unamplified
        one and an extrapolation fitting a slope through a variable that
        never varied.

        The correct check is both halves at once: the unitary must be
        preserved AND the gate count must actually have gone up.
        """
        Operator, _, _ = _require_qiskit()
        base_ops = sum(base.count_ops().values())
        submitted_ops = sum(submitted.count_ops().values())
        try:
            equivalent = Operator(base).equiv(Operator(submitted))
        except Exception as e:
            raise MeasurementError(f"could not build the unitary for comparison: {e}") from e

        evidence = {
            "equivalent_up_to_global_phase": bool(equivalent),
            "base_gate_count": base_ops,
            "submitted_gate_count": submitted_ops,
            "amplification_ratio": submitted_ops / base_ops if base_ops else None,
        }
        if not equivalent:
            return ControlMeasurement(
                "unitary_equivalence", False,
                f"the folded circuit does not preserve the base unitary -- folding "
                f"changed the computation, not just the noise",
                evidence)
        if submitted_ops <= base_ops:
            return ControlMeasurement(
                "unitary_equivalence", False,
                f"the noise-amplifying gates did NOT survive to execution: base has "
                f"{base_ops} gates, submitted has {submitted_ops}. The amplified arm "
                f"carries the same noise as the unamplified one, so any extrapolation "
                f"over fold factor is fitting a variable that never varied",
                evidence)
        return ControlMeasurement(
            "unitary_equivalence", True,
            f"folding preserved the unitary and survived transpilation "
            f"({base_ops} -> {submitted_ops} gates, "
            f"{submitted_ops / base_ops:.2f}x amplification)",
            evidence)

    # -- ideal control -------------------------------------------------

    def measure_ideal_control(self, circuit, observable, mitigator: Mitigator,
                              shots: int = 100_000,
                              degradation_factor: float = 10.0) -> ControlMeasurement:
        """Run the claimant's mitigation against a noiseless model.

        The pipeline gets an expectation oracle backed by an exact
        statevector plus honest shot noise, and nothing else changes.

        On `degradation_factor`: SOME amplification here is expected and
        benign. Any zero-noise extrapolator amplifies statistical noise by
        roughly the norm of its coefficients -- Richardson's (3, -3, 1)
        sits near 4.4x -- and on a noiseless model there is no physical
        noise for that cost to buy anything back. So this check is not
        looking for amplification; it is looking for PATHOLOGICAL
        amplification, the ill-conditioning that turned a 0.0652 kcal/mol
        error into 33.48 (513x). The default of 10x sits above the
        well-conditioned regime and orders of magnitude below the failure
        it exists to catch. The measured ratio is reported either way, so a
        caller who knows their estimator's coefficient norm can compare
        against it directly.
        """
        _, SparsePauliOp, Statevector = _require_qiskit()
        exact = self._exact_expectation(circuit, observable)
        oracle, calls = self._noiseless_oracle(shots)
        try:
            mitigated = float(mitigator(oracle))
        except Exception as e:
            raise MeasurementError(
                f"the mitigation pipeline raised on noiseless input: {e}. That is "
                f"itself worth knowing -- a pipeline that cannot run on a clean model "
                f"has not been tested on one."
            ) from e

        raw = self._sampled_expectation(circuit, observable, shots, self.seed)
        raw_error = abs(raw - exact)
        mitigated_error = abs(mitigated - exact)
        evidence = {
            "exact": exact,
            "raw": raw,
            "mitigated": mitigated,
            "raw_error": raw_error,
            "mitigated_error": mitigated_error,
            "shots": shots,
            "oracle_calls": calls["n"],
        }
        if raw_error > 0:
            evidence["amplification"] = mitigated_error / raw_error

        if mitigated_error > degradation_factor * max(raw_error, 1e-12):
            ratio = mitigated_error / max(raw_error, 1e-12)
            return ControlMeasurement(
                "ideal_control", False,
                f"on a noiseless model, mitigation amplified the error {ratio:.1f}x "
                f"({raw_error:.6g} -> {mitigated_error:.6g}) -- with zero physical noise "
                f"to correct, this is the estimator amplifying shot noise",
                evidence)
        ratio = mitigated_error / max(raw_error, 1e-12)
        return ControlMeasurement(
            "ideal_control", True,
            f"on a noiseless model, mitigation is not pathologically conditioned "
            f"(raw error {raw_error:.6g}, mitigated {mitigated_error:.6g}, "
            f"{ratio:.1f}x -- within the {degradation_factor:.0f}x bar; compare against "
            f"your estimator's own coefficient norm)", evidence)

    # -- determinism ---------------------------------------------------

    def measure_determinism(self, computation: Callable[[], float],
                            runs: int = 3, tolerance: float = 0.0) -> ControlMeasurement:
        """Run the identical computation N times and diff the results.

        Default tolerance is exact equality. A pipeline that is supposed to
        be deterministic given fixed seeds either is or is not; a
        "close enough" default here would hide precisely the ordering and
        threading bugs this check exists to find.
        """
        if runs < 2:
            raise MeasurementError("determinism needs at least 2 runs to compare")
        try:
            values = [float(computation()) for _ in range(runs)]
        except Exception as e:
            raise MeasurementError(f"the computation raised during a repeat run: {e}") from e
        spread = max(values) - min(values)
        evidence = {"values": values, "spread": spread, "runs": runs}
        if spread > tolerance:
            return ControlMeasurement(
                "determinism_check", False,
                f"{runs} identical runs produced different results (spread {spread:.6g}; "
                f"values {values}) -- the reported number is one draw from an unstated "
                f"distribution",
                evidence)
        return ControlMeasurement(
            "determinism_check", True,
            f"{runs} identical runs produced identical results ({values[0]:.6g})",
            evidence)

    # -- internals -----------------------------------------------------

    def _exact_expectation(self, circuit, observable) -> float:
        _, _, Statevector = _require_qiskit()
        state = Statevector.from_instruction(circuit)
        return float(state.expectation_value(observable).real)

    def _sampled_expectation(self, circuit, observable, shots: int, seed: int) -> float:
        """Exact expectation plus honest shot noise.

        For an observable with eigenvalues +-1, N shots give a binomial
        estimate: p(+1) = (1 + <O>)/2. Sampling that directly is exact for
        a single Pauli term and avoids depending on any particular
        primitives API. Weighted sums of Paulis are sampled term by term,
        which slightly overstates the variance (it ignores the covariance
        between commuting terms measured in one basis) -- stated here
        rather than left for a reader to discover.
        """
        import random

        rng = random.Random(seed)
        _, SparsePauliOp, _ = _require_qiskit()
        obs = SparsePauliOp(observable) if not hasattr(observable, "paulis") else observable
        total = 0.0
        for pauli, coeff in zip(obs.paulis, obs.coeffs):
            exact = self._exact_expectation(circuit, SparsePauliOp(pauli))
            p_plus = min(1.0, max(0.0, (1.0 + exact) / 2.0))
            hits = sum(1 for _ in range(shots) if rng.random() < p_plus) if shots < 20_000 \
                else _fast_binomial(rng, shots, p_plus)
            estimate = 2.0 * hits / shots - 1.0
            total += float(coeff.real) * estimate
        return total

    def _noiseless_oracle(self, shots: int):
        """An expectation oracle over an exact, noiseless model.

        Each call gets its own seed so repeated calls carry independent
        shot noise, as separate real submissions would.
        """
        calls = {"n": 0}

        def oracle(circuit, observable) -> float:
            calls["n"] += 1
            return self._sampled_expectation(circuit, observable, shots,
                                             self.seed + calls["n"])

        return oracle, calls


def _fast_binomial(rng, n: int, p: float) -> int:
    """Normal approximation to Binomial(n, p), for shot counts where
    looping would dominate the runtime. Valid well inside the regime these
    checks use (n >= 20,000)."""
    import math

    mean = n * p
    sd = math.sqrt(max(n * p * (1 - p), 0.0))
    return int(min(n, max(0, round(rng.gauss(mean, sd)))))
