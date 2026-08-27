"""Where expectation values come from.

The adapter's checks do not care whether a number came from an exact
statevector, a density-matrix simulation under a noise model, or real
hardware. They care that it came from somewhere stated. So the source is
the pluggable piece, and adding IBM or Quantinuum is a new source rather
than a new adapter.

**One invariant governs everything here**: the ideal control must run on a
NOISELESS model. That is its entire content -- if a noiseless model does
not recover a sane result, the method is broken before hardware noise
enters the picture. An adapter configured with a noisy source must not
quietly run the ideal control through it, so `ExpectationSource` exposes
`noiseless_twin()` and the ideal control always uses that.

**And one trap, found by testing rather than assumed away**: Aer's
`method="automatic"` silently returns a NOISELESS density matrix (purity
1.0) even when a noise model is attached. A source that trusted the
default would report noiseless numbers as noisy ones -- corrupting every
measurement built on it, invisibly. `AerNoiseSource` pins the method and
then VERIFIES the noise took effect before returning anything.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from .base import MeasurementError


@runtime_checkable
class ExpectationSource(Protocol):
    """Anything that can produce an expectation value for a circuit."""

    name: str

    @property
    def is_noiseless(self) -> bool:
        ...

    def exact(self, circuit: Any, observable: Any) -> float:
        """The expectation with no shot noise, under this source's model."""

    def sampled(self, circuit: Any, observable: Any, shots: int,
                seed: int) -> float:
        """The expectation with honest shot noise."""

    def noiseless_twin(self) -> "ExpectationSource":
        """A source with the same machinery and no noise, for the ideal
        control. A noiseless source returns itself."""


def _require_qiskit():
    try:
        from qiskit.quantum_info import SparsePauliOp, Statevector  # noqa: F401
    except ImportError as e:  # pragma: no cover - environment dependent
        raise MeasurementError(
            "the Qiskit sources need qiskit installed: pip install qiskit") from e
    from qiskit.quantum_info import SparsePauliOp, Statevector

    return SparsePauliOp, Statevector


def _as_observable(observable):
    SparsePauliOp, _ = _require_qiskit()
    return observable if hasattr(observable, "paulis") else SparsePauliOp(observable)


def _sample_pauli(exact: float, shots: int, rng: random.Random) -> float:
    """Shot noise for a single +-1 eigenvalue observable.

    p(+1) = (1 + <P>)/2, so N shots give a binomial estimate. Exact for one
    Pauli term, and cheap enough to run inside a loop.
    """
    p_plus = min(1.0, max(0.0, (1.0 + exact) / 2.0))
    if shots >= 20_000:
        mean = shots * p_plus
        sd = math.sqrt(max(shots * p_plus * (1 - p_plus), 0.0))
        hits = int(min(shots, max(0, round(rng.gauss(mean, sd)))))
    else:
        hits = sum(1 for _ in range(shots) if rng.random() < p_plus)
    return 2.0 * hits / shots - 1.0


@dataclass
class StatevectorSource:
    """Exact statevector, with shot noise sampled on top. No device noise.

    This is what the ideal control needs, and it is the default: a check
    that is supposed to isolate the estimator's own conditioning must not
    have device noise mixed into it.
    """

    name: str = "statevector (noiseless)"

    @property
    def is_noiseless(self) -> bool:
        return True

    def exact(self, circuit: Any, observable: Any) -> float:
        _, Statevector = _require_qiskit()
        state = Statevector.from_instruction(circuit)
        return float(state.expectation_value(_as_observable(observable)).real)

    def sampled(self, circuit: Any, observable: Any, shots: int,
                seed: int) -> float:
        SparsePauliOp, _ = _require_qiskit()
        rng = random.Random(seed)
        obs = _as_observable(observable)
        total = 0.0
        for pauli, coeff in zip(obs.paulis, obs.coeffs):
            term = self.exact(circuit, SparsePauliOp(pauli))
            total += float(coeff.real) * _sample_pauli(term, shots, rng)
        return total

    def noiseless_twin(self) -> "StatevectorSource":
        return self


@dataclass
class AerNoiseSource:
    """Density-matrix simulation under an Aer noise model.

    Lets the auditor ask what the ideal control cannot: does the mitigation
    actually HELP once real noise is present? The ideal control only
    establishes that a method does not break without noise, which is a
    necessary condition and nowhere near a sufficient one.
    """

    noise_model: Any
    basis_gates: tuple[str, ...] = ("u", "cx")
    name: str = "aer (density matrix, noisy)"
    _verified: bool = field(default=False, init=False, repr=False)

    @property
    def is_noiseless(self) -> bool:
        return False

    def _simulator(self):
        try:
            from qiskit_aer import AerSimulator
        except ImportError as e:  # pragma: no cover - environment dependent
            raise MeasurementError(
                "the Aer source needs qiskit-aer: pip install qiskit-aer") from e
        # method pinned deliberately: "automatic" silently returns a
        # NOISELESS density matrix even with a noise model attached.
        return AerSimulator(noise_model=self.noise_model, method="density_matrix")

    def _density_matrix(self, circuit: Any):
        from qiskit import transpile
        from qiskit.quantum_info import DensityMatrix

        prepared = transpile(circuit, basis_gates=list(self.basis_gates),
                             optimization_level=0)
        prepared = prepared.copy()
        prepared.save_density_matrix()
        result = self._simulator().run(prepared, shots=1).result()
        try:
            return DensityMatrix(result.data(0)["density_matrix"])
        except Exception as e:
            raise MeasurementError(
                f"the Aer run returned no density matrix: {e}") from e

    def _verify_noise_applied(self, circuit: Any) -> None:
        """Confirm the noise actually took effect, once, before trusting it.

        Aer will happily return a pure state when the method is wrong, the
        basis does not match what the noise model covers, or the circuit
        transpiles to gates the model says nothing about. Any of those
        gives silently noiseless numbers labelled as noisy. Purity < 1 is
        the cheap check that the model is doing something.
        """
        if self._verified:
            return
        purity = float(self._density_matrix(circuit).purity().real)
        if purity > 1.0 - 1e-9:
            raise MeasurementError(
                f"the noise model had no effect (purity {purity:.9f}, i.e. a pure "
                f"state). Usually the circuit transpiles to gates the model does not "
                f"cover -- check that basis_gates={list(self.basis_gates)} matches "
                f"the model's own basis {sorted(getattr(self.noise_model, 'basis_gates', []))}. "
                f"Returning these as noisy results would be worse than failing.")
        object.__setattr__(self, "_verified", True)

    def exact(self, circuit: Any, observable: Any) -> float:
        """Expectation under the noise model, with no shot noise."""
        self._verify_noise_applied(circuit)
        dm = self._density_matrix(circuit)
        return float(dm.expectation_value(_as_observable(observable)).real)

    def sampled(self, circuit: Any, observable: Any, shots: int,
                seed: int) -> float:
        SparsePauliOp, _ = _require_qiskit()
        self._verify_noise_applied(circuit)
        dm = self._density_matrix(circuit)
        rng = random.Random(seed)
        obs = _as_observable(observable)
        total = 0.0
        for pauli, coeff in zip(obs.paulis, obs.coeffs):
            term = float(dm.expectation_value(SparsePauliOp(pauli)).real)
            total += float(coeff.real) * _sample_pauli(term, shots, rng)
        return total

    def noiseless_twin(self) -> StatevectorSource:
        return StatevectorSource()
