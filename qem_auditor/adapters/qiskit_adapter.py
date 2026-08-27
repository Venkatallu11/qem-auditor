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
from .sources import AerNoiseSource, ExpectationSource, StatevectorSource

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
    """Runs the mechanizable controls against Qiskit circuits.

    `source` decides where expectation values come from. The default is a
    noiseless statevector; pass an `AerNoiseSource` (or, later, an IBM or
    Quantinuum source) to run the claim under real device noise.

    The ideal control ignores `source` on purpose and always runs
    noiseless. Its whole content is "does this method break with no noise
    to correct", so running it through a noisy source would quietly turn
    it into a different, much weaker check.
    """

    name = "qiskit"

    def __init__(self, seed: int = 0,
                 source: Optional[ExpectationSource] = None) -> None:
        self.seed = seed
        self.source = source or StatevectorSource()
        _require_qiskit()

    @property
    def is_noisy(self) -> bool:
        return not self.source.is_noiseless

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
                              degradation_factor: float = 10.0,
                              trials: int = 8) -> ControlMeasurement:
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

        Judged over `trials` paired draws, not one. The ratio of two noisy
        single draws is itself extremely noisy -- a lucky raw draw makes a
        well-behaved estimator look catastrophic, and an unlucky one hides
        a real blowup. Comparing typical magnitudes instead is the same
        correction paired trials make in measure_mitigation_benefit, and
        it was a single-draw ratio here reporting 43x for an estimator
        whose true amplification is about 4.4x that prompted it.
        """
        _, SparsePauliOp, Statevector = _require_qiskit()
        # Noiseless throughout, whatever source the adapter was built with.
        noiseless = self.source.noiseless_twin()
        exact = noiseless.exact(circuit, observable)
        if trials < 2:
            raise MeasurementError(
                "a single trial cannot separate amplification from a lucky draw")

        raw_errors, mitigated_errors, total_calls = [], [], 0
        base_seed = self.seed
        for t in range(trials):
            try:
                self.seed = base_seed + 1000 * (t + 1)
                oracle, calls = self._oracle(shots, noiseless)
                mitigated = float(mitigator(oracle))
                raw = noiseless.sampled(circuit, observable, shots, self.seed)
            except Exception as e:
                self.seed = base_seed
                raise MeasurementError(
                    f"the mitigation pipeline raised on noiseless input: {e}. That is "
                    f"itself worth knowing -- a pipeline that cannot run on a clean "
                    f"model has not been tested on one.") from e
            finally:
                self.seed = base_seed
            raw_errors.append(abs(raw - exact))
            mitigated_errors.append(abs(mitigated - exact))
            total_calls += calls["n"]

        raw_errors.sort()
        mitigated_errors.sort()
        raw_error = raw_errors[len(raw_errors) // 2]
        mitigated_error = mitigated_errors[len(mitigated_errors) // 2]
        evidence = {
            "exact": exact,
            "raw_error": raw_error,
            "mitigated_error": mitigated_error,
            "shots": shots,
            "trials": trials,
            "oracle_calls": total_calls,
            "source": noiseless.name,
            "statistic": "median over paired trials",
        }
        ratio = mitigated_error / max(raw_error, 1e-12)
        evidence["amplification"] = ratio

        if mitigated_error > degradation_factor * max(raw_error, 1e-12):
            return ControlMeasurement(
                "ideal_control", False,
                f"on a noiseless model, mitigation amplified the median error "
                f"{ratio:.1f}x across {trials} paired trials "
                f"({raw_error:.6g} -> {mitigated_error:.6g}) -- with zero physical "
                f"noise to correct, this is the estimator amplifying shot noise",
                evidence)
        return ControlMeasurement(
            "ideal_control", True,
            f"on a noiseless model, mitigation is not pathologically conditioned "
            f"(median over {trials} trials: {raw_error:.6g} -> {mitigated_error:.6g}, "
            f"{ratio:.1f}x -- within the {degradation_factor:.0f}x bar; compare "
            f"against your estimator's own coefficient norm)",
            evidence)

    # -- does mitigation actually help? --------------------------------

    def measure_mitigation_benefit(self, circuit, observable, mitigator,
                                   shots: int = 20_000, trials: int = 8,
                                   min_improvement: float = 1.1,
                                   min_win_rate: float = 0.75) -> ControlMeasurement:
        """Under real device noise, does the mitigation reduce the error?

        The complement to the ideal control, and a genuinely different
        question. The ideal control establishes that a method does not
        BREAK when there is no noise -- necessary, and nowhere near
        sufficient. A method can pass it and still fail to help at all
        once noise is present, and nothing else in the auditor catches
        that.

        Judged over PAIRED trials rather than one draw, because a single
        comparison cannot separate a real improvement from shot noise: a
        do-nothing mitigator returns a ratio of 1.0 plus noise and so
        "improves" about half the time. Requiring both a median ratio
        above `min_improvement` and a win rate above `min_win_rate`
        mirrors how this family of result is reported honestly elsewhere
        ("wins 66/80 trials"), and a no-op fails both.

        Requires a noisy source, and requires the exact answer to be
        computable. Both hold in simulation, which is where this check
        belongs; on hardware large enough to matter the exact value is not
        available and this cannot be run. The auditor says so rather than
        pretending otherwise.
        """
        if self.source.is_noiseless:
            raise MeasurementError(
                "measure_mitigation_benefit needs a noisy source -- with a noiseless "
                "one there is no error for mitigation to reduce. Build the adapter "
                "with QiskitAdapter(source=AerNoiseSource(noise_model)).")
        if trials < 2:
            raise MeasurementError(
                "a single trial cannot separate a real improvement from shot noise")

        noiseless = self.source.noiseless_twin()
        truth = noiseless.exact(circuit, observable)

        ratios, wins = [], 0
        base_seed = self.seed
        for t in range(trials):
            try:
                self.seed = base_seed + 1000 * (t + 1)
                raw = self.source.sampled(circuit, observable, shots, self.seed)
                oracle, _ = self._oracle(shots, self.source)
                mitigated = float(mitigator(oracle))
            except Exception as e:
                self.seed = base_seed
                raise MeasurementError(
                    f"the mitigation pipeline raised under noise: {e}") from e
            finally:
                self.seed = base_seed
            raw_error = abs(raw - truth)
            mitigated_error = abs(mitigated - truth)
            if raw_error <= 1e-12:
                continue
            ratios.append(raw_error / max(mitigated_error, 1e-12))
            if mitigated_error < raw_error:
                wins += 1

        if len(ratios) < 2:
            return ControlMeasurement(
                "mitigation_benefit", None,
                "the raw result is already exact under this noise model, so there is "
                "nothing for mitigation to improve -- pick a noisier model or a "
                "deeper circuit",
                {"truth": truth, "trials": trials, "source": self.source.name})

        ratios.sort()
        median = ratios[len(ratios) // 2]
        win_rate = wins / len(ratios)
        evidence = {
            "truth": truth, "median_improvement": median, "win_rate": win_rate,
            "wins": wins, "trials": len(ratios), "shots": shots,
            "source": self.source.name, "ratios": ratios,
        }

        if median < min_improvement or win_rate < min_win_rate:
            return ControlMeasurement(
                "mitigation_benefit", False,
                f"under {self.source.name} the mitigation did not clearly help: median "
                f"{median:.2f}x over {len(ratios)} paired trials, winning {wins}/"
                f"{len(ratios)} (needed {min_improvement:.2f}x and "
                f"{min_win_rate:.0%}). Passing the ideal control only shows a method "
                f"does not break without noise; it does not show it helps with noise",
                evidence)
        return ControlMeasurement(
            "mitigation_benefit", True,
            f"under {self.source.name} the mitigation reduced the error by a median "
            f"{median:.2f}x, winning {wins}/{len(ratios)} paired trials",
            evidence)

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
        return self.source.exact(circuit, observable)

    def _sampled_expectation(self, circuit, observable, shots: int,
                             seed: int, source=None) -> float:
        """Expectation plus honest shot noise, from the configured source.

        Weighted sums of Paulis are sampled term by term, which slightly
        overstates the variance -- it ignores the covariance between
        commuting terms measured in one basis. Stated here rather than
        left for a reader to discover; it is conservative in the direction
        that matters.
        """
        return (source or self.source).sampled(circuit, observable, shots, seed)

    def _oracle(self, shots: int, source: ExpectationSource):
        """An expectation oracle over a given source.

        Each call gets its own seed so repeated calls carry independent
        shot noise, as separate real submissions would.
        """
        calls = {"n": 0}

        def oracle(circuit, observable) -> float:
            calls["n"] += 1
            return source.sampled(circuit, observable, shots, self.seed + calls["n"])

        return oracle, calls

    def _noiseless_oracle(self, shots: int):
        return self._oracle(shots, self.source.noiseless_twin())


def _fast_binomial(rng, n: int, p: float) -> int:
    """Normal approximation to Binomial(n, p), for shot counts where
    looping would dominate the runtime. Valid well inside the regime these
    checks use (n >= 20,000)."""
    import math

    mean = n * p
    sd = math.sqrt(max(n * p * (1 - p), 0.0))
    return int(min(n, max(0, round(rng.gauss(mean, sd)))))
