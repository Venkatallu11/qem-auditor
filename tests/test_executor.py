"""The executor: does it run what it can, and refuse to pretend about what
it cannot?
"""
import unittest

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import SparsePauliOp

    from qem_auditor.adapters.qiskit_adapter import QiskitAdapter

    HAVE_QISKIT = True
except ImportError:  # pragma: no cover
    HAVE_QISKIT = False

from qem_auditor import audit
from qem_auditor.adversary import AdversarialScientist, GRAMMAR
from qem_auditor.executor import AttackExecutor, AttackReport

from .helpers import make_experiment

BASIS = ["u", "cx"]


def _base():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def _folded(base, folds):
    qc = base.copy()
    for _ in range(folds):
        qc.cx(0, 1)
        qc.cx(0, 1)
    return qc


class NotRunSemanticsTest(unittest.TestCase):
    """The distinction that makes the whole thing honest: an attack that
    could not be run is not an attack the claim survived."""

    def test_an_unrunnable_attack_reports_survived_as_none(self):
        exp = make_experiment()
        plan = AdversarialScientist().propose(exp, audit(exp))
        report = AttackExecutor().run(exp, plan)
        for outcome in report.outcomes:
            if not outcome.ran:
                self.assertIsNone(outcome.survived)
                self.assertNotEqual(outcome.survived, False)

    def test_unrun_attacks_are_counted_separately_from_survivors(self):
        exp = make_experiment()
        plan = AdversarialScientist().propose(exp, audit(exp))
        report = AttackExecutor().run(exp, plan)
        self.assertTrue(report.not_run)
        for outcome in report.not_run:
            self.assertNotIn(outcome, report.survived)

    def test_missing_adapter_is_reported_not_silently_passed(self):
        exp = make_experiment()
        plan = AdversarialScientist().propose(exp, audit(exp))
        report = AttackExecutor(adapter=None).run(exp, plan)
        self.assertTrue(any("adapter" in o.detail for o in report.not_run))

    def test_missing_artifacts_are_reported(self):
        exp = make_experiment()
        plan = AdversarialScientist().propose(exp, audit(exp))
        report = AttackExecutor(adapter=object()).run(exp, plan)
        self.assertTrue(report.not_run)


class HookTest(unittest.TestCase):
    def test_a_domain_hook_runs_an_otherwise_unrunnable_attack(self):
        exp = make_experiment()
        plan = AdversarialScientist().propose(exp, audit(exp))
        executor = AttackExecutor(hooks={
            "T_label": lambda e: ("artifact", "shuffled fit was comparable")})
        report = executor.run(exp, plan)
        label = next(o for o in report.outcomes if o.attack.transformation == "T_label")
        self.assertTrue(label.ran)
        self.assertIs(label.survived, False)

    def test_a_raising_hook_is_not_a_pass(self):
        exp = make_experiment()
        plan = AdversarialScientist().propose(exp, audit(exp))

        def broken(e):
            raise RuntimeError("fitting code blew up")

        report = AttackExecutor(hooks={"T_label": broken}).run(exp, plan)
        label = next(o for o in report.outcomes if o.attack.transformation == "T_label")
        self.assertIsNone(label.survived)
        self.assertIn("blew up", label.detail)


@unittest.skipUnless(HAVE_QISKIT, "qiskit not installed")
class RealExecutionTest(unittest.TestCase):
    """Against the real transpiler, reproducing the historical failure."""

    def _plan_and_run(self, optimization_level):
        exp = make_experiment()
        exp.controls.provenance.clear()
        plan = AdversarialScientist().propose(exp, audit(exp))
        base = _base()
        submitted = transpile(_folded(base, 2), basis_gates=BASIS,
                              optimization_level=optimization_level)
        return AttackExecutor(adapter=QiskitAdapter(seed=5)).run(
            exp, plan, base_circuit=base, submitted_circuit=submitted,
            computation=lambda: 1.0)

    def test_optimization_level_3_is_falsified_by_the_compiler_attack(self):
        report = self._plan_and_run(3)
        compiler = next(o for o in report.outcomes
                        if o.attack.transformation == "T_compiler")
        self.assertIs(compiler.survived, False)

    def test_optimization_level_0_survives_it(self):
        report = self._plan_and_run(0)
        compiler = next(o for o in report.outcomes
                        if o.attack.transformation == "T_compiler")
        self.assertIs(compiler.survived, True)

    def test_a_composed_attack_runs_every_part(self):
        exp = make_experiment()
        exp.controls.provenance.clear()
        plan = AdversarialScientist().propose(exp, audit(exp))
        composed = [a for a in plan.attacks if " o " in a.transformation]
        self.assertTrue(composed, "expected the proposer to compose interacting attacks")
        base = _base()
        submitted = transpile(_folded(base, 2), basis_gates=BASIS, optimization_level=3)
        report = AttackExecutor(adapter=QiskitAdapter(seed=5)).run(
            exp, plan, base_circuit=base, submitted_circuit=submitted,
            circuit=base, observable=SparsePauliOp("ZZ"),
            mitigator=lambda expectation: expectation(base, SparsePauliOp("ZZ")),
            computation=lambda: 1.0)
        out = next(o for o in report.outcomes if " o " in o.attack.transformation)
        self.assertTrue(out.ran)
        self.assertIs(out.survived, False)  # the compiler part falsifies it


if __name__ == "__main__":
    unittest.main()


class FitBasedAttackTest(unittest.TestCase):
    """T_label, T_sign and T_shot, now executable through a reconstructor."""

    def _pieces(self):
        from statistics import mean

        from qem_auditor.reconstruct import FitData, Measurement

        ideal = {"XX": 0.8, "YY": -0.6, "ZZ": 0.4}

        data = FitData([
            Measurement(slot=slot, label=label, value=v * 0.9 + 0.001 * (d + 1),
                        sigma=0.01, shots=1000, draw=d)
            for d in range(4) for slot in ("u0", "u1")
            for label, v in ideal.items()])

        class Genuine:
            def fit(self, data):
                return {lab: mean([m.value for m in data.measurements
                                   if m.label == lab]) / v
                        for lab, v in ideal.items()}

            def reconstruct(self, fit, data):
                return sum(mean([m.value for m in data.measurements if m.label == lab])
                           / (fit.get(lab) or 1.0) for lab in ideal)

            def goodness_of_fit(self, fit, data):
                return sum(((m.value - ideal[m.label] * fit[m.label]) / m.sigma) ** 2
                           for m in data.measurements) / len(data.measurements)

        class Flexible:
            def fit(self, data):
                return {(m.slot, m.label, m.draw): m.value for m in data.measurements}

            def reconstruct(self, fit, data):
                return sum(fit.values())

            def goodness_of_fit(self, fit, data):
                return 0.0

        return data, Genuine(), Flexible()

    def _run(self, name, reconstructor, data, **extra):
        from qem_auditor.adversary import AttackPlan, GRAMMAR

        exp = make_experiment()
        plan = AttackPlan(exp.experiment_id, [GRAMMAR[name](exp)])
        report = AttackExecutor().run(exp, plan, reconstructor=reconstructor,
                                      fit_data=data, **extra)
        return report.outcomes[0]

    def test_they_need_no_backend_adapter(self):
        """They need the claimant's fitting code, not a quantum backend."""
        data, genuine, _ = self._pieces()
        self.assertTrue(self._run("T_label", genuine, data).ran)

    def test_label_shuffle_separates_genuine_from_flexible(self):
        data, genuine, flexible = self._pieces()
        self.assertIs(self._run("T_label", genuine, data).survived, True)
        self.assertIs(self._run("T_label", flexible, data).survived, False)

    def test_a_flexible_model_is_falsified_by_the_sign_flip(self):
        data, _, flexible = self._pieces()
        self.assertIs(self._run("T_sign", flexible, data).survived, False)

    def test_missing_artifacts_are_reported_not_passed(self):
        from qem_auditor.adversary import AttackPlan, GRAMMAR

        exp = make_experiment()
        for name in ("T_label", "T_sign", "T_shot"):
            with self.subTest(attack=name):
                plan = AttackPlan(exp.experiment_id, [GRAMMAR[name](exp)])
                outcome = AttackExecutor().run(exp, plan).outcomes[0]
                self.assertIsNone(outcome.survived)
                self.assertIn("reconstructor", outcome.detail)

    def test_a_two_label_dataset_cannot_judge_rather_than_passing(self):
        from qem_auditor.reconstruct import FitData, Measurement

        data, genuine, _ = self._pieces()
        two = FitData([m for m in data.measurements if m.label in ("XX", "YY")])
        outcome = self._run("T_label", genuine, two)
        self.assertIsNone(outcome.survived)
        self.assertIn("cannot judge", outcome.detail)

    def test_shot_attack_needs_more_than_one_draw(self):
        from qem_auditor.reconstruct import FitData

        data, genuine, _ = self._pieces()
        one = FitData([m for m in data.measurements if m.draw == 0])
        outcome = self._run("T_shot", genuine, one)
        self.assertIsNone(outcome.survived)
        self.assertIn("draw", outcome.detail)

    def test_an_unresponsive_reconstruction_cannot_judge(self):
        """Zero spread under both knobs is not evidence that shots dominate."""
        from qem_auditor.reconstruct import FitData

        data, _, _ = self._pieces()

        class Constant:
            def fit(self, data):
                return {}

            def reconstruct(self, fit, data):
                return 1.0

            def goodness_of_fit(self, fit, data):
                return 1.0

        outcome = self._run("T_shot", Constant(), data)
        self.assertIsNone(outcome.survived)
        self.assertIn("did not move", outcome.detail)


class OptionalCapabilityAttackTest(unittest.TestCase):
    """T_leakage, T_calibration and T_correlation, through the optional
    capabilities."""

    def _data(self):
        from qem_auditor.reconstruct import FitData, Measurement

        return FitData([
            Measurement(slot=s, label=l, value=v * 0.9 + 0.002 * (d + 1),
                        sigma=0.01, shots=1000, draw=d)
            for d in range(4) for s in ("u0", "u1")
            for l, v in (("XX", 0.8), ("YY", -0.6), ("ZZ", 0.4))])

    def _run(self, name, reconstructor, **extra):
        from qem_auditor.adversary import AttackPlan, GRAMMAR

        exp = make_experiment()
        plan = AttackPlan(exp.experiment_id, [GRAMMAR[name](exp)])
        return AttackExecutor().run(exp, plan, reconstructor=reconstructor,
                                    fit_data=self._data(), **extra).outcomes[0]

    def test_a_pipeline_without_the_capability_reports_how_to_add_it(self):
        class Bare:
            def fit(self, d):
                return {}

            def reconstruct(self, f, d):
                return 1.0

            def goodness_of_fit(self, f, d):
                return 1.0

        for name, expected in (("T_leakage", "free_parameters"),
                               ("T_calibration", "noise_parameters"),
                               ("T_correlation", "fit_without_structure")):
            with self.subTest(attack=name):
                outcome = self._run(name, Bare())
                self.assertIsNone(outcome.survived)
                self.assertIn(expected, outcome.detail)

    def test_leakage_catches_a_degenerating_parameter(self):
        from statistics import mean

        class Degenerate:
            def fit(self, d):
                return {}

            def reconstruct(self, f, d):
                return mean([m.value for m in d.measurements])

            def goodness_of_fit(self, f, d):
                return 1.0

            def free_parameters(self):
                return {"radius": (0.0, 1.0)}

            def evaluate_at(self, name, value, d):
                measured = mean([m.value for m in d.measurements])
                return value * measured + (1.0 - value) * 42.0

        outcome = self._run("T_leakage", Degenerate())
        self.assertIs(outcome.survived, False)
        self.assertIn("deriving rather than measuring", outcome.detail)

    def test_calibration_catches_a_fragile_noise_dependence(self):
        from statistics import mean

        class Fragile:
            def fit(self, d):
                return {}

            def reconstruct(self, f, d):
                return mean([m.value for m in d.measurements])

            def goodness_of_fit(self, f, d):
                return 1.0

            def noise_parameters(self):
                return {"p": (0.02, 0.4)}

            def evaluate_under_noise(self, params, d):
                return mean([m.value for m in d.measurements]) / params["p"]

        outcome = self._run("T_calibration", Fragile(), calibration_draws=64)
        self.assertIs(outcome.survived, False)
        self.assertIn("one lucky calibration", outcome.detail)

    def test_correlation_catches_a_load_bearing_assumption(self):
        from statistics import mean

        class Propped:
            """Stable only because of its constraint; wild without it."""

            def fit(self, d):
                return {"scale": 1.0}

            def reconstruct(self, f, d):
                return mean([m.value for m in d.measurements])

            def goodness_of_fit(self, f, d):
                return 1.0

            def fit_without_structure(self, d):
                # Without pooling, the scale comes from the difference of
                # two same-label measurements from adjacent draws. They
                # differ by ~0.002 against a shot sigma of 0.01, so the
                # denominator swings through zero under resampling: the
                # classic ill-conditioned fit that only the pooling
                # constraint was holding together.
                same_label = [m for m in d.measurements if m.label == "XX"]
                return {"scale": same_label[0].value - same_label[2].value}

            def reconstruct_without_structure(self, f, d):
                scale = f["scale"]
                return mean([m.value for m in d.measurements]) / (
                    scale if abs(scale) > 1e-12 else 1e-12)

        outcome = self._run("T_correlation", Propped())
        self.assertIs(outcome.survived, False)
        self.assertIn("buying existence", outcome.detail)
