"""The wiring: does the auditor actually use what it can remember?

Every capability tested elsewhere in this suite existed before these
tests did, and none of it reached anybody: `AuditResult.consult` and
`.recalled` were never populated, the CLI had no way to ask, and nothing
persisted. These test the connection rather than the parts.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from qem_auditor import Auditor, Verdict
from qem_auditor.prescribe import ErrorBudget, ErrorSource, budget_from_calibration
from qem_auditor.store import DEFAULT_DIRECTORY, ENV_VAR, Store, default_directory

from .helpers import make_experiment

E = ErrorSource
REPO = Path(__file__).resolve().parent.parent


def a_budget():
    return budget_from_calibration(
        two_qubit_gates=2, one_qubit_gates=6, measured_qubits=2,
        two_qubit_error=0.003, one_qubit_error=0.0001,
        readout_error=0.029, shots=40_000)


class StoreTest(unittest.TestCase):
    def test_opening_a_store_writes_nothing(self):
        """Importing or opening should not create files. Only saving does."""
        with tempfile.TemporaryDirectory() as tmp:
            Store.open(Path(tmp) / "fresh")
            self.assertFalse((Path(tmp) / "fresh").exists())

    def test_saving_creates_both_halves(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store.open(tmp)
            self.assertTrue(store.save())
            self.assertTrue((Path(tmp) / "memory.json").exists())
            self.assertTrue((Path(tmp) / "ledger.json").exists())

    def test_an_ephemeral_store_accumulates_but_writes_nothing(self):
        store = Store.ephemeral()
        self.assertFalse(store.persistent)
        self.assertFalse(store.save())

    def test_the_location_is_overridable(self):
        """So a project can keep its corpus beside the project rather than
        in a home directory shared with unrelated work."""
        import os

        previous = os.environ.get(ENV_VAR)
        os.environ[ENV_VAR] = "/tmp/elsewhere"
        try:
            self.assertEqual(str(default_directory()), "/tmp/elsewhere")
        finally:
            os.environ.pop(ENV_VAR, None)
            if previous is not None:
                os.environ[ENV_VAR] = previous

    def test_the_default_is_not_the_working_directory(self):
        """Auditing in a repo should not litter it."""
        self.assertTrue(DEFAULT_DIRECTORY.startswith("~"))


class AuditorUsesTheStoreTest(unittest.TestCase):
    def test_without_a_store_the_auditor_recalls_nothing(self):
        result = Auditor().audit(make_experiment())
        self.assertIsNone(result.recalled)

    def test_with_a_store_the_second_audit_recalls_the_first(self):
        auditor = Auditor(store=Store.ephemeral())
        first = make_experiment(ideal_control=False)
        first.experiment_id = "first"
        auditor.audit(first)

        second = make_experiment(ideal_control=False)
        second.experiment_id = "second"
        recalled = auditor.audit(second).recalled
        self.assertFalse(recalled.is_empty)
        self.assertIn("first", [c.experiment_id for c in recalled.seen_before])

    def test_what_failed_before_is_offered_as_what_to_check_first(self):
        auditor = Auditor(store=Store.ephemeral())
        for i in range(2):
            exp = make_experiment(ideal_control=False)
            exp.experiment_id = f"run_{i}"
            auditor.audit(exp)
        recalled = auditor.audit(make_experiment(ideal_control=False)).recalled
        self.assertEqual(recalled.check_first[0][0], "ideal_control")

    def test_memory_does_not_reach_the_verdict(self):
        """The rule, tested through the wiring rather than the module: a
        clean circuit that memory associates only with failures still
        certifies."""
        auditor = Auditor(store=Store.ephemeral())
        for i in range(5):
            exp = make_experiment(ideal_control=False)
            exp.experiment_id = f"bad_{i}"
            auditor.audit(exp)
        clean = make_experiment()
        clean.experiment_id = "clean"
        self.assertIs(auditor.audit(clean).verdict, Verdict.CERTIFIED_UNDER_SCOPE)


class AuditorPrescribesTest(unittest.TestCase):
    def test_no_budget_means_no_prescription(self):
        self.assertIsNone(Auditor().audit(make_experiment()).consult)

    def test_and_the_report_says_why_rather_than_going_quiet(self):
        rendered = Auditor().audit(make_experiment()).render()
        self.assertIn("NO REMEDY OFFERED", rendered)
        self.assertIn("budget_from_calibration", rendered)

    def test_a_budget_produces_a_prescription(self):
        result = Auditor().audit(make_experiment(), budget=a_budget())
        self.assertIsNotNone(result.consult)
        self.assertTrue(result.consult.prescriptions)

    def test_a_readout_dominated_budget_does_not_lead_with_zne(self):
        """The wiring carries the finding through, not just the module."""
        result = Auditor().audit(make_experiment(), budget=a_budget())
        self.assertNotEqual(result.consult.leading.action,
                            "zero-noise extrapolation (ZNE)")

    def test_the_prescription_reaches_the_rendered_report(self):
        rendered = Auditor().audit(make_experiment(), budget=a_budget()).render()
        self.assertIn("WHAT TO DO ABOUT IT", rendered)
        self.assertIn("Will NOT help here", rendered)

    def test_declaring_a_symmetry_changes_the_advice(self):
        without = Auditor().audit(make_experiment(), budget=a_budget())
        with_it = Auditor().audit(make_experiment(), budget=a_budget(),
                                  symmetry_available=True)
        self.assertNotEqual([p.action for p in without.consult.prescriptions],
                            [p.action for p in with_it.consult.prescriptions])


class CommandLineTest(unittest.TestCase):
    """The path most users meet. Run as a subprocess so what is tested is
    the command, not an internal function that resembles it."""

    def run_cli(self, *args, store=None):
        import os

        env = dict(os.environ, PYTHONPATH=str(REPO))
        if store is not None:
            env[ENV_VAR] = str(store)
        return subprocess.run(
            [sys.executable, "-m", "qem_auditor.cli", *args],
            capture_output=True, text=True, env=env, cwd=REPO)

    def write_record(self, directory, name, **overrides):
        from qem_auditor import record

        exp = make_experiment(**overrides)
        exp.experiment_id = name
        path = Path(directory) / f"{name}.json"
        record.save(exp, path)
        return path

    def write_calibration(self, directory):
        path = Path(directory) / "cal.json"
        path.write_text(json.dumps({
            "two_qubit_gates": 2, "one_qubit_gates": 6, "measured_qubits": 2,
            "two_qubit_error": 0.003, "one_qubit_error": 0.0001,
            "readout_error": 0.029, "shots": 40000}))
        return path

    def test_audit_without_a_calibration_says_what_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            record_path = self.write_record(tmp, "run", ideal_control=False)
            out = self.run_cli("audit", str(record_path), "--no-store").stdout
            self.assertIn("NO REMEDY OFFERED", out)
            self.assertIn("--calibration", out)

    def test_audit_with_a_calibration_prescribes(self):
        with tempfile.TemporaryDirectory() as tmp:
            record_path = self.write_record(tmp, "run", ideal_control=False)
            cal = self.write_calibration(tmp)
            out = self.run_cli("audit", str(record_path), "--calibration",
                               str(cal), "--no-store").stdout
            self.assertIn("WHAT TO DO ABOUT IT", out)

    def test_an_incomplete_calibration_is_refused_by_name(self):
        """A default for any of those would be a guess wearing a number's
        clothes."""
        with tempfile.TemporaryDirectory() as tmp:
            record_path = self.write_record(tmp, "run")
            cal = Path(tmp) / "bad.json"
            cal.write_text(json.dumps({"two_qubit_error": 0.003}))
            result = self.run_cli("audit", str(record_path), "--calibration",
                                  str(cal), "--no-store")
            self.assertIn("missing", result.stderr)
            self.assertIn("readout_error", result.stderr)

    def test_audits_accumulate_across_invocations(self):
        """The whole point: a tool that forgets between runs is not one."""
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "store"
            first = self.write_record(tmp, "monday", ideal_control=False)
            second = self.write_record(tmp, "tuesday", ideal_control=False)
            self.run_cli("audit", str(first), store=store)
            out = self.run_cli("audit", str(second), store=store).stdout
            self.assertIn("WHAT THIS REMINDS THE AUDITOR OF", out)
            self.assertIn("monday", out)

    def test_no_store_leaves_no_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "store"
            record_path = self.write_record(tmp, "run")
            self.run_cli("audit", str(record_path), "--no-store", store=store)
            self.assertFalse(store.exists())

    def test_the_remember_command_shows_what_is_held(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "store"
            record_path = self.write_record(tmp, "monday", ideal_control=False)
            self.run_cli("audit", str(record_path), store=store)
            out = self.run_cli("remember", store=store).stdout
            self.assertIn("monday", out)
            self.assertIn("1 circuit remembered", out)

    def test_remember_can_recall_against_one_circuit(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "store"
            first = self.write_record(tmp, "monday", ideal_control=False)
            self.run_cli("audit", str(first), store=store)
            out = self.run_cli("remember", "--circuit", str(first),
                               store=store).stdout
            self.assertIn("audited before", out)
            self.assertIn("gates still decide", out)

    def test_json_output_carries_the_guidance_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            record_path = self.write_record(tmp, "run", ideal_control=False)
            cal = self.write_calibration(tmp)
            out = self.run_cli("audit", str(record_path), "--calibration",
                               str(cal), "--no-store", "--json").stdout
            data = json.loads(out)
            self.assertTrue(data["prescriptions"])
            self.assertTrue(data["will_not_help"])

    def test_the_store_location_is_announced_when_it_is_created(self):
        """Accumulating by default is defensible; doing it invisibly is
        not."""
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "store"
            record_path = self.write_record(tmp, "run")
            result = self.run_cli("audit", str(record_path), store=store)
            self.assertIn("remembering this audit in", result.stderr)


if __name__ == "__main__":
    unittest.main()
