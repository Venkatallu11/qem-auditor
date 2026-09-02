"""The public entry points: the Auditor facade and the CLI."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from qem_auditor import Auditor, Verdict, record
from qem_auditor.cli import EXIT_BAD_RECORD, EXIT_NOT_CERTIFIED, EXIT_OK, main

from .helpers import make_experiment


class AuditorTest(unittest.TestCase):
    def test_accepts_an_experiment_object(self):
        self.assertIs(Auditor().audit(make_experiment()).verdict,
                      Verdict.CERTIFIED_UNDER_SCOPE)

    def test_accepts_a_dict(self):
        data = json.loads(record.dumps(make_experiment()))
        self.assertIs(Auditor().audit(data).verdict, Verdict.CERTIFIED_UNDER_SCOPE)

    def test_accepts_a_path(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.json"
            record.save(make_experiment(), p)
            self.assertIs(Auditor().audit(p).verdict, Verdict.CERTIFIED_UNDER_SCOPE)

    def test_passed_is_narrow(self):
        """Only certification counts as a pass. PROMISING is permission to
        keep working, not a result."""
        self.assertTrue(Auditor().audit(make_experiment()).passed)
        incomplete = make_experiment(real_hardware_full_validation=False)
        result = Auditor().audit(incomplete)
        self.assertIs(result.verdict, Verdict.PROMISING)
        self.assertFalse(result.passed)

    def test_result_bundles_everything(self):
        result = Auditor().audit(make_experiment(ideal_control=False,
                                                 replicate_errors_kcal=[0.10] * 4))
        self.assertTrue(result.failure_modes)
        self.assertTrue(result.claim.render())
        self.assertIsNotNone(result.next_experiment)

    def test_verify_without_an_adapter_explains_itself(self):
        with self.assertRaises(RuntimeError) as ctx:
            Auditor().verify_determinism(make_experiment(), lambda: 1.0)
        self.assertIn("adapter", str(ctx.exception))


class CliTest(unittest.TestCase):
    def _run(self, argv):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(argv)
        return code, out.getvalue()

    def test_template_produces_a_loadable_record(self):
        code, out = self._run(["template"])
        self.assertEqual(code, EXIT_OK)
        exp = record.loads(out)
        self.assertIsNone(exp.controls.ideal_control,
                          "a template must not pre-assert controls as passed")

    def test_audit_of_a_certified_record_exits_zero(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.json"
            record.save(make_experiment(), p)
            code, _ = self._run(["audit", str(p)])
            self.assertEqual(code, EXIT_OK)

    def test_audit_of_an_uncertified_record_exits_nonzero(self):
        """So CI can gate on it."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.json"
            record.save(make_experiment(ideal_control=False), p)
            code, _ = self._run(["audit", str(p)])
            self.assertEqual(code, EXIT_NOT_CERTIFIED)

    def test_unreadable_record_exits_two(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.json"
            p.write_text("{not json")
            code, _ = self._run(["audit", str(p)])
            self.assertEqual(code, EXIT_BAD_RECORD)

    def test_json_output_is_machine_readable(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.json"
            record.save(make_experiment(ideal_control=False,
                                        replicate_errors_kcal=[0.1] * 4), p)
            _, out = self._run(["audit", str(p), "--json"])
            data = json.loads(out)
            self.assertEqual(data["verdict"], "INVALID")
            self.assertTrue(data["gates"])
            self.assertTrue(data["failure_modes"])
            self.assertTrue(data["licence"])

    def test_validate_accepts_a_good_record(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.json"
            record.save(make_experiment(), p)
            self.assertEqual(self._run(["validate", str(p)])[0], EXIT_OK)

    def test_validate_rejects_an_inconsistent_record(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.json"
            record.save(make_experiment(shots=0), p)
            code = main(["validate", str(p)])
            self.assertEqual(code, EXIT_BAD_RECORD)


if __name__ == "__main__":
    unittest.main()


class LoadFailureNamesTheRemedyTest(unittest.TestCase):
    """The first thing a new reader does is follow the quick start.

    The template `check --template` prints builds its circuit with qiskit,
    which the dependency-free core install does not have. "No module named
    'qiskit'" is true, and useless at that moment, so the error carries
    the fix. Found by installing the package into a clean virtualenv and
    following the README as written.
    """

    def _check(self, source: str):
        from contextlib import redirect_stderr
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "mine.py"
            script.write_text(source)
            buffer = io.StringIO()
            with redirect_stderr(buffer):
                code = main(["check", str(script)])
        return code, buffer.getvalue()

    def test_a_missing_qiskit_says_which_extra_to_install(self):
        code, message = self._check('raise ImportError("No module named \'qiskit\'")\n')
        self.assertEqual(code, EXIT_BAD_RECORD)
        self.assertIn('pip install -e ".[adapters]"', message)

    def test_an_unrelated_import_error_gets_no_invented_remedy(self):
        """Guessing a fix for an import this package knows nothing about
        would be worse than silence."""
        code, message = self._check("import a_package_that_does_not_exist_xyz\n")
        self.assertEqual(code, EXIT_BAD_RECORD)
        self.assertNotIn("adapters", message)
