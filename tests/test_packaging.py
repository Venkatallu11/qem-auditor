"""What the installed package promises about itself.

A first release freezes a public namespace. These pin the things that
are cheap to get wrong now and breaking to fix later.
"""
import subprocess
import sys
import unittest

import qem_auditor


class VersionTest(unittest.TestCase):

    def test_the_package_reports_a_version(self):
        self.assertTrue(qem_auditor.__version__)

    def test_the_version_comes_from_the_metadata_not_a_literal(self):
        """Hardcoding it is how a package and its own metadata end up
        disagreeing about what version they are."""
        try:
            from importlib.metadata import version
        except ImportError:  # pragma: no cover
            self.skipTest("no importlib.metadata")
        try:
            installed = version("qem-auditor")
        except Exception:  # pragma: no cover - source tree, never installed
            self.skipTest("not installed as a distribution")
        self.assertEqual(qem_auditor.__version__, installed)


class NamespaceTest(unittest.TestCase):
    """The ambiguity that was there before the first release."""

    def test_the_two_prediction_classes_are_distinguishable(self):
        from qem_auditor.adversary import Prediction as Outcome
        from qem_auditor.predict import Prediction as Method

        self.assertIsNot(Outcome, Method)
        self.assertIs(qem_auditor.OutcomePrediction, Outcome)
        self.assertIs(qem_auditor.MethodPrediction, Method)

    def test_the_ambiguous_name_is_not_exported(self):
        """Exported, it resolved to whichever module was imported last,
        and the wrong one came back with no error."""
        self.assertFalse(hasattr(qem_auditor, "Prediction"))

    def test_prescribe_is_callable_and_its_module_still_reachable(self):
        """The function shadows the module at the root on purpose. That
        is only acceptable while the module path still works."""
        self.assertTrue(callable(qem_auditor.prescribe))
        from qem_auditor.prescribe import CATALOGUE
        self.assertTrue(CATALOGUE)

    def test_every_name_the_package_exports_actually_resolves(self):
        """A stale re-export is an AttributeError someone else finds."""
        for name in dir(qem_auditor):
            if not name.startswith("_"):
                getattr(qem_auditor, name)


class ConsoleScriptTest(unittest.TestCase):

    def run_cli(self, *argv, stdout=subprocess.PIPE):
        return subprocess.run([sys.executable, "-m", "qem_auditor", *argv],
                              stdout=stdout, stderr=subprocess.PIPE, text=True)

    def test_the_entry_point_runs(self):
        done = self.run_cli("template")
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_a_closed_pipe_is_not_a_traceback(self):
        """`qem-auditor postmortem job.json | head` closed the pipe and
        printed a stack trace for a command that did nothing wrong.
        People pipe into head and less constantly."""
        made = self.run_cli("postmortem", "--template")
        self.assertEqual(made.returncode, 0, made.stderr)
        import json, tempfile, os
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "job.json"
            path.write_text(made.stdout)
            head = subprocess.Popen(["head", "-2"], stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL)
            done = subprocess.run(
                [sys.executable, "-m", "qem_auditor", "postmortem", str(path),
                 "--budget", "170"],
                stdout=head.stdin, stderr=subprocess.PIPE, text=True)
            head.stdin.close()
            head.wait()
        self.assertNotIn("BrokenPipeError", done.stderr)
        self.assertNotIn("Traceback", done.stderr)


if __name__ == "__main__":
    unittest.main()
