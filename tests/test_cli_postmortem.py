"""The hardware post-mortem, from the command line.

The three modules a real trapped-ion run paid for -- counts, vendor and
cost -- were reachable only from Python. Someone holding a job record
and a bill is exactly the person least likely to write a script to read
them, so this is the door.
"""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from qem_auditor.cli import EXIT_BAD_RECORD, EXIT_NOT_CERTIFIED, EXIT_OK, main

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ionq_forte_job.json"


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


def in_a_file(payload, directory):
    path = Path(directory) / "job.json"
    path.write_text(json.dumps(payload))
    return str(path)


class TemplateTest(unittest.TestCase):

    def test_the_template_is_readable_by_the_command_that_prints_it(self):
        """A template nobody can feed back in is documentation pretending
        to be a starting point."""
        code, out, _ = run("postmortem", "--template")
        self.assertEqual(code, EXIT_OK)
        with tempfile.TemporaryDirectory() as d:
            path = in_a_file(json.loads(out), d)
            code, printed, _ = run("postmortem", path)
        self.assertEqual(code, EXIT_OK)
        self.assertIn("as EXECUTED", printed)

    def test_no_path_and_no_template_says_what_to_do(self):
        code, _, err = run("postmortem")
        self.assertEqual(code, EXIT_BAD_RECORD)
        self.assertIn("--template", err)


class RealJobTest(unittest.TestCase):
    """The real IonQ job, read end to end."""

    def test_it_reads_the_vendor_record(self):
        code, out, _ = run("postmortem", str(FIXTURE))
        self.assertEqual(code, EXIT_OK)
        self.assertIn("120 one-qubit, 11 two-qubit", out)
        self.assertIn("none declared", out)

    def test_it_checks_the_counts_against_the_declared_shots(self):
        code, out, _ = run("postmortem", str(FIXTURE))
        self.assertIn("counts look sound", out)

    def test_corrupted_counts_are_caught_and_change_the_exit_code(self):
        record = json.loads(FIXTURE.read_text())
        record["counts"] = {b: n * 100 for b, n in record["counts"].items()}
        with tempfile.TemporaryDirectory() as d:
            code, out, _ = run("postmortem", in_a_file(record, d))
        self.assertEqual(code, EXIT_NOT_CERTIFIED)
        self.assertIn("TOO TIGHT", out)

    def test_the_compilation_blowup_is_reported_against_what_was_written(self):
        code, out, _ = run("postmortem", str(FIXTURE),
                           "--one-qubit-gates", "20", "--two-qubit-gates", "11")
        self.assertEqual(code, EXIT_NOT_CERTIFIED)
        self.assertIn("120 after compilation", out)

    def test_pricing_uses_the_EXECUTED_gates_not_the_written_ones(self):
        """Cost is charged per executed gate per shot. Pricing the
        circuit as written would quote for one the machine never ran --
        which this did, until a run of it printed 20 where the record
        said 120."""
        code, out, _ = run("postmortem", str(FIXTURE), "--budget", "170",
                           "--shots", "2000",
                           "--one-qubit-gates", "20", "--two-qubit-gates", "11")
        self.assertIn("120 one-qubit and 11 two-qubit gates", out)
        # Padded into a column, so match the number rather than "$64.02".
        self.assertIn("64.02", out)
        self.assertIn("up to 806 shots", out)
        self.assertNotIn("1,653 shots", out)

    def test_a_budget_with_no_gate_counts_anywhere_says_so(self):
        record = json.loads(FIXTURE.read_text())
        record["raw_job_metadata"]["stats"] = {}
        with tempfile.TemporaryDirectory() as d:
            code, out, _ = run("postmortem", in_a_file(record, d),
                               "--budget", "170")
        self.assertIn("no gate counts to price with", out)


class IBMPathTest(unittest.TestCase):

    def test_an_options_mapping_is_recognised_as_IBM(self):
        with tempfile.TemporaryDirectory() as d:
            code, out, _ = run("postmortem",
                               in_a_file({"resilience_level": 1}, d),
                               "--claims-unmitigated")
        self.assertEqual(code, EXIT_NOT_CERTIFIED)
        self.assertIn("readout error mitigation", out)
        self.assertIn("measurement twirling", out)

    def test_level_zero_passes(self):
        with tempfile.TemporaryDirectory() as d:
            code, out, _ = run("postmortem",
                               in_a_file({"resilience_level": 0}, d),
                               "--claims-unmitigated")
        self.assertEqual(code, EXIT_OK)
        self.assertIn("really is a raw run", out)


class UnknownShapeTest(unittest.TestCase):

    def test_something_that_is_neither_is_refused_by_name(self):
        """Guessing which vendor wrote an unrecognised file is how a
        reader invents fields that were never there."""
        with tempfile.TemporaryDirectory() as d:
            code, _, err = run("postmortem", in_a_file({"hello": 1}, d))
        self.assertEqual(code, EXIT_BAD_RECORD)
        self.assertIn("neither an IonQ job record", err)
        self.assertIn("--template", err)

    def test_a_file_that_is_not_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.json"
            path.write_text("not json at all")
            code, _, err = run("postmortem", str(path))
        self.assertEqual(code, EXIT_BAD_RECORD)
        self.assertIn("could not read", err)


if __name__ == "__main__":
    unittest.main()
