"""The record format: does a document survive the round trip, and does a
malformed one get rejected rather than silently misread?
"""
import json
import tempfile
import unittest
from pathlib import Path

from qem_auditor import Provenance, ReplicateKind, audit, record
from qem_auditor.record import RecordError

from .helpers import make_experiment


class RoundTripTest(unittest.TestCase):
    def test_round_trip_preserves_the_verdict(self):
        exp = make_experiment()
        back = record.loads(record.dumps(exp))
        self.assertIs(audit(back).verdict, audit(exp).verdict)

    def test_round_trip_is_stable(self):
        text = record.dumps(make_experiment())
        self.assertEqual(record.dumps(record.loads(text)), text)

    def test_enums_survive(self):
        exp = make_experiment()
        back = record.loads(record.dumps(exp))
        self.assertIs(back.outputs.replicates[0].kind, ReplicateKind.INDEPENDENT_SUBMISSION)
        self.assertIs(back.controls.provenance_of("ideal_control"), Provenance.MEASURED)
        self.assertEqual(back.claim_type, exp.claim_type)

    def test_uncertainty_coverage_survives(self):
        back = record.loads(record.dumps(make_experiment()))
        self.assertTrue(back.outputs.uncertainty.is_complete)

    def test_every_benchmark_round_trips(self):
        import run_benchmarks

        for module in run_benchmarks.BENCHMARKS:
            exp = module.EXPERIMENT
            with self.subTest(case=exp.experiment_id):
                back = record.loads(record.dumps(exp))
                self.assertIs(audit(back).verdict, module.EXPECTED_VERDICT)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "rec.json"
            record.save(make_experiment(), p)
            self.assertIs(audit(record.load(p)).verdict, audit(make_experiment()).verdict)


class RejectionTest(unittest.TestCase):
    """A typo in a control name is the difference between 'passed' and
    'never run'. Silence is never acceptable here."""

    def test_unknown_field_is_rejected_not_ignored(self):
        data = json.loads(record.dumps(make_experiment()))
        data["controls"]["ideal_controls"] = True  # plural typo
        with self.assertRaises(RecordError) as ctx:
            record.from_dict(data)
        self.assertIn("ideal_controls", str(ctx.exception))

    def test_the_error_names_the_valid_fields(self):
        data = json.loads(record.dumps(make_experiment()))
        data["nonsense"] = 1
        with self.assertRaises(RecordError) as ctx:
            record.from_dict(data)
        self.assertIn("experiment_id", str(ctx.exception))

    def test_bad_enum_value_names_the_alternatives(self):
        data = json.loads(record.dumps(make_experiment()))
        data["outputs"]["replicates"][0]["kind"] = "TOTALLY_INDEPENDENT"
        with self.assertRaises(RecordError) as ctx:
            record.from_dict(data)
        self.assertIn("INDEPENDENT_SUBMISSION", str(ctx.exception))

    def test_missing_required_field_is_rejected(self):
        data = json.loads(record.dumps(make_experiment()))
        del data["backend"]
        with self.assertRaises(RecordError):
            record.from_dict(data)

    def test_unsupported_format_version_is_rejected(self):
        data = json.loads(record.dumps(make_experiment()))
        data["format_version"] = 999
        with self.assertRaises(RecordError):
            record.from_dict(data)

    def test_invalid_json_is_reported_as_such(self):
        with self.assertRaises(RecordError):
            record.loads("{not json")

    def test_missing_file_is_reported(self):
        with self.assertRaises(RecordError):
            record.load("/nonexistent/nope.json")

    def test_a_minimal_record_is_accepted(self):
        """Controls and outputs default to empty -- a fresh experiment with
        nothing run yet is a valid record, just not an established claim."""
        exp = record.from_dict({
            "experiment_id": "minimal", "description": "d",
            "backend": "b", "shots": 100,
        })
        self.assertEqual(exp.experiment_id, "minimal")
        self.assertIsNone(exp.controls.ideal_control)


if __name__ == "__main__":
    unittest.main()
