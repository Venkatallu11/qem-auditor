"""Evidence bundles: does a digest actually pin the inputs, and does a
mismatch say WHICH input moved?
"""
import json
import tempfile
import unittest
from pathlib import Path

from qem_auditor.provenance import (
    EvidenceBundle,
    ProvenanceError,
    build_bundle,
    environment_fingerprint,
    hash_file,
    hash_json,
    hash_text,
)


def _bundle(**over):
    base = dict(experiment_id="e", claim="c",
                artifact_objects={"counts": {"00": 10, "11": 12}},
                seeds={"bootstrap": 0}, backend_id="sim",
                analysis_version="1.0")
    base.update(over)
    return build_bundle(**base)


class HashingTest(unittest.TestCase):
    def test_key_order_does_not_change_a_json_hash(self):
        """Load-bearing: without canonical ordering, the same counts hash
        differently by insertion order and every bundle is spuriously
        unique."""
        self.assertEqual(hash_json({"a": 1, "b": 2}), hash_json({"b": 2, "a": 1}))

    def test_different_content_hashes_differently(self):
        self.assertNotEqual(hash_json({"a": 1}), hash_json({"a": 2}))

    def test_file_hash_matches_its_content(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.txt"
            p.write_text("hello")
            self.assertEqual(hash_file(p), hash_text("hello"))

    def test_missing_file_is_reported(self):
        with self.assertRaises(ProvenanceError):
            hash_file("/nonexistent/nope")


class DigestTest(unittest.TestCase):
    def test_identical_inputs_give_identical_digests(self):
        self.assertEqual(_bundle().digest, _bundle().digest)

    def test_any_changed_artifact_changes_the_digest(self):
        a = _bundle()
        b = _bundle(artifact_objects={"counts": {"00": 11, "11": 12}})
        self.assertNotEqual(a.digest, b.digest)

    def test_a_changed_seed_changes_the_digest(self):
        self.assertNotEqual(_bundle().digest, _bundle(seeds={"bootstrap": 1}).digest)

    def test_a_changed_analysis_version_changes_the_digest(self):
        self.assertNotEqual(_bundle().digest, _bundle(analysis_version="1.1").digest)


class DiffTest(unittest.TestCase):
    """A bare digest mismatch says something moved. A diff says what."""

    def test_diff_names_the_changed_artifact(self):
        a = _bundle()
        b = _bundle(artifact_objects={"counts": {"00": 11, "11": 12}})
        self.assertIn("artifacts.counts", a.diff(b))

    def test_diff_names_a_changed_scalar_field(self):
        self.assertIn("backend_id", _bundle().diff(_bundle(backend_id="hw")))

    def test_identical_bundles_diff_empty(self):
        self.assertEqual(_bundle().diff(_bundle()), {})

    def test_an_added_artifact_shows_up(self):
        a = _bundle()
        b = _bundle(artifact_objects={"counts": {"00": 10, "11": 12},
                                      "calibration": {"p": 0.1}})
        self.assertIn("artifacts.calibration", a.diff(b))


class ReproducibilityTest(unittest.TestCase):
    def test_an_unset_hash_seed_is_flagged(self):
        """The exact condition that made identical-seed reruns diverge."""
        b = _bundle()
        b.environment = {"pythonhashseed": "unset"}
        ok, problems = b.is_reproducible
        self.assertFalse(ok)
        self.assertTrue(any("PYTHONHASHSEED" in p for p in problems))

    def test_a_dirty_tree_is_flagged(self):
        b = _bundle()
        b.git_commit, b.git_dirty = "abc123", True
        b.environment = {"pythonhashseed": "0"}
        ok, problems = b.is_reproducible
        self.assertFalse(ok)
        self.assertTrue(any("dirty" in p for p in problems))

    def test_a_complete_bundle_is_reproducible(self):
        b = _bundle()
        b.git_commit, b.git_dirty = "abc123", False
        b.environment = {"pythonhashseed": "0"}
        ok, problems = b.is_reproducible
        self.assertTrue(ok, problems)

    def test_no_artifacts_is_flagged(self):
        b = EvidenceBundle("e")
        self.assertFalse(b.is_reproducible[0])


class SerializationTest(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "b.json"
            original = _bundle()
            original.save(p)
            self.assertEqual(EvidenceBundle.load(p).digest, original.digest)

    def test_an_edited_manifest_is_detected_on_load(self):
        """The bundle validates against its own recorded digest."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "b.json"
            _bundle().save(p)
            data = json.loads(p.read_text())
            data["backend_id"] = "tampered"
            p.write_text(json.dumps(data))
            with self.assertRaises(ProvenanceError) as ctx:
                EvidenceBundle.load(p)
            self.assertIn("edited", str(ctx.exception))

    def test_missing_bundle_is_reported(self):
        with self.assertRaises(ProvenanceError):
            EvidenceBundle.load("/nonexistent/b.json")


class BuildTest(unittest.TestCase):
    def test_a_duplicate_artifact_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.txt"
            p.write_text("x")
            with self.assertRaises(ProvenanceError) as ctx:
                build_bundle("e", artifact_paths={"a": p}, artifact_objects={"a": {}})
            self.assertIn("one name, one artifact", str(ctx.exception))

    def test_environment_records_the_hash_seed(self):
        self.assertIn("pythonhashseed", environment_fingerprint())


if __name__ == "__main__":
    unittest.main()
