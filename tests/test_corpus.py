"""The corpus: what a run leaves behind when nobody knows the answer.

The ledger's `Observation` needs the truth to compute an error. The
person this project is most for ran something on hardware precisely
because nobody knows what it should give. So an encounter records what a
method DID and never whether it helped -- and these tests exist mostly to
keep that distinction from eroding.
"""
import json
import tempfile
import unittest
from pathlib import Path

from qem_auditor.corpus import (MINIMUM_ENCOUNTERS, Corpus, Encounter,
                                MethodEffect, encounter_from_report)
from qem_auditor.results import analyse


def encounter(shift=0.01, n_qubits=2, shots=8000, amplification=1.15,
              unmitigated=-0.89, method="REM (tensored)"):
    return Encounter(
        n_qubits=n_qubits, settings=1, shots=shots, floor=0.004,
        unmitigated=unmitigated,
        effects=(MethodEffect(method=method, estimate=unmitigated + shift,
                              shift=shift, amplification=amplification),))


class RecordingTest(unittest.TestCase):

    def test_the_same_run_recorded_twice_counts_once(self):
        """Otherwise re-running the same analysis makes a finding look
        twice as well supported as it is."""
        corpus = Corpus()
        self.assertTrue(corpus.record(encounter()))
        self.assertFalse(corpus.record(encounter()))
        self.assertEqual(len(corpus), 1)

    def test_a_run_with_no_shots_is_refused(self):
        with self.assertRaises(ValueError):
            Encounter(n_qubits=2, settings=1, shots=0, floor=0.1, unmitigated=-1.0)

    def test_no_counts_or_bitstrings_are_kept(self):
        """A corpus that stored the raw data would be a copy of everyone's
        experiments. It has to be safe to keep."""
        stored = json.loads(Corpus.from_json(
            Corpus.to_json(_corpus_with(encounter()))).to_json())
        text = json.dumps(stored)
        for leak in ("counts", "bitstring", "00", "11"):
            self.assertNotIn(f'"{leak}"', text)
        self.assertEqual(set(stored[0]) & {"counts", "measurements"}, set())


def _corpus_with(*entries):
    corpus = Corpus()
    for entry in entries:
        corpus.record(entry)
    return corpus


class ComparabilityTest(unittest.TestCase):

    def test_runs_of_similar_width_and_shots_inform_each_other(self):
        base = encounter(n_qubits=4, shots=8000)
        self.assertTrue(base.resembles(encounter(n_qubits=5, shots=20_000)))

    def test_a_very_different_width_does_not(self):
        self.assertFalse(encounter(n_qubits=2).resembles(encounter(n_qubits=18)))

    def test_a_very_different_shot_count_does_not(self):
        self.assertFalse(
            encounter(shots=1000).resembles(encounter(shots=500_000)))


class ExpectationTest(unittest.TestCase):

    def test_it_refuses_to_summarise_from_too_few(self):
        corpus = _corpus_with(*[encounter(shift=0.01 * i) for i in range(1, 3)])
        expectation = corpus.expectation("REM (tensored)", encounter(shift=0.5))
        self.assertFalse(expectation.worth_relying_on)
        self.assertIn("too few to summarise", expectation.describe())
        self.assertIsNone(expectation.surprise(0.5))

    def test_enough_runs_produce_a_typical_shift_and_cost(self):
        corpus = _corpus_with(*[encounter(shift=0.01 + 0.001 * i)
                                for i in range(MINIMUM_ENCOUNTERS + 2)])
        expectation = corpus.expectation("REM (tensored)", encounter(shift=0.99))
        self.assertTrue(expectation.worth_relying_on)
        self.assertAlmostEqual(expectation.typical_shift, 0.013, places=3)
        self.assertAlmostEqual(expectation.typical_amplification, 1.15)

    def test_an_unseen_method_reports_nothing_rather_than_guessing(self):
        corpus = _corpus_with(*[encounter(shift=0.01 * i) for i in range(1, 8)])
        expectation = corpus.expectation("CDR", encounter())
        self.assertEqual(expectation.n, 0)
        self.assertIn("no comparable runs", expectation.describe())


class AnomalyTest(unittest.TestCase):
    """The point of the whole module. No ground truth is available, so it
    cannot say a method was wrong -- only that this run is unlike the
    others, which is exactly what one run can never tell you."""

    def setUp(self):
        self.corpus = _corpus_with(*[encounter(shift=0.009 + 0.0005 * i)
                                     for i in range(8)])

    def test_a_typical_run_draws_no_comment(self):
        report = self.corpus.report_on(encounter(shift=0.0105))
        self.assertNotIn("Worth a look", report)

    def test_a_run_that_shifts_far_more_is_flagged(self):
        report = self.corpus.report_on(encounter(shift=0.045))
        self.assertIn("Worth a look", report)
        self.assertIn("the usual", report)

    def test_a_run_that_shifts_far_less_is_also_flagged(self):
        """A method that suddenly stops doing anything is as interesting
        as one that overreacts."""
        self.assertIn("Worth a look", self.corpus.report_on(encounter(shift=0.001)))

    def test_the_flag_never_says_the_run_is_wrong(self):
        report = self.corpus.report_on(encounter(shift=0.045))
        self.assertIn("not wrong", report)
        for forbidden in ("incorrect", "invalid", "failed", "better", "worse"):
            self.assertNotIn(forbidden, report)

    def test_an_empty_corpus_says_so_plainly(self):
        report = Corpus().report_on(encounter())
        self.assertIn("nothing to compare against yet", report)


class FromReportTest(unittest.TestCase):

    def test_an_analysis_becomes_an_encounter(self):
        report = analyse({"ZZ": {"00": 7000, "11": 500, "01": 300, "10": 200}},
                         [("ZZ", 0.4), ("II", -1.05)],
                         calibration={"prepared_0": {"00": 8600, "01": 600,
                                                     "10": 700, "11": 100},
                                      "prepared_1": {"11": 8500, "10": 700,
                                                     "01": 700, "00": 100}})
        entry = encounter_from_report(report, n_qubits=2, device="lab")
        self.assertEqual(entry.shots, 8000)
        self.assertEqual(entry.device, "lab")
        methods = {effect.method for effect in entry.effects}
        self.assertIn("REM (tensored)", methods)
        self.assertNotIn("unmitigated", methods,
                         "unmitigated is the baseline, not an effect")
        rem = entry.effect_of("REM (tensored)")
        self.assertGreater(rem.amplification, 1.0,
                           "readout correction amplifies shot noise")


class PersistenceTest(unittest.TestCase):

    def test_a_corpus_survives_a_round_trip(self):
        corpus = _corpus_with(encounter(shift=0.01), encounter(shift=0.02))
        restored = Corpus.from_json(corpus.to_json())
        self.assertEqual(len(restored), 2)
        self.assertEqual(restored.encounters[0].digest,
                         corpus.encounters[0].digest)

    def test_a_missing_file_is_an_empty_corpus_not_a_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(len(Corpus.load(Path(directory) / "absent.json")), 0)

    def test_a_corrupt_file_starts_fresh_rather_than_crashing_an_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            path.write_text("{not json at all")
            self.assertEqual(len(Corpus.load(path)), 0)

    def test_the_store_keeps_it_apart_from_the_ledger(self):
        """Mixing them would let truth-free data masquerade as a measured
        outcome."""
        from qem_auditor.store import Store
        store = Store.ephemeral()
        self.assertTrue(store.record_encounter(encounter()))
        self.assertEqual(len(store.corpus), 1)
        self.assertEqual(len(store.ledger), 0)
