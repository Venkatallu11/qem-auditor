"""Every audit makes the next one better, without anyone taking its word.

The properties tested are the ones that stop a growing corpus becoming a
machine that launders its own guesses: duplicates cannot inflate a
sample, small samples say they are small, and disagreement with the
catalogue is surfaced rather than averaged away.
"""
import json
import tempfile
import unittest
from pathlib import Path

from qem_auditor import Provenance
from qem_auditor.ledger import (EvidenceLedger, Observation, budget_similarity,
                                shares_of)
from qem_auditor.prescribe import ErrorBudget, ErrorSource, prescribe

E = ErrorSource
REM = "readout error mitigation (REM)"
ZNE = "zero-noise extrapolation (ZNE)"
CDR = "Clifford data regression (CDR)"


def budget(contributions=None):
    return ErrorBudget(contributions or {E.READOUT: 30.0, E.GATE_STOCHASTIC: 5.0},
                       Provenance.MEASURED)


def observation(name, method=REM, mitigated=6.0, shares=None, b=None):
    return Observation(experiment_id=name, device="kyiv", method=method,
                       budget_shares=shares if shares is not None
                       else shares_of(b or budget()),
                       raw_error=36.0, mitigated_error=mitigated,
                       provenance=Provenance.MEASURED)


def fill(ledger, n, method=REM, mitigated=6.0, b=None):
    for i in range(n):
        ledger.record(observation(f"{method}_{i}", method, mitigated + i * 0.01, b=b))
    return ledger


class ObservationTest(unittest.TestCase):
    def test_a_method_outside_the_catalogue_is_refused(self):
        """An observation about something nothing can prescribe cannot
        inform a prescription."""
        with self.assertRaises(ValueError):
            observation("x", method="wishful thinking")

    def test_shares_that_do_not_sum_to_one_are_refused(self):
        with self.assertRaises(ValueError):
            observation("x", shares={"READOUT": 0.5, "GATE_STOCHASTIC": 0.9})

    def test_a_zero_error_is_refused_because_it_has_no_ratio(self):
        with self.assertRaises(ValueError):
            Observation("x", "kyiv", REM, shares_of(budget()), 36.0, 0.0)

    def test_the_gain_is_the_ratio(self):
        self.assertAlmostEqual(observation("x", mitigated=9.0).gain, 4.0)


class DuplicateTest(unittest.TestCase):
    def test_the_same_run_recorded_twice_is_one_data_point(self):
        ledger = EvidenceLedger()
        self.assertTrue(ledger.record(observation("a")))
        self.assertFalse(ledger.record(observation("a")))
        self.assertEqual(len(ledger), 1)

    def test_duplicates_are_detected_by_content_not_by_name(self):
        """Renaming a run does not make it new evidence."""
        ledger = EvidenceLedger()
        first = observation("run_one")
        second = Observation("run_one", first.device, first.method,
                             first.budget_shares, first.raw_error,
                             first.mitigated_error, first.provenance)
        ledger.record(first)
        self.assertFalse(ledger.record(second))

    def test_a_genuinely_different_outcome_is_new(self):
        ledger = EvidenceLedger()
        ledger.record(observation("a", mitigated=6.0))
        self.assertTrue(ledger.record(observation("b", mitigated=7.0)))


class SimilarityTest(unittest.TestCase):
    def test_identical_budgets_are_identical(self):
        shares = shares_of(budget())
        self.assertAlmostEqual(budget_similarity(shares, shares), 1.0)

    def test_disjoint_budgets_share_nothing(self):
        self.assertAlmostEqual(
            budget_similarity({"READOUT": 1.0}, {"GATE_STOCHASTIC": 1.0}), 0.0)

    def test_how_much_of_the_budget_a_source_owns_matters_not_only_which(self):
        """Cosine similarity would call these near-identical because they
        point the same way. The question is how much, so it is wrong."""
        mostly = {"READOUT": 0.9, "GATE_STOCHASTIC": 0.1}
        barely = {"READOUT": 0.3, "GATE_STOCHASTIC": 0.7}
        self.assertLess(budget_similarity(mostly, barely), 0.5)

    def test_a_dissimilar_budget_is_not_consulted(self):
        ledger = fill(EvidenceLedger(), 6)
        elsewhere = budget({E.GATE_STOCHASTIC: 30.0, E.DECOHERENCE: 5.0})
        self.assertEqual(ledger.evidence_for(REM, elsewhere).n, 0)


class SampleSizeTest(unittest.TestCase):
    def test_a_small_sample_says_it_is_small(self):
        evidence = fill(EvidenceLedger(), 3).evidence_for(REM, budget())
        self.assertEqual(evidence.n, 3)
        self.assertFalse(evidence.supports_a_ranking)
        self.assertIn("too few", evidence.summarise())

    def test_enough_observations_earn_a_ranking(self):
        self.assertTrue(
            fill(EvidenceLedger(), 6).evidence_for(REM, budget()).supports_a_ranking)

    def test_nothing_recorded_reports_nothing_rather_than_a_default(self):
        evidence = EvidenceLedger().evidence_for(REM, budget())
        self.assertEqual(evidence.n, 0)
        self.assertIsNone(evidence.median_gain)
        self.assertIn("nothing recorded", evidence.summarise())

    def test_the_worst_run_is_reported_alongside_the_median(self):
        ledger = EvidenceLedger()
        ledger.record(observation("good", mitigated=2.0))
        ledger.record(observation("bad", mitigated=30.0))
        evidence = ledger.evidence_for(REM, budget())
        self.assertLess(evidence.worst_gain, evidence.median_gain)

    def test_self_reported_observations_are_flagged_as_such(self):
        ledger = EvidenceLedger()
        ledger.record(Observation("claimed", "kyiv", REM, shares_of(budget()),
                                  36.0, 6.0, Provenance.SELF_REPORTED))
        self.assertIn("self-reported",
                      ledger.evidence_for(REM, budget()).summarise())


class ReorderingTest(unittest.TestCase):
    """Observations outrank mechanism only when there are enough of them
    for every method being compared."""

    def test_a_thin_corpus_does_not_reorder_anything(self):
        ledger = fill(EvidenceLedger(), 2, method=REM, mitigated=1.0)
        before = [p.action for p in prescribe(budget()).prescriptions]
        after = [p.action for p in prescribe(budget(), ledger=ledger).prescriptions]
        self.assertEqual(before, after)

    def test_a_full_corpus_reorders_and_says_so(self):
        ledger = EvidenceLedger()
        b = budget()
        fill(ledger, 6, REM, mitigated=2.0, b=b)
        fill(ledger, 6, CDR, mitigated=20.0, b=b)
        fill(ledger, 6, "REM then ZNE", mitigated=12.0, b=b)
        consult = prescribe(b, ledger=ledger)
        self.assertEqual(consult.leading.action, REM)
        self.assertTrue(any("what past audits measured" in c
                            for c in consult.caveats))

    def test_a_partial_corpus_does_not_let_the_best_studied_method_win(self):
        """Reordering on incomplete evidence ranks by how much attention a
        method has had, not by how well it works."""
        ledger = fill(EvidenceLedger(), 6, REM, mitigated=2.0)
        consult = prescribe(budget(), ledger=ledger)
        self.assertFalse(any("what past audits measured" in c
                             for c in consult.caveats))

    def test_recommendations_cite_what_was_seen(self):
        ledger = fill(EvidenceLedger(), 6, REM, mitigated=2.0)
        text = prescribe(budget(), ledger=ledger).format_consult()
        self.assertIn("observations", text)


class ContradictionTest(unittest.TestCase):
    def test_a_method_that_should_work_and_does_not_is_reported(self):
        ledger = fill(EvidenceLedger(), 6, REM, mitigated=35.9)
        found = dict(ledger.contradictions(budget()))
        self.assertIn(REM, found)
        self.assertIn("catalogue says it reaches", found[REM])

    def test_a_method_that_should_not_work_and_does_is_reported(self):
        ledger = fill(EvidenceLedger(), 6, ZNE, mitigated=6.0)
        found = dict(ledger.contradictions(budget()))
        self.assertIn(ZNE, found)
        self.assertIn("missing something", found[ZNE])

    def test_a_contradiction_reaches_the_prescription_as_a_caveat(self):
        ledger = fill(EvidenceLedger(), 6, REM, mitigated=35.9)
        self.assertTrue(any("disagrees with the catalogue" in c
                            for c in prescribe(budget(), ledger=ledger).caveats))

    def test_too_few_observations_raise_no_contradiction(self):
        """Three runs disagreeing with the physics is three runs."""
        self.assertEqual(
            fill(EvidenceLedger(), 3, REM, mitigated=35.9).contradictions(budget()),
            [])


class PersistenceTest(unittest.TestCase):
    def test_a_ledger_survives_a_round_trip(self):
        ledger = fill(EvidenceLedger(), 4)
        restored = EvidenceLedger.from_json(ledger.to_json())
        self.assertEqual(len(restored), len(ledger))
        self.assertEqual(restored.evidence_for(REM, budget()).median_gain,
                         ledger.evidence_for(REM, budget()).median_gain)

    def test_it_is_a_readable_file_not_an_opaque_model(self):
        """A recommender that improves in ways nobody can read is not an
        improvement anybody should accept."""
        entries = json.loads(fill(EvidenceLedger(), 2).to_json())
        self.assertEqual(len(entries), 2)
        self.assertIn("method", entries[0])
        self.assertIn("budget_shares", entries[0])

    def test_loading_a_missing_file_gives_an_empty_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(len(EvidenceLedger.load(Path(tmp) / "nope.json")), 0)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.json"
            fill(EvidenceLedger(), 3).save(path)
            self.assertEqual(len(EvidenceLedger.load(path)), 3)

    def test_reloading_does_not_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.json"
            ledger = fill(EvidenceLedger(), 3)
            ledger.save(path)
            reloaded = EvidenceLedger.load(path)
            for obs in ledger.observations:
                reloaded.record(obs)
            self.assertEqual(len(reloaded), 3)


if __name__ == "__main__":
    unittest.main()
