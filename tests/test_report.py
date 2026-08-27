"""The report: does it give support and limits equal weight?"""
import re
import unittest

from qem_auditor import Verdict, audit
from qem_auditor.report import VERDICT_TONE, render_console, render_html

from .helpers import make_experiment


class ConsoleTest(unittest.TestCase):
    def test_both_sections_always_render(self):
        text = render_console(make_experiment(replicate_errors_kcal=[0.1] * 4))
        self.assertIn("ESTABLISHED", text)
        self.assertIn("NOT ESTABLISHED", text)

    def test_the_licence_sits_with_the_verdict(self):
        text = render_console(make_experiment(ideal_control=False))
        verdict_line = next(i for i, l in enumerate(text.splitlines())
                            if l.startswith("VERDICT"))
        licence_line = next(i for i, l in enumerate(text.splitlines())
                            if l.startswith("LICENCE"))
        self.assertEqual(licence_line, verdict_line + 1)

    def test_untested_and_failed_are_both_counted_as_not_established(self):
        exp = make_experiment(adversarial_check=None, ideal_control=False)
        text = render_console(exp)
        header = re.search(r"NOT ESTABLISHED \((\d+)\)", text)
        report = audit(exp)
        expected = sum(1 for g in report.gate_results if g.passed is not True)
        self.assertEqual(int(header.group(1)), expected)

    def test_no_congratulatory_language(self):
        """Nothing that reads as praise for surviving."""
        for exp in (make_experiment(), make_experiment(ideal_control=False)):
            text = render_console(exp).lower()
            for banned in ("great", "excellent", "success!", "well done", "looks good"):
                self.assertNotIn(banned, text)

    def test_every_verdict_has_a_tone(self):
        for verdict in Verdict:
            self.assertIn(verdict, VERDICT_TONE)
            tone, colour = VERDICT_TONE[verdict]
            self.assertTrue(tone.strip())
            self.assertTrue(colour.startswith("#"))


class HtmlTest(unittest.TestCase):
    def test_it_is_self_contained(self):
        """No external requests: openable from a filesystem, attachable."""
        page = render_html(make_experiment())
        for external in ("http://", "src=", "<script"):
            self.assertNotIn(external, page.replace("https://", ""))

    def test_it_escapes_content(self):
        exp = make_experiment()
        exp.claim = '<script>alert("xss")</script>'
        page = render_html(exp)
        self.assertNotIn("<script>alert", page)
        self.assertIn("&lt;script&gt;", page)

    def test_both_columns_render_with_counts(self):
        page = render_html(make_experiment(replicate_errors_kcal=[0.1] * 4))
        self.assertIn("Established (", page)
        self.assertIn("Not established (", page)

    def test_it_adapts_to_dark_mode(self):
        self.assertIn("prefers-color-scheme:dark", render_html(make_experiment()))

    def test_the_footer_states_who_decides(self):
        page = render_html(make_experiment())
        self.assertIn("never asserted by a", page)

    def test_every_benchmark_renders(self):
        import run_benchmarks

        for module in run_benchmarks.BENCHMARKS:
            with self.subTest(case=module.EXPERIMENT.experiment_id):
                page = render_html(module.EXPERIMENT)
                self.assertIn("<!doctype html>", page)
                self.assertTrue(page.strip().endswith("</html>"))


class InvestigationRenderingTest(unittest.TestCase):
    def test_an_investigation_renders_in_both(self):
        from qem_auditor.agent import AuditAgent

        exp = make_experiment(ideal_control=False)
        investigation = AuditAgent().investigate(exp)
        self.assertIn("INVESTIGATION", render_console(exp, investigation))
        self.assertIn("Investigation", render_html(exp, investigation))

    def test_the_stopping_reason_is_always_shown(self):
        from qem_auditor.agent import AuditAgent

        exp = make_experiment(ideal_control=False)
        investigation = AuditAgent().investigate(exp)
        self.assertIn(investigation.stopped_because[:30],
                      render_console(exp, investigation))


if __name__ == "__main__":
    unittest.main()
