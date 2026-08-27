"""The LLM layer: can a model get structure in, and is it prevented from
grading anything?
"""
import json
import unittest
from dataclasses import dataclass

from qem_auditor.llm import (
    FORBIDDEN_KEYS,
    LLMError,
    NullProvider,
    OpenAICompatibleProvider,
    ask,
    extract_json,
    provider_from_env,
    sanitize_proposal,
)


@dataclass
class StubProvider:
    reply: str
    name: str = "stub"

    def complete(self, system, user, max_tokens=2048):
        return self.reply


class ExtractionTest(unittest.TestCase):
    """Models wrap JSON in prose and fences however firmly you ask them not to."""

    def test_bare_json(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_fenced_json(self):
        self.assertEqual(extract_json('sure!\n```json\n{"a": 1}\n```\nhope that helps'),
                         {"a": 1})

    def test_fenced_without_language(self):
        self.assertEqual(extract_json('```\n{"a": 1}\n```'), {"a": 1})

    def test_json_embedded_in_prose(self):
        self.assertEqual(extract_json('Here it is: {"a": 1} — let me know'), {"a": 1})

    def test_braces_inside_strings_do_not_confuse_it(self):
        self.assertEqual(extract_json('x {"s": "}{"} y'), {"s": "}{"})

    def test_escaped_quotes_inside_strings(self):
        self.assertEqual(extract_json(r'{"s": "a \" }"}'), {"s": 'a " }'})

    def test_arrays(self):
        self.assertEqual(extract_json("result: [1, 2, 3]"), [1, 2, 3])

    def test_empty_response_is_an_error(self):
        with self.assertRaises(LLMError):
            extract_json("   ")

    def test_unparseable_response_is_an_error(self):
        with self.assertRaises(LLMError):
            extract_json("I'd rather not answer that.")


class SanitizationTest(unittest.TestCase):
    """The rule made mechanical: a model may not mark its own homework."""

    def test_verdict_is_stripped(self):
        self.assertEqual(sanitize_proposal({"a": 1, "verdict": "PASS"}), {"a": 1})

    def test_control_assertions_are_stripped(self):
        cleaned = sanitize_proposal({"ideal_control": True, "adversarial_check": True,
                                     "description": "keep me"})
        self.assertEqual(cleaned, {"description": "keep me"})

    def test_stripping_is_recursive(self):
        cleaned = sanitize_proposal({"outer": {"inner": {"passed": True, "keep": 1}}})
        self.assertEqual(cleaned, {"outer": {"inner": {"keep": 1}}})

    def test_stripping_reaches_into_lists(self):
        cleaned = sanitize_proposal({"items": [{"certified": True, "id": "x"}]})
        self.assertEqual(cleaned, {"items": [{"id": "x"}]})

    def test_the_legitimate_part_of_a_proposal_survives(self):
        """Discarding a whole proposal over one bad field would be worse."""
        cleaned = sanitize_proposal({"description": "a real attack", "verdict": "PASS"})
        self.assertIn("description", cleaned)

    def test_case_insensitive(self):
        self.assertEqual(sanitize_proposal({"VERDICT": "PASS", "a": 1}), {"a": 1})

    def test_every_control_field_is_forbidden(self):
        for field in ("ideal_control", "target_leakage_check", "adversarial_check",
                      "determinism_check", "reproducibility_checked"):
            self.assertIn(field, FORBIDDEN_KEYS)


class AskTest(unittest.TestCase):
    def test_reports_which_forbidden_keys_the_model_tried(self):
        proposal = ask(StubProvider('{"a": 1, "verdict": "PASS", "passed": true}'),
                       "s", "u", kind="test")
        self.assertEqual(proposal.removed_keys, ["passed", "verdict"])
        self.assertEqual(proposal.payload, {"a": 1})

    def test_clean_output_reports_nothing_removed(self):
        proposal = ask(StubProvider('{"a": 1}'), "s", "u", kind="test")
        self.assertEqual(proposal.removed_keys, [])


class ProviderTest(unittest.TestCase):
    def test_null_provider_explains_how_to_configure_one(self):
        with self.assertRaises(LLMError) as ctx:
            NullProvider().complete("s", "u")
        self.assertIn("QEM_LLM_PROVIDER", str(ctx.exception))

    def test_env_defaults_to_null_so_the_auditor_never_needs_a_model(self):
        import os

        saved = os.environ.pop("QEM_LLM_PROVIDER", None)
        try:
            self.assertIsInstance(provider_from_env(), NullProvider)
        finally:
            if saved is not None:
                os.environ["QEM_LLM_PROVIDER"] = saved

    def test_openai_compatible_requires_a_model_name(self):
        import os

        saved = dict(os.environ)
        try:
            os.environ["QEM_LLM_PROVIDER"] = "ollama"
            os.environ.pop("QEM_LLM_MODEL", None)
            with self.assertRaises(LLMError):
                provider_from_env()
        finally:
            os.environ.clear()
            os.environ.update(saved)

    def test_unknown_provider_is_rejected_by_name(self):
        import os

        saved = dict(os.environ)
        try:
            os.environ["QEM_LLM_PROVIDER"] = "telepathy"
            with self.assertRaises(LLMError) as ctx:
                provider_from_env()
            self.assertIn("telepathy", str(ctx.exception))
        finally:
            os.environ.clear()
            os.environ.update(saved)

    def test_a_local_provider_needs_no_key(self):
        p = OpenAICompatibleProvider(base_url="http://localhost:11434/v1",
                                     model="llama3.1")
        self.assertIsNone(p.api_key)


if __name__ == "__main__":
    unittest.main()
