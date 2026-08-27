"""Talking to a language model, without letting it grade anything.

The project's rule has been the same since the first gate: an AI may
propose experiments, read literature, and write prose, but it never
decides whether a claim passed. This module is where that rule stops
being a convention and becomes enforced code.

A model here can do exactly three things:

    propose hypotheses      that might still explain the evidence
    propose attacks         that might falsify the claim
    draft prose             explaining what the gates already decided

It cannot set a verdict, mark a control as measured, or assert that a
control passed. Those are not "discouraged" -- `sanitize_proposal`
strips them, and the record-building path never accepts a control value
from a model at all. A proposal that arrives claiming `ideal_control:
true` is a proposal that gets that field removed, because the only thing
entitled to set it is a measurement.

Provider-agnostic on purpose, so the tool works with whatever the user
has:

    OpenAI-compatible   Ollama, LM Studio, llama.cpp, vLLM, Groq,
                        OpenRouter, Together -- anything speaking
                        /v1/chat/completions. Local ones are free and
                        need no key.
    Anthropic           the Messages API.
    Null                no model at all: the deterministic grammar in
                        adversary.py still works, and the auditor
                        degrades to exactly what it did before.

Configuration is by environment so nothing is hardcoded:

    QEM_LLM_PROVIDER    openai | anthropic | null   (default: null)
    QEM_LLM_BASE_URL    e.g. http://localhost:11434/v1  for Ollama
    QEM_LLM_MODEL       e.g. llama3.1, qwen2.5, claude-sonnet-4-5
    QEM_LLM_API_KEY     omitted for local providers

Stdlib only: urllib, so the core keeps its zero-dependency property.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

DEFAULT_TIMEOUT = 120

# Keys a model is never allowed to set, whatever it emits. These are
# outputs of measurement and of the gates; a proposal that carries them
# is proposing to grade itself.
FORBIDDEN_KEYS = frozenset({
    "verdict", "passed", "certified", "provenance", "measured",
    "ideal_control", "target_leakage_check", "adversarial_check",
    "unitary_equivalence", "heldout_check", "extrapolation_in_domain",
    "free_parameter_floor_test", "determinism_check",
    "reproducibility_checked", "real_hardware_full_validation",
})


class LLMError(RuntimeError):
    """The model could not be reached, or returned something unusable."""


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str:
        ...


@dataclass
class NullProvider:
    """No model. Every LLM-optional path falls back to deterministic code.

    Not a degraded mode to apologize for: the nine-transformation grammar,
    the gates, the planner and the power analysis are all deterministic
    and need no model at all. The model adds proposals beyond that
    grammar; it is not load-bearing for any verdict.
    """

    name: str = "null"

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str:
        raise LLMError(
            "no LLM provider configured. Set QEM_LLM_PROVIDER (and "
            "QEM_LLM_BASE_URL / QEM_LLM_MODEL), or use the deterministic grammar, "
            "which needs no model.")


@dataclass
class OpenAICompatibleProvider:
    """Anything speaking /v1/chat/completions.

    Covers the free local options (Ollama, LM Studio, llama.cpp, vLLM)
    and the hosted ones alike -- only base_url, model and key differ.
    """

    base_url: str
    model: str
    api_key: Optional[str] = None
    timeout: int = DEFAULT_TIMEOUT
    name: str = "openai-compatible"

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            # Low but non-zero: proposals benefit from some variety, and a
            # fully greedy model tends to emit the same attack every time.
            "temperature": 0.3,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = _post_json(f"{self.base_url.rstrip('/')}/chat/completions",
                          payload, headers, self.timeout)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"unexpected response shape from {self.base_url}: {e}") from e


@dataclass
class AnthropicProvider:
    model: str
    api_key: Optional[str] = None
    base_url: str = "https://api.anthropic.com/v1"
    timeout: int = DEFAULT_TIMEOUT
    version: str = "2023-06-01"
    name: str = "anthropic"

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str:
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("the Anthropic provider needs an API key "
                           "(QEM_LLM_API_KEY or ANTHROPIC_API_KEY)")
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {"Content-Type": "application/json", "x-api-key": key,
                   "anthropic-version": self.version}
        data = _post_json(f"{self.base_url.rstrip('/')}/messages",
                          payload, headers, self.timeout)
        try:
            return "".join(block.get("text", "") for block in data["content"])
        except (KeyError, TypeError) as e:
            raise LLMError(f"unexpected response shape from Anthropic: {e}") from e


def _post_json(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:400]
        raise LLMError(f"{url} returned HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise LLMError(
            f"could not reach {url}: {e.reason}. For a local model, check it is "
            f"running (e.g. `ollama serve`) and that QEM_LLM_BASE_URL points at it.") from e
    except json.JSONDecodeError as e:
        raise LLMError(f"{url} returned something that is not JSON: {e}") from e


def provider_from_env() -> LLMProvider:
    """Build a provider from the environment, defaulting to none.

    Defaulting to NullProvider is deliberate: the auditor must work with
    no model configured, and a tool that silently requires one would make
    the verdicts depend on an external service.
    """
    kind = os.environ.get("QEM_LLM_PROVIDER", "null").strip().lower()
    model = os.environ.get("QEM_LLM_MODEL", "")
    key = os.environ.get("QEM_LLM_API_KEY") or None
    if kind in ("null", "none", ""):
        return NullProvider()
    if kind in ("openai", "openai-compatible", "ollama", "local", "groq",
                "openrouter", "together", "vllm", "lmstudio"):
        base = os.environ.get("QEM_LLM_BASE_URL", "http://localhost:11434/v1")
        if not model:
            raise LLMError("QEM_LLM_MODEL is required for an OpenAI-compatible provider")
        return OpenAICompatibleProvider(base_url=base, model=model, api_key=key)
    if kind == "anthropic":
        if not model:
            raise LLMError("QEM_LLM_MODEL is required for the Anthropic provider")
        return AnthropicProvider(model=model, api_key=key)
    raise LLMError(f"unknown QEM_LLM_PROVIDER {kind!r}; expected openai, anthropic or null")


# --------------------------------------------------------------------------
# Getting usable structure out of model output
# --------------------------------------------------------------------------

def extract_json(text: str) -> Any:
    """Pull the first JSON value out of a model's reply.

    Models wrap JSON in prose and fences no matter how firmly the prompt
    says not to. Rather than failing on that, find the payload: fenced
    block first, then the first balanced object or array. A model that
    emits nothing parseable is an error, but a model that emits correct
    JSON inside chatter is not.
    """
    if not text or not text.strip():
        raise LLMError("the model returned an empty response")

    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidates = [fence.group(1)] if fence else []
    candidates.append(text)

    for candidate in candidates:
        try:
            return json.loads(candidate.strip())
        except json.JSONDecodeError:
            pass
        for opener, closer in (("{", "}"), ("[", "]")):
            start = candidate.find(opener)
            if start == -1:
                continue
            depth, in_string, escape = 0, False, False
            for i in range(start, len(candidate)):
                ch = candidate[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(candidate[start:i + 1])
                        except json.JSONDecodeError:
                            break
    raise LLMError(f"no parseable JSON in the model's reply: {text[:200]!r}")


def sanitize_proposal(obj: Any) -> Any:
    """Strip anything a model is not entitled to set.

    Recursive, because a forbidden key nested three levels down is still a
    model trying to mark its own homework. Silent removal rather than an
    error: a model that adds `"passed": true` to an otherwise good attack
    has produced a usable attack with one field that does not belong to
    it, and discarding the whole proposal would be worse.
    """
    if isinstance(obj, dict):
        return {k: sanitize_proposal(v) for k, v in obj.items()
                if k.lower() not in FORBIDDEN_KEYS}
    if isinstance(obj, list):
        return [sanitize_proposal(v) for v in obj]
    return obj


@dataclass
class Proposal:
    """What a model suggested, before anything validates it."""

    kind: str
    payload: Any
    raw: str = ""
    provider: str = ""
    removed_keys: list[str] = field(default_factory=list)


def _forbidden_present(obj: Any, found: list[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in FORBIDDEN_KEYS:
                found.append(k)
            _forbidden_present(v, found)
    elif isinstance(obj, list):
        for v in obj:
            _forbidden_present(v, found)


def ask(provider: LLMProvider, system: str, user: str, kind: str,
        max_tokens: int = 2048) -> Proposal:
    """Ask for a structured proposal, and strip what it may not decide."""
    raw = provider.complete(system, user, max_tokens=max_tokens)
    parsed = extract_json(raw)
    removed: list[str] = []
    _forbidden_present(parsed, removed)
    return Proposal(kind=kind, payload=sanitize_proposal(parsed), raw=raw,
                    provider=getattr(provider, "name", "unknown"),
                    removed_keys=sorted(set(removed)))
