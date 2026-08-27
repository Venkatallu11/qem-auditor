"""Reading and writing experiment records as JSON.

The point of a file format is that the auditor stops being a Python
library only its author can drive. A record is a plain document a
researcher writes, a pipeline emits, or a reviewer reads -- and it is the
same object whether it came from a file or from code.

JSON rather than YAML deliberately: it is in the standard library, so the
core keeps its zero-dependency property, and there is exactly one way to
write any given record.

Unknown fields are an error, not a warning. A typo in a control name is
the difference between "this control passed" and "this control was never
run", and those must never be silently interchangeable.
"""
from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, get_args, get_origin

from .schema import (
    CircuitSpec,
    ClaimType,
    Controls,
    Experiment,
    FailureMode,
    NoiseSpec,
    Outputs,
    Provenance,
    Replicate,
    ReplicateKind,
    TranspilationStatus,
    UncertaintyCoverage,
)

FORMAT_VERSION = 1

_ENUMS = {
    "claim_type": ClaimType,
    "transpilation_status": TranspilationStatus,
    "kind": ReplicateKind,
}


class RecordError(ValueError):
    """A record that cannot be read as written."""


def _enum_to_json(value: Enum) -> str:
    return value.name


def to_dict(exp: Experiment) -> dict[str, Any]:
    """Experiment -> plain JSON-safe dict."""

    def convert(value: Any) -> Any:
        if isinstance(value, Enum):
            return _enum_to_json(value)
        if is_dataclass(value):
            return {f.name: convert(getattr(value, f.name)) for f in fields(value)}
        if isinstance(value, list):
            return [convert(v) for v in value]
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        return value

    return {"format_version": FORMAT_VERSION,
            **{f.name: convert(getattr(exp, f.name)) for f in fields(Experiment)}}


def _build(cls, data: Any, path: str):
    """Recursively builds a dataclass from a dict, rejecting unknown keys."""
    if data is None:
        return None
    if not isinstance(data, dict):
        raise RecordError(f"{path}: expected an object, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise RecordError(
            f"{path}: unknown field(s) {sorted(unknown)}. Known fields are "
            f"{sorted(known)}. A misspelled control reads as 'never run', so this "
            f"is rejected rather than ignored."
        )
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        field_path = f"{path}.{name}"
        annotation = known[name].type
        if name in _ENUMS:
            kwargs[name] = _parse_enum(_ENUMS[name], value, field_path)
        elif name == "provenance":
            kwargs[name] = {k: _parse_enum(Provenance, v, f"{field_path}.{k}")
                            for k, v in (value or {}).items()}
        elif name == "suspected_failure_modes":
            kwargs[name] = [_parse_enum(FailureMode, v, field_path) for v in (value or [])]
        elif name == "replicates":
            kwargs[name] = [_build(Replicate, v, f"{field_path}[{i}]")
                            for i, v in enumerate(value or [])]
        elif name == "controls":
            kwargs[name] = _build(Controls, value, field_path)
        elif name == "outputs":
            kwargs[name] = _build(Outputs, value, field_path)
        elif name == "circuit":
            kwargs[name] = _build(CircuitSpec, value, field_path)
        elif name == "noise":
            kwargs[name] = _build(NoiseSpec, value, field_path)
        elif name == "uncertainty":
            kwargs[name] = _build(UncertaintyCoverage, value, field_path)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _parse_enum(enum_cls, value, path: str):
    if value is None:
        return None
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls[str(value)]
    except KeyError:
        raise RecordError(
            f"{path}: {value!r} is not a valid {enum_cls.__name__}. "
            f"Valid values: {sorted(m.name for m in enum_cls)}"
        ) from None


def from_dict(data: dict[str, Any]) -> Experiment:
    """Plain dict -> Experiment, with every field validated."""
    if not isinstance(data, dict):
        raise RecordError(f"a record must be a JSON object, got {type(data).__name__}")
    data = dict(data)
    version = data.pop("format_version", FORMAT_VERSION)
    if version != FORMAT_VERSION:
        raise RecordError(
            f"record format_version {version} is not supported by this auditor "
            f"(expected {FORMAT_VERSION})"
        )
    missing = {"experiment_id", "description", "backend", "shots"} - set(data)
    if missing:
        raise RecordError(f"record is missing required field(s): {sorted(missing)}")
    data.setdefault("controls", {})
    data.setdefault("outputs", {})
    return _build(Experiment, data, "record")


def dumps(exp: Experiment, indent: int = 2) -> str:
    return json.dumps(to_dict(exp), indent=indent)


def loads(text: str) -> Experiment:
    try:
        return from_dict(json.loads(text))
    except json.JSONDecodeError as e:
        raise RecordError(f"not valid JSON: {e}") from None


def save(exp: Experiment, path: str | Path) -> None:
    Path(path).write_text(dumps(exp) + "\n")


def load(path: str | Path) -> Experiment:
    p = Path(path)
    if not p.exists():
        raise RecordError(f"no such record: {p}")
    return loads(p.read_text())
