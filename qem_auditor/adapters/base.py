"""What an adapter is, independent of any SDK.

An adapter's whole job is to convert a control from something asserted
into something executed. It reports what it found, including that it could
not tell -- an adapter that returns "passed" when it could not run the
check would be worse than no adapter at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


class MeasurementError(RuntimeError):
    """The control could not be executed. Never silently a failure: a
    control that could not run is not a control that failed."""


@dataclass
class ControlMeasurement:
    control: str
    """The Controls field this measures, e.g. 'unitary_equivalence'."""

    passed: Optional[bool]
    """True/False from actual execution, or None if it could not be judged."""

    detail: str
    """What was executed and what came back, specifically enough to check."""

    evidence: dict = None  # type: ignore[assignment]
    """Raw numbers behind the verdict, for anyone who wants to recompute it."""

    def __post_init__(self) -> None:
        if self.evidence is None:
            self.evidence = {}


@runtime_checkable
class BackendAdapter(Protocol):
    """Anything that can execute controls against a claimant's artifacts."""

    name: str

    def measure(self, *args, **kwargs) -> list[ControlMeasurement]:
        ...
