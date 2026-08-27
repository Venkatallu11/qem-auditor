"""Core experiment record schema.

An Experiment is the unit of evidence the auditor reasons over. It does
NOT store a verdict -- verdicts are computed by gates.py/verdict.py from
the record's controls and outputs, never asserted by whoever creates the
record. That separation is the whole point: a claim's author should not
be the one who gets to say whether it passed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Controls:
    """Each field is None (not yet run), True (passed), or False (failed).
    'Passed' for an adversarial/negative control means it behaved as a
    genuine effect should: the perturbation destroyed the result."""

    ideal_control: Optional[bool] = None
    target_leakage_check: Optional[bool] = None
    adversarial_check: Optional[bool] = None
    reproducibility_checked: bool = False


@dataclass
class Outputs:
    raw_error_kcal: Optional[float] = None
    mitigated_error_kcal: Optional[float] = None
    replicate_errors_kcal: list[float] = field(default_factory=list)
    q95_kcal: Optional[float] = None
    n_replicates_target: int = 8  # this project's own established replication convention


@dataclass
class Experiment:
    experiment_id: str
    description: str
    backend: str
    shots: int
    controls: Controls
    outputs: Outputs
    real_hardware_full_validation: bool = False
    notes: str = ""
