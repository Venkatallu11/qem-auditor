from .schema import (
    ClaimType,
    CircuitSpec,
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
from .verdict import AuditReport, Verdict, audit
from .api import AuditResult, Auditor
from . import (active_design, blind, llm, power, provenance, reconstruct,
               estimation, layout, ledger, memory, prescribe, record,
               report, store, trust)
# Two different classes were called Prediction -- the adversary's
# pre-registered "what each outcome would mean", and the predictor's
# "which method should win". Exported under one name, the later import
# silently won and anyone reaching for the other got the wrong class
# with no error at all. Both now say which they are, and `Prediction`
# is deliberately absent from this namespace so the ambiguous name
# cannot resolve to a coin toss. Each module still calls its own class
# Prediction, where there is nothing to be ambiguous with.
from .adversary import (AdversarialScientist, Attack, AttackPlan,
                        Prediction as OutcomePrediction)
from .executor import AttackExecutor, AttackReport
from .reconstruct import FitData, Measurement, Reconstructor
from .agent import AuditAgent, Investigation
from .llm_scientist import LLMAdversary
from .failure_modes import Diagnosis, FailureAnalysis, classify
# `qem_auditor.prescribe(budget)` is the function on purpose; the module
# stays reachable as `from qem_auditor.prescribe import CATALOGUE`, which
# is checked by a test rather than assumed.
from .prescribe import Consult, ErrorBudget, ErrorSource, Prescription, prescribe
from .engine import Consensus, consensus
from .predict import (Predictor, Prediction as MethodPrediction,
                      TrainingCase, Validation, cross_validate)
from .corpus import (Corpus, Encounter, Expectation, MethodEffect,
                     encounter_from_report)
from .devices import (PROFILES, Architecture, DeviceProfile, budget_for,
                      profile)
from .results import (MitigatedEstimate, ResultsReport, ShotNoise, analyse,
                      shot_noise)
from .counts import CountsError, CountsProblem, CountsReport, check_counts
from .calibration import (CalibrationComparison, CalibrationError,
                          ErrorBarAudit, GroupedCalibration, audit_against_truth,
                          audit_groups, audit_residuals, compare_to_model)
from .cost import (RATE_CARDS, Affordability, MethodCost, PricedAdvice,
                   Quote, RateCard, affordable_methods, circuits_for,
                   price_consult)
from .vendor import (IBM_DEFAULT_RESILIENCE_LEVEL, AppliedMitigation,
                     Contradiction, JobRecord, VendorRecordError, cross_check,
                     cross_check_ibm, ibm_estimator_mitigation, ionq_job)
from .control import (ControlError, DistributionShift, IsolatedEffect,
                      build_control, distribution_shift, isolate_effect)
from .layout import DeviceLayout, LayoutAdvice, QubitProperties, advise_layout
from .reversible import (Discrepancy, NotReversible, OracleReport, audit_oracle,
                         evaluate, preflight_gate)
from .ledger import EvidenceLedger, Observation
from .memory import (CaseMemory, CircuitFingerprint, PastCase, Recollection,
                     case_from_audit, fingerprint_from_spec)
from .store import Store

__all__ = [
    "ClaimType", "CircuitSpec", "Controls", "Experiment", "FailureMode",
    "NoiseSpec", "Outputs", "Provenance", "Replicate", "ReplicateKind",
    "TranspilationStatus", "UncertaintyCoverage",
    "AuditReport", "Verdict", "audit",
    "Auditor", "AuditResult", "record",
    "AdversarialScientist", "Attack", "AttackPlan", "Prediction",
    "AttackExecutor", "AttackReport",
    "active_design", "blind", "llm", "power", "provenance", "reconstruct",
    "report", "trust", "prescribe", "layout", "ledger", "memory", "store",
    "estimation",
    "FitData", "Measurement", "Reconstructor",
    "AuditAgent", "Investigation", "LLMAdversary",
    "Diagnosis", "FailureAnalysis", "classify",
    "Consult", "ErrorBudget", "ErrorSource", "Prescription", "prescribe",
    "DeviceLayout", "LayoutAdvice", "QubitProperties", "advise_layout",
    "Discrepancy", "NotReversible", "OracleReport", "audit_oracle", "evaluate",
    "preflight_gate",
    "Consensus", "consensus",
    "Predictor", "Prediction", "TrainingCase", "Validation", "cross_validate",
    "Corpus", "Encounter", "Expectation", "MethodEffect",
    "encounter_from_report",
    "PROFILES", "Architecture", "DeviceProfile", "budget_for", "profile",
    "MitigatedEstimate", "ResultsReport", "ShotNoise", "analyse", "shot_noise",
    "ControlError", "DistributionShift", "IsolatedEffect",
    "build_control", "distribution_shift", "isolate_effect",
    "EvidenceLedger", "Observation",
    "CaseMemory", "CircuitFingerprint", "PastCase", "Recollection",
    "case_from_audit", "fingerprint_from_spec", "Store",
]


#: Read from the installed distribution rather than hardcoded, so the
#: package and its metadata cannot disagree about what version this is.
#: Falls back for a source tree that was never installed.
try:
    from importlib.metadata import version as _version

    __version__ = _version("qem-auditor")
except Exception:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0+unknown"
