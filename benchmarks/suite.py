"""The QEM-Trust suite: the six real cases, as scoreable cases.

Separate from `run_benchmarks.py` on purpose. That file pins qem-auditor's
verdicts against themselves and fails on regression. This file states what
each case is KNOWN to deserve independently of what this package says
about it, so that any auditor -- ours, a competitor's, an LLM, a person --
can be scored on the same six records.

Every truth verdict below traces to a disclosed result in
quantum-chemistry-vqe's RESEARCH_LEDGER.md, not to this package's output.
Where the two agree, that is a finding; it is not a definition.

Layering note: `qem_auditor` never imports this module. The scoring
machinery lives in `qem_auditor.trust` and knows nothing about these
cases; the cases live here and depend on the package. A benchmark that
its own subject imports is not a benchmark.
"""
from qem_auditor.trust import Case, CaseProvenance

from . import (
    h4_ancilla_qed,
    h4_compiler_cancellation,
    h4_cross_fitting,
    h4_joint_schmidt_frame,
    h4_one_off_pec,
    h4_zne_blowup,
)

_WHAT_EACH_TESTS = {
    "h4_compiler_cancellation":
        "the mitigated circuit was never the circuit that ran -- the "
        "compiler cancelled the folds. Separates auditors that check what "
        "was EXECUTED from auditors that read what was WRITTEN.",
    "h4_zne_blowup":
        "pure shot noise through the production extrapolator, on a model "
        "with no real noise to correct. Separates auditors that run an "
        "ideal control from auditors that trust an improvement number.",
    "h4_one_off_pec":
        "a single draw with no replication and no uncertainty. Separates "
        "auditors that treat an unrun control as unproven from auditors "
        "that treat it as passed.",
    "h4_cross_fitting":
        "held-out validation performed in the interpolation direction for "
        "a method used in the extrapolation direction. Separates auditors "
        "that check whether the validation tests the USE.",
    "h4_joint_schmidt_frame":
        "a real, large improvement that holds only under the noise model "
        "it was tuned on. Separates auditors that can say 'true under this "
        "model' from auditors with only pass and fail.",
    "h4_ancilla_qed":
        "the project's own best result: sound, replicated, and still not "
        "certifiable under its stated scope. Separates auditors that "
        "withhold certification from good work from auditors that reward "
        "a good number.",
}

_MODULES = [
    h4_compiler_cancellation,
    h4_zne_blowup,
    h4_one_off_pec,
    h4_cross_fitting,
    h4_joint_schmidt_frame,
    h4_ancilla_qed,
]


def _case(module) -> Case:
    case_id = module.EXPERIMENT.experiment_id
    key = module.__name__.rsplit(".", 1)[-1]
    return Case(
        case_id=case_id,
        experiment=module.EXPERIMENT,
        truth=module.EXPECTED_VERDICT,
        what_it_tests=_WHAT_EACH_TESTS[key],
        truth_mode=getattr(module, "EXPECTED_PRIMARY_FAILURE_MODE", None),
        provenance=CaseProvenance.DISCLOSED,
    )


#: The disclosed six, whose truth was settled by the project's own
#: follow-up work rather than by any rule in this package.
CASES = [_case(m) for m in _MODULES]

#: The disclosed six plus the constructed minimal pairs. Scored together
#: the split still shows: `qem_auditor.trust` reports exact-match and
#: credit separately for each provenance, so a tool that has learned the
#: schema cannot hide behind a blended headline.
from .constructed import CASES as CONSTRUCTED_CASES, PAIRS  # noqa: E402

ALL_CASES = CASES + CONSTRUCTED_CASES
