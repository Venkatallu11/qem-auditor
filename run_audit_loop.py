#!/usr/bin/env python3
"""The closed loop, run against this project's real history.

Replays the H4 decision trail through the auditor: audit each real result,
update the hypothesis ledger with what it actually showed, then ask the
planner what to run next given the evidence now in hand.

The interesting output is the last section. The planner is never told
which method won; it sees only the belief state and the audit gaps, and
recommends from those.
"""
import sys

from qem_auditor import audit, classify
from qem_auditor.claim import compile_claim
from qem_auditor.hypothesis import Hypothesis, HypothesisLedger, Observation
from qem_auditor.planner import Recommendation, candidates_from_audit, plan

from benchmarks import h4_ancilla_qed, h4_joint_schmidt_frame, h4_one_off_pec, h4_zne_blowup

# The competing explanations the project actually held, with the weights
# it would plausibly have assigned before any of them was tested.
HYPOTHESES = [
    Hypothesis("H1", "One-qubit noise dominates the H4 forged-energy error.", 0.25),
    Hypothesis("H2", "The all-gate ZNE extrapolator is stable enough to trust.", 0.25),
    Hypothesis("H3", "PEC alone suffices under a calibrated channel model.", 0.25),
    Hypothesis("H4", "Ancilla-parity leakage detection with conditioned PEC "
                     "materially reduces the error.", 0.25),
]

# Each real result, and how likely it was under each hypothesis. Supplied
# explicitly rather than inferred: the arithmetic stays inspectable.
TRAIL = [
    # Not every real experiment needs a full record in the benchmark
    # suite; an observation alone is enough to move the ledger. This one
    # converted an assumption into a measurement and killed it.
    (None, Observation(
        "task28a_1q_noise_spectroscopy", {"H1": 0.02},
        "measured one-qubit noise far too small to dominate the error budget")),
    (h4_zne_blowup, Observation(
        "h4_all_gate_zne_ideal_control", {"H2": 0.001},
        "513x error amplification on a model with zero real noise to correct")),
    (h4_one_off_pec, Observation(
        "h4_calibrated_pec_manifold_one_off", {"H3": 0.30},
        "0.115 kcal/mol from one submission, unreplicated; the same pipeline also "
        "produced 0.317 and 0.438")),
    (h4_joint_schmidt_frame, Observation(
        "h4_joint_schmidt_frame", {"H3": 0.20, "H4": 1.5},
        "shared-frame regularization cuts MSE 88.8%, so per-slot independent fitting "
        "was itself a major error source")),
    (h4_ancilla_qed, Observation(
        "h4_ancilla_qed_conditioned_pec", {"H4": 3.0},
        "4 independent draws in a 0.0105-0.0192 kcal/mol band, every hard gate clean")),
]

REAL_HARDWARE_BUDGET_USD = 170.0


def main() -> int:
    ledger = HypothesisLedger(HYPOTHESES)
    print("=" * 72)
    print("REPLAYING THE H4 DECISION TRAIL")
    print("=" * 72)
    print(f"\nPrior: {len(HYPOTHESES)} competing explanations, "
          f"entropy {ledger.entropy:.3f} bits")

    for module, observation in TRAIL:
        if module is None:
            ledger.update(observation)
            print(f"\n{'-' * 72}\n{observation.experiment_id}\n{'-' * 72}")
            print(f"  (no full record in the suite -- observation only)")
            print(f"  ledger:  entropy {ledger.entropy:.3f} bits, "
                  f"leading {ledger.leading()[0]} at {ledger.leading()[1]:.1%}")
            continue
        exp = module.EXPERIMENT
        report = audit(exp)
        print(f"\n{'-' * 72}\n{exp.experiment_id}\n{'-' * 72}")
        print(f"  verdict: {report.verdict.value}")
        analysis = classify(exp, report)
        if analysis.primary:
            print(f"  why:     {analysis.primary.mode.name} -- {analysis.primary.evidence}")
        ledger.update(observation)
        print(f"  ledger:  entropy {ledger.entropy:.3f} bits, "
              f"leading {ledger.leading()[0]} at {ledger.leading()[1]:.1%}")

    ledger.print_ledger()

    # What should be run next, given everything above?
    print("\n" + "=" * 72)
    print("WHAT TO RUN NEXT")
    print("=" * 72)

    best = h4_ancilla_qed.EXPERIMENT
    report = audit(best)
    gaps = candidates_from_audit(best, report)
    recommendation, proposals, reason = plan(ledger, gaps,
                                             budget_usd=REAL_HARDWARE_BUDGET_USD)
    # Two different questions, deliberately reported separately. Closing a
    # record gap makes the leading result citable; discriminating between
    # hypotheses decides which explanation is true. A free experiment can
    # be worth running for the first reason while gaining nothing on the
    # second, which is exactly the situation here.
    print(f"\n1. To make the leading result citable ({best.experiment_id}):")
    if gaps:
        for c in gaps:
            cost = f"${c.cost_usd:,.2f}" if c.cost_usd > 0 else "free"
            print(f"     [{cost:>9}] {c.description}")
    else:
        print("     no open gaps in the record")

    print(f"\n2. To discriminate between the remaining hypotheses: {recommendation.value}")
    print(f"     {reason}")
    if recommendation is Recommendation.STOP and gaps:
        print("     Note: the gaps above are still worth closing -- they change what the "
              "result\n     can be cited as, not which hypothesis is true.")

    compile_claim(best, report).print_claim()
    return 0


if __name__ == "__main__":
    sys.exit(main())
