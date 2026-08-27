"""Command-line entry point.

    python -m qem_auditor audit record.json
    python -m qem_auditor audit record.json --json
    python -m qem_auditor validate record.json
    python -m qem_auditor template > my_experiment.json

Exit codes are meant to be used in CI, so a claim cannot quietly regress:

    0  the record audited to CERTIFIED UNDER SCOPE
    1  audited to anything else (the reason is printed)
    2  the record could not be read at all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import record
from .api import Auditor
from .record import RecordError
from .schema import (
    CircuitSpec,
    Controls,
    Experiment,
    NoiseSpec,
    Outputs,
    Replicate,
    ReplicateKind,
    UncertaintyCoverage,
)
from .verdict import Verdict

EXIT_OK = 0
EXIT_NOT_CERTIFIED = 1
EXIT_BAD_RECORD = 2


def _template() -> Experiment:
    """A skeleton record with every field a reviewer would ask about,
    filled with values that are honest for a fresh experiment: controls
    unrun (None) rather than optimistically True."""
    return Experiment(
        experiment_id="my_experiment",
        claim="State exactly what is being claimed, in one sentence.",
        description="What was run, on what, with what mitigation.",
        backend="backend_name",
        shots=20_000,
        circuit=CircuitSpec(circuit_id="my_circuit", native_gate_set="",
                            n_qubits=None),
        noise=NoiseSpec(noise_model="", calibration_source=""),
        controls=Controls(),  # every control None: not yet run
        outputs=Outputs(
            raw_error_kcal=None,
            mitigated_error_kcal=None,
            replicates=[Replicate(0.0, ReplicateKind.INDEPENDENT_SUBMISSION, "draw_0")],
            uncertainty=UncertaintyCoverage(),
        ),
        notes="",
    )


def _cmd_audit(args) -> int:
    try:
        exp = record.load(args.path)
    except RecordError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_BAD_RECORD

    result = Auditor().audit(exp)

    if args.json:
        print(json.dumps({
            "experiment_id": exp.experiment_id,
            "verdict": result.verdict.name,
            "verdict_text": result.verdict.value,
            "licence": result.claim.licence,
            "gates": [{"name": g.name, "passed": g.passed, "reason": g.reason}
                      for g in result.report.gate_results],
            "integrity_violations": result.report.integrity_violations,
            "failure_modes": [{"mode": d.mode.name, "confidence": d.confidence,
                               "evidence": d.evidence, "remedy": d.remedy}
                              for d in result.analysis.diagnoses],
            "not_established": result.claim.not_established,
            "next_experiment": (
                {"description": result.next_experiment.description,
                 "cost_usd": result.next_experiment.cost_usd}
                if result.next_experiment else None),
        }, indent=2))
    elif args.html:
        from .report import render_console, render_html

        print(render_console(exp))
        Path(args.html).write_text(render_html(exp))
        print(f"\nwrote {args.html}")
    else:
        from .report import render_console

        print(render_console(exp))

    return EXIT_OK if result.verdict is Verdict.CERTIFIED_UNDER_SCOPE else EXIT_NOT_CERTIFIED


def _cmd_validate(args) -> int:
    try:
        exp = record.load(args.path)
    except RecordError as e:
        print(f"invalid: {e}", file=sys.stderr)
        return EXIT_BAD_RECORD
    from .integrity import integrity_violations

    violations = integrity_violations(exp)
    if violations:
        print(f"{exp.experiment_id}: {len(violations)} integrity violation(s)",
              file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return EXIT_BAD_RECORD
    print(f"{exp.experiment_id}: record is well-formed and internally consistent")
    return EXIT_OK


def _cmd_investigate(args) -> int:
    """Hand it a record; it audits, attacks, and decides when to stop."""
    try:
        exp = record.load(args.path)
    except RecordError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_BAD_RECORD
    from .agent import AuditAgent
    from .llm import LLMError, provider_from_env
    from .report import render_console, render_html

    try:
        provider = provider_from_env()
    except LLMError as e:
        print(f"note: {e}", file=sys.stderr)
        provider = None

    adapter = None
    if args.qiskit:
        try:
            from .adapters.qiskit_adapter import QiskitAdapter

            adapter = QiskitAdapter()
        except Exception as e:  # qiskit missing or unusable
            print(f"note: qiskit adapter unavailable ({e}); attacks needing it "
                  f"will be reported as not run", file=sys.stderr)

    agent = AuditAgent(adapter=adapter, provider=provider,
                       max_rounds=args.max_rounds, budget_usd=args.budget)
    investigation = agent.investigate(exp)
    investigation.print_investigation()
    print()
    print(render_console(exp, investigation))

    if args.html:
        Path(args.html).write_text(render_html(exp, investigation))
        print(f"\nwrote {args.html}")
    from .verdict import Verdict as _V

    return EXIT_OK if investigation.verdict is _V.CERTIFIED_UNDER_SCOPE \
        else EXIT_NOT_CERTIFIED


def _cmd_attack(args) -> int:
    """What would falsify this claim, and what has it not been subjected to?"""
    try:
        exp = record.load(args.path)
    except RecordError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_BAD_RECORD
    plan = Auditor().attack(exp)
    if args.json:
        print(json.dumps({
            "experiment_id": exp.experiment_id,
            "attacks": [{
                "id": a.attack_id,
                "transformation": a.transformation,
                "targets": a.targets.name,
                "description": a.description,
                "measures": a.prediction.statistic,
                "if_genuine": a.prediction.if_genuine,
                "if_artifact": a.prediction.if_artifact,
                "discrimination": a.discrimination,
                "executable": a.executable,
                "cost_usd": a.cost_usd,
            } for a in plan.attacks],
            "skipped": [{"transformation": n, "why": w} for n, w in plan.skipped],
        }, indent=2))
    else:
        plan.print_plan()
    # Untested attack surface is not a pass.
    return EXIT_OK if not plan.attacks else EXIT_NOT_CERTIFIED


def _cmd_blind(args) -> int:
    """Audit without seeing the outcome, then reveal and score."""
    try:
        exp = record.load(args.path)
    except RecordError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_BAD_RECORD
    from .blind import BlindChallenge, auto_decide

    challenge = BlindChallenge(exp)
    decision = auto_decide(challenge.blinded)
    challenge.decide(decision)
    print(f"blind decision: {decision.reasoning}")
    print("evidence it says is still required:")
    for item in decision.required_evidence:
        print(f"  - {item}")
    result = challenge.reveal()
    print()
    print(result.describe())
    return EXIT_OK if result.correct else EXIT_NOT_CERTIFIED


def _cmd_template(args) -> int:
    print(record.dumps(_template()))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qem-auditor",
        description="Audit a quantum error-mitigation claim against its evidence.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_audit = sub.add_parser("audit", help="audit an experiment record")
    p_audit.add_argument("path", help="path to a JSON experiment record")
    p_audit.add_argument("--json", action="store_true",
                         help="emit machine-readable output")
    p_audit.add_argument("--html", metavar="PATH",
                         help="also write a self-contained HTML report")
    p_audit.set_defaults(func=_cmd_audit)

    p_validate = sub.add_parser(
        "validate", help="check a record is readable and self-consistent, without auditing")
    p_validate.add_argument("path")
    p_validate.set_defaults(func=_cmd_validate)

    p_attack = sub.add_parser(
        "attack", help="propose falsification experiments for a claim")
    p_attack.add_argument("path")
    p_attack.add_argument("--json", action="store_true")
    p_attack.set_defaults(func=_cmd_attack)

    p_inv = sub.add_parser(
        "investigate",
        help="run the audit loop autonomously: audit, attack, execute, repeat")
    p_inv.add_argument("path")
    p_inv.add_argument("--max-rounds", type=int, default=4)
    p_inv.add_argument("--budget", type=float, default=None,
                       help="spending cap in USD for executable attacks")
    p_inv.add_argument("--qiskit", action="store_true",
                       help="use the Qiskit adapter to execute attacks")
    p_inv.add_argument("--html", metavar="PATH",
                       help="also write a self-contained HTML report")
    p_inv.set_defaults(func=_cmd_investigate)

    p_blind = sub.add_parser(
        "blind", help="audit the record with its outcome hidden, then reveal")
    p_blind.add_argument("path")
    p_blind.set_defaults(func=_cmd_blind)

    p_template = sub.add_parser("template", help="print a blank record to fill in")
    p_template.set_defaults(func=_cmd_template)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
