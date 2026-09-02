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


def _add_store_arguments(parser) -> None:
    parser.add_argument("--store", metavar="DIR",
                        help="where to keep what past audits found "
                             "(default: $QEM_AUDITOR_STORE or ~/.qem-auditor)")
    parser.add_argument("--no-store", action="store_true",
                        help="do not read or write the corpus for this run")


def _open_store(args):
    """The store this invocation should use, or None.

    The CLI accumulates by default and the library does not: a tool that
    forgets everything between invocations is not much of a tool, and a
    library that starts writing to a home directory on import is a
    surprise. Where it is writing is printed the first time it creates
    anything, so it is visible rather than silent.
    """
    from .store import Store

    if getattr(args, "no_store", False):
        return None
    directory = getattr(args, "store", None)
    store = Store.open(directory)
    if not store.directory.exists():
        print(f"note: remembering this audit in {store.directory} "
              f"(--no-store to skip, --store DIR to move it)", file=sys.stderr)
    return store


def _budget_from(args):
    """An error budget from a calibration file, or None.

    None means no advice, which is the honest outcome of not knowing
    where the error comes from. Nothing here guesses a budget in order to
    have something to say.
    """
    path = getattr(args, "calibration", None)
    if not path:
        return None
    from .prescribe import budget_from_calibration

    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise RecordError(f"could not read calibration {path}: {e}") from e
    missing = {"two_qubit_error", "one_qubit_error", "readout_error",
               "two_qubit_gates", "one_qubit_gates", "measured_qubits",
               "shots"} - set(data)
    if missing:
        raise RecordError(
            f"calibration {path} is missing {', '.join(sorted(missing))}. "
            "Every one of these is needed to attribute the error, and a "
            "default for any of them would be a guess wearing a number's "
            "clothes.")
    return budget_from_calibration(
        two_qubit_gates=data["two_qubit_gates"],
        one_qubit_gates=data["one_qubit_gates"],
        measured_qubits=data["measured_qubits"],
        two_qubit_error=data["two_qubit_error"],
        one_qubit_error=data["one_qubit_error"],
        readout_error=data["readout_error"],
        shots=data["shots"],
        circuit_duration_s=data.get("circuit_duration_s"),
        t2_s=data.get("t2_s"))


def _render_guidance(result) -> str:
    """The half of the answer that is not a verdict.

    `report.render_console` re-derives its output from the record and so
    cannot see what the audit found in memory or what it recommends --
    which is how four working capabilities stayed invisible to everyone
    using the command line. This appends them.
    """
    parts = []
    if result.recalled is not None and not result.recalled.is_empty:
        parts.append("\nWHAT THIS REMINDS THE AUDITOR OF")
        parts.append(result.recalled.format_recollection())
    if result.consult is not None:
        parts.append("\n" + result.consult.format_consult())
    else:
        parts.append(
            "\nNO REMEDY OFFERED\n"
            "  No error budget was supplied, and one is not something this can\n"
            "  invent. Pass --calibration with your device's published error\n"
            "  rates and your own gate counts, and the verdict comes back with\n"
            "  what to do about it.")
    return "\n".join(parts)


def _cmd_remember(args) -> int:
    """What the corpus holds, and what it says about one circuit.

    Exists because a corpus that silently steers recommendations and
    cannot be read is the thing this package refuses everywhere else.
    """
    from .memory import fingerprint_from_spec

    store = _open_store(args)
    if store is None:
        print("nothing to show: --no-store was given", file=sys.stderr)
        return 0
    print(store.summarise())

    if not args.circuit:
        if store.memory.cases:
            print("\n  circuits remembered, most recent last:")
            for case in store.memory.cases[-10:]:
                failed = ", ".join(case.failed_gates) or "nothing failed"
                print(f"    {case.experiment_id}: {case.verdict.value} -- {failed}")
        return 0

    try:
        exp = record.load(args.circuit)
    except RecordError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_BAD_RECORD
    print()
    print(store.memory.recall(fingerprint_from_spec(exp.circuit))
          .format_recollection())
    return 0


def _cmd_audit(args) -> int:
    try:
        exp = record.load(args.path)
    except RecordError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_BAD_RECORD

    store = _open_store(args)
    try:
        budget = _budget_from(args)
    except RecordError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_BAD_RECORD

    result = Auditor(store=store).audit(
        exp, budget=budget,
        symmetry_available=getattr(args, "symmetry", False))
    if store is not None:
        store.save()

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
            "prescriptions": ([
                {"action": p.action, "because": p.because,
                 "best_case": p.best_case}
                for p in result.consult.prescriptions]
                if result.consult else None),
            "will_not_help": ([{"method": n, "why": w}
                               for n, w in result.consult.will_not_help]
                              if result.consult else None),
            "recalled": ({"seen_before": [c.experiment_id
                                          for c in result.recalled.seen_before],
                          "similar": [c.experiment_id
                                      for c, _ in result.recalled.resembling],
                          "check_first": [n for n, _, _ in
                                          result.recalled.check_first]}
                         if result.recalled and not result.recalled.is_empty
                         else None),
        }, indent=2))
    elif args.html:
        from .report import render_console, render_html

        print(render_console(exp))
        print(_render_guidance(result))
        Path(args.html).write_text(render_html(exp))
        print(f"\nwrote {args.html}")
    else:
        from .report import render_console

        print(render_console(exp))
        print(_render_guidance(result))

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


def _load_module(path: str):
    """Load a user's Python file so their circuit and pipeline are audited
    as they actually wrote them.

    This executes the file. That is inherent to auditing a real pipeline --
    a mitigation function has to run to be tested -- and it is the user's
    own code, but it is worth stating rather than leaving quiet.
    """
    import importlib.util

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no such file: {p}")
    spec = importlib.util.spec_from_file_location(p.stem, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {p} as a Python module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cmd_check(args) -> int:
    """Audit a circuit directly. The front door."""
    from .templates import CHECK_TEMPLATE

    if args.template:
        print(CHECK_TEMPLATE)
        return EXIT_OK
    if not args.path:
        print("error: give a Python file, or --template for a starting point",
              file=sys.stderr)
        return EXIT_BAD_RECORD
    try:
        module = _load_module(args.path)
    except Exception as e:
        print(f"error: could not load {args.path}: {e}", file=sys.stderr)
        # The first thing a new reader does is follow the quick start, and
        # the template it prints defines its circuit with qiskit -- which
        # the dependency-free core install does not have. "No module named
        # 'qiskit'" is a true error and a useless one at that moment, so
        # it carries the remedy.
        if isinstance(e, ImportError) and "qiskit" in str(e):
            print('       the starting template builds its circuit with qiskit: '
                  'pip install -e ".[adapters]"', file=sys.stderr)
        return EXIT_BAD_RECORD

    circuit = getattr(module, "circuit", None)
    if circuit is None:
        print(f"error: {args.path} defines no `circuit`. Run "
              f"`qem-auditor check --template > my_circuit.py` for a starting point.",
              file=sys.stderr)
        return EXIT_BAD_RECORD

    from .api import Auditor
    from .report import render_console, render_html

    adapter = None
    try:
        from .adapters.qiskit_adapter import QiskitAdapter

        adapter = QiskitAdapter()
    except Exception as e:
        print(f"note: qiskit unavailable ({e}); nothing can be executed, so every "
              f"control is reported as not run", file=sys.stderr)

    result = Auditor(adapter=adapter).verify(
        circuit=circuit,
        observable=getattr(module, "observable", None),
        mitigator=getattr(module, "mitigator", None),
        submitted_circuit=getattr(module, "submitted_circuit", None),
        amplified_circuit=getattr(module, "amplified_circuit", None),
        claim=getattr(module, "claim", ""),
        backend=getattr(module, "backend", "unspecified"),
        shots=getattr(module, "shots", 20_000),
        replicate_errors=getattr(module, "replicate_errors", ()),
    )

    print(render_console(result.experiment))
    if result.measurements:
        print()
        print("EXECUTED BY THE AUDITOR")
        for m in result.measurements:
            mark = "PASS" if m.passed is True else "FAIL" if m.passed is False else "N/A"
            print(f"  [{mark}] {m.control}: {m.detail}")
    if result.outside_scope:
        print()
        print("OUTSIDE WHAT THIS CHECK CAN ESTABLISH")
        for name, why in result.outside_scope:
            print(f"  {name}: {why}")

    if args.html:
        Path(args.html).write_text(render_html(result.experiment))
        print(f"\nwrote {args.html}")
    if args.save:
        record.save(result.experiment, args.save)
        print(f"wrote {args.save}")

    return EXIT_OK if result.passed else EXIT_NOT_CERTIFIED


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

    p_check = sub.add_parser(
        "check", help="audit a circuit directly -- the front door")
    p_check.add_argument("path", nargs="?",
                         help="a Python file defining `circuit` (and optionally "
                              "observable, mitigator, submitted_circuit, claim)")
    p_check.add_argument("--template", action="store_true",
                         help="print a starting-point file and exit")
    p_check.add_argument("--html", metavar="PATH",
                         help="also write a self-contained HTML report")
    p_check.add_argument("--save", metavar="PATH",
                         help="also save the derived experiment record as JSON")
    p_check.set_defaults(func=_cmd_check)

    p_audit = sub.add_parser("audit", help="audit an experiment record")
    p_audit.add_argument("path", help="path to a JSON experiment record")
    p_audit.add_argument("--json", action="store_true",
                         help="emit machine-readable output")
    p_audit.add_argument("--html", metavar="PATH",
                         help="also write a self-contained HTML report")
    p_audit.add_argument("--calibration", metavar="PATH",
                         help="device error rates and gate counts, as JSON; "
                              "turns the verdict into a remedy")
    p_audit.add_argument("--symmetry", action="store_true",
                         help="this state obeys a symmetry checkable in the "
                              "measured basis, so post-selection is available")
    _add_store_arguments(p_audit)
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

    p_remember = sub.add_parser(
        "remember",
        help="show what past audits found, and what they found it in")
    p_remember.add_argument("--circuit", metavar="PATH",
                            help="a record whose circuit to recall against; "
                                 "without one, summarise the whole corpus")
    _add_store_arguments(p_remember)
    p_remember.set_defaults(func=_cmd_remember)

    p_template = sub.add_parser("template", help="print a blank record to fill in")
    p_template.set_defaults(func=_cmd_template)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
