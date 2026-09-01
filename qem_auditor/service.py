"""The engine as a service surface: one call per question a person asks.

Deliberately NOT a quantum platform. There are no tools here for listing
backends, submitting jobs, checking queues, or estimating vendor cost --
`quantum-verifier` already serves that surface well, and duplicating it
would make two projects that are each worse than one. The division is
clean:

    quantum-verifier  ->  will the right circuit survive this hardware?
    qem-auditor       ->  is it the right circuit, and is the number real?

So every tool here is a judgement, not an errand. Each returns plain
JSON-able data with the reasoning attached, because a recommendation
whose reason a person cannot inspect is an oracle, and this project's
entire objection is to oracles.

The functions are pure and importable without any MCP machinery, so they
are tested directly and the protocol binding stays a thin shell over
code that is already checked.
"""
from __future__ import annotations

from typing import Any, Optional

from .control import distribution_shift, isolate_effect
from .engine import MethodOutcome, recommend
from .power import compare as compare_runs
from .prescribe import ErrorBudget, ErrorSource, budget_from_calibration, feasibility
from .reversible import preflight_gate


def _load_circuit(qasm: str):
    """Parse OPENQASM, refusing in the way that says what to do next.

    `mcx` and `mcz` are the common case: they read as standard and are
    not in qelib1.inc, so a file using them compiles nowhere. Saying that
    by name saves the reader the hour it cost to find out here.
    """
    from qiskit import qasm2
    try:
        return qasm2.loads(qasm, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    except Exception as failure:
        message = str(failure)
        if "mcx" in message or "mcz" in message:
            message += (" -- note that mcx and mcz are NOT OPENQASM 2.0 gates and "
                        "are not in qelib1.inc, so this file compiles nowhere "
                        "standard. Define them with a `gate` declaration, or emit "
                        "the decomposition your framework produces.")
        raise ValueError(message)


def check_circuit_computes(qasm: str, marked_inputs: list, n_inputs: int,
                           input_qubits: Optional[int] = None,
                           ancillas: Optional[list] = None) -> dict:
    """Does the circuit mark exactly the states its author says it does?

    Exhaustive and exact, not sampled. This is the check that belongs
    before any simulation, because an ideal simulation and a noisy one of
    the SAME wrong circuit agree with each other -- the disagreement a
    simulation-based pipeline looks for never appears.

    `marked_inputs` is the specification, supplied by whoever is making
    the claim. It is never inferred: an auditor that writes both sides of
    the comparison is not auditing anything.
    """
    circuit = _load_circuit(qasm)
    width = input_qubits if input_qubits is not None else circuit.num_qubits
    marked = set(marked_inputs)

    def encode(value):
        bits = [0] * circuit.num_qubits
        for bit in range(width):
            bits[bit] = (value >> bit) & 1
        return bits

    result = preflight_gate(circuit, predicate=lambda v: v in marked,
                            n_inputs=n_inputs, encode=encode,
                            ancillas=ancillas or range(width, circuit.num_qubits))
    report = result.pop("report")
    if report is not None:
        result["inputs_checked"] = report.n_inputs
        result["specified"] = len(report.expected)
        result["circuit_marks"] = len(report.marked)
        result["missing"] = len(report.false_negatives)
        result["spurious"] = len(report.false_positives)
        result["ancillas_not_restored"] = report.dirty_ancillas
        result["is_a_phase_oracle"] = report.is_a_phase_oracle
    return result


def check_feasibility(two_qubit_gates: int, n_qubits: int,
                      two_qubit_error: float, readout_error: float) -> dict:
    """Is there a signal left for any method to improve?

    Asked before methods are ranked, because none of them creates an
    estimate -- they only improve one that exists.
    """
    verdict = feasibility(two_qubit_gates,
                          {"ecr_error": two_qubit_error,
                           "readout_error": readout_error},
                          n_qubits=n_qubits)
    return {
        "mitigable": verdict.is_mitigable,
        "survival": verdict.survival,
        "gate_survival": verdict.gate_survival,
        "readout_survival": verdict.readout_survival,
        "shots_for_a_signal": verdict.shots_for_a_signal,
        "affordable_two_qubit_gates": verdict.affordable_two_qubit_gates,
        "verdict": verdict.format_verdict(),
    }


def recommend_mitigation(results: list, two_qubit_gates: Optional[int] = None,
                         n_qubits: Optional[int] = None,
                         two_qubit_error: Optional[float] = None,
                         readout_error: Optional[float] = None) -> dict:
    """Which method to use, why, and what the runs do not settle.

    `results` is one entry per method: `{"name", "errors" (per
    independent run), "cost", "sensitivity"}`. Sensitivity is the
    scramble-attack ratio; a method without one is reported as never
    attacked rather than assumed honest.
    """
    survival = None
    if None not in (two_qubit_gates, n_qubits, two_qubit_error, readout_error):
        survival = feasibility(two_qubit_gates,
                               {"ecr_error": two_qubit_error,
                                "readout_error": readout_error}, n_qubits)

    outcomes = [MethodOutcome(name=entry["name"],
                              errors=tuple(entry["errors"]),
                              cost=float(entry.get("cost", 1.0)),
                              sensitivity=entry.get("sensitivity"))
                for entry in results]
    advice = recommend(outcomes, feasibility=survival)
    return {
        "recommended": advice.recommended,
        "reason": advice.reason,
        "tiers": [list(tier) for tier in advice.tiers],
        "tied_with": list(advice.tied_with),
        "disqualified": list(advice.disqualified),
        "never_attacked": list(advice.unverified),
        "report": advice.format_report(),
    }


def compare_two_methods(name_a: str, errors_a: list,
                        name_b: str, errors_b: list) -> dict:
    """Can these runs tell the two apart, and if not, what would it take?"""
    result = compare_runs(name_a, list(errors_a), name_b, list(errors_b))
    return {
        "distinguishable": result.distinguishable,
        "better": result.better,
        "gap": result.gap,
        "separation_in_standard_errors": result.separation,
        "runs_each_to_settle": result.required_n,
        "summary": result.describe(),
    }


def falsify_claim(circuit_counts: dict, control_counts: dict,
                  marked: Optional[list] = None,
                  entangling_gates_removed: int = 0) -> dict:
    """Is the effect the mechanism, or the apparatus?

    Give the counts from the circuit and from its entanglement-free
    control. With `marked`, reports the isolated effect against the bar
    it has to clear; without, reports how far the distribution moved
    against what identical distributions give at this width and shot
    count -- which is emphatically not zero.
    """
    if marked:
        effect = isolate_effect(circuit_counts, control_counts, marked,
                                removed=entangling_gates_removed)
        return {
            "mode": "marked",
            "isolated_effect": effect.effect,
            "half_width": effect.half_width,
            "distinguishable_from_zero": effect.real,
            "shots_each_to_resolve": effect.shots_for_a_signal,
            "summary": effect.describe(),
        }
    shift = distribution_shift(circuit_counts, control_counts)
    return {
        "mode": "discovery",
        "total_variation": shift.observed,
        "same_distribution_null": shift.null_median,
        "null_95th_percentile": shift.null_high,
        "excess_over_null": shift.excess,
        "shift_established": shift.real,
        "most_boosted": list(shift.gainers),
        "summary": shift.describe(),
    }


def budget_from_device(two_qubit_error: float, readout_error: float,
                       measured_qubits: int,
                       one_qubit_error: float = 0.0,
                       two_qubit_gates: int = 1, one_qubit_gates: int = 0,
                       shots: int = 10_000) -> dict:
    """Where the error actually comes from, from calibration numbers alone.

    An estimate, and labelled as one: it earns an ORDERING of methods,
    never a quoted ceiling. Quoting a number from an estimated budget is
    how methods came to "beat" their own bound by 2.5x here once.
    """
    budget = budget_from_calibration(
        two_qubit_error=two_qubit_error, readout_error=readout_error,
        one_qubit_error=one_qubit_error, two_qubit_gates=two_qubit_gates,
        one_qubit_gates=one_qubit_gates, measured_qubits=measured_qubits,
        shots=shots)
    return {
        "shares": {source.name: budget.share(source) for source in ErrorSource
                   if budget.share(source) > 0},
        "dominant": budget.dominant.name if budget.is_decisive else None,
        "provenance": str(budget.provenance),
        "note": ("estimated from calibration: this supports an ordering of "
                 "methods, not a quoted best case"),
    }


#: Everything this service exposes. Named here so the MCP binding and the
#: tests enumerate the same list and cannot drift apart.
TOOLS = {
    "check_circuit_computes": check_circuit_computes,
    "check_feasibility": check_feasibility,
    "recommend_mitigation": recommend_mitigation,
    "compare_two_methods": compare_two_methods,
    "falsify_claim": falsify_claim,
    "budget_from_device": budget_from_device,
}
