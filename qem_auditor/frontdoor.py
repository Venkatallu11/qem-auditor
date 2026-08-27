"""Bring a circuit, get a verdict.

Everything else in this package takes an `Experiment` record. That is the
right internal object -- it is what the gates reason over -- but it is the
wrong thing to ask a researcher for. Someone who has a circuit, an
observable and a mitigation function should not have to hand-write a JSON
record describing their own work before the auditor will talk to them,
and asking them to means the honest answer depends on how honestly they
filled in a form.

So this builds the record from the artifacts themselves. Gate counts,
qubit counts and the basis are read off the circuit; the controls it can
execute, it executes. Nothing here is taken on the user's word, because
nothing here is asked of the user.

What it deliberately does NOT do is fill in the controls it cannot run.
`target_leakage`, `adversarial` and `free_parameter_floor` stay unrun,
and the verdict reflects that. A front door that quietly marked them
passed would make the whole tool a rubber stamp with a nice interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from .schema import (
    CircuitSpec,
    ClaimType,
    Controls,
    Experiment,
    NoiseSpec,
    Outputs,
    Replicate,
    ReplicateKind,
    TranspilationStatus,
    UncertaintyCoverage,
)

# What an auditor can establish from artifacts alone, versus what needs
# the claimant's own experimental record. Kept explicit so the front door
# can say so rather than leaving a user to infer it.
MECHANIZABLE = ("unitary_equivalence", "ideal_control", "determinism_check")
NEEDS_THE_RESEARCHER = (
    ("target_leakage", "whether the known answer influenced tuning -- procedural, "
                       "and not visible in a circuit"),
    ("adversarial", "whether negative controls fail loudly -- needs your own "
                    "fitting code to shuffle and refit"),
    ("free_parameter_floor", "whether a free parameter degenerates toward the "
                             "answer -- needs your method's parameters"),
    ("reproducibility", "independent re-executions -- the auditor cannot run "
                        "your submissions for you"),
)


def describe_circuit(circuit: Any, optimization_level: Optional[int] = None,
                     basis_gates: Optional[Sequence[str]] = None) -> CircuitSpec:
    """Read a CircuitSpec off a real circuit, rather than asking for one."""
    try:
        ops = dict(circuit.count_ops())
    except Exception:
        ops = {}
    two_qubit = 0
    one_qubit = 0
    try:
        for instruction in circuit.data:
            n = len(getattr(instruction, "qubits", ()) or ())
            if n >= 2:
                two_qubit += 1
            elif n == 1:
                one_qubit += 1
    except Exception:
        pass
    return CircuitSpec(
        circuit_id=getattr(circuit, "name", "") or "circuit",
        native_gate_set=", ".join(basis_gates) if basis_gates else ", ".join(sorted(ops)),
        transpilation_status=TranspilationStatus.UNVERIFIED,
        optimization_level=optimization_level,
        n_1q_gates=one_qubit or None,
        n_2q_gates=two_qubit or None,
        n_qubits=getattr(circuit, "num_qubits", None),
    )


@dataclass
class VerificationInputs:
    """What a researcher actually has in front of them."""

    circuit: Any
    observable: Any = None
    mitigator: Optional[Callable] = None
    submitted_circuit: Any = None
    """The circuit as it would actually be SENT -- after transpilation and
    everything else the pipeline does to it."""

    amplified_circuit: Any = None
    """The noise-amplified arm your pipeline builds, before transpilation,
    if it builds one (ZNE folding and the like).

    Supplying this is what tells the auditor that extra gates were
    inserted DELIBERATELY, and that changes the check completely. Without
    it, the auditor can only ask whether the submitted circuit implements
    the intended unitary -- and a fold pair is *supposed* to leave the
    unitary unchanged, so that question passes happily while the
    transpiler quietly removes the pairs. With it, the auditor also checks
    the gate count survived, which is the half that catches the failure.
    """

    claim: str = ""
    backend: str = "unspecified"
    shots: int = 20_000
    threshold_kcal: float = 0.25
    replicate_errors: Sequence[float] = ()
    """Independent re-execution results, if any exist. Left empty is
    honest and produces a lower verdict; it is not a field to invent."""


def build_experiment(inputs: VerificationInputs,
                     experiment_id: str = "") -> Experiment:
    """An Experiment record derived from artifacts, with nothing assumed.

    Every control starts None. The verifier fills in only what it
    measures, so a record that comes out of here with a control set has
    that control's evidence behind it.
    """
    spec = describe_circuit(inputs.circuit)
    return Experiment(
        experiment_id=experiment_id or f"verify_{spec.circuit_id}",
        claim=inputs.claim or "This mitigated result is trustworthy.",
        claim_type=ClaimType.ABSOLUTE_ACCURACY,
        description=(
            f"Submitted for verification: {spec.n_qubits} qubit(s), "
            f"{spec.n_2q_gates or 0} two-qubit gate(s), "
            f"{'with' if inputs.mitigator else 'without'} a mitigation pipeline."),
        backend=inputs.backend,
        shots=inputs.shots,
        circuit=spec,
        noise=NoiseSpec(noise_model=inputs.backend),
        controls=Controls(),  # nothing asserted; only measurement fills these
        outputs=Outputs(
            replicates=[Replicate(v, ReplicateKind.INDEPENDENT_SUBMISSION,
                                  f"draw_{i}")
                        for i, v in enumerate(inputs.replicate_errors)],
            uncertainty=UncertaintyCoverage(shot_noise=True),
        ),
    )


def unverifiable_here() -> list[tuple[str, str]]:
    """What this path structurally cannot establish, and why.

    Surfaced in the report rather than left implicit. A user who runs the
    front door and sees NOT ESTABLISHED deserves to know which part of
    that is their method's fault and which part is simply outside what an
    auditor can check from a circuit.
    """
    return list(NEEDS_THE_RESEARCHER)
