"""QEM-Trust case 6: the current best result from quantum-chemistry-vqe,
run through the same auditor that flags failures -- deliberately NOT told
in advance that this is "the winner."

Source: quantum-chemistry-vqe, Tasks 39-40 (native ancilla-parity leakage
detection + conditioned probabilistic error cancellation, H4 entanglement
forging, K=6). Real evidence collected:

- ideal_control: the native ancilla-parity circuit was verified locally
  against its own exact/noiseless regression check to machine precision
  (diff < 1e-8) before any real submission.
- target_leakage_check: the joint Schmidt frame used for correction was
  FREELY fit (never fixed to the identity / never tuned against the
  known exact H4 energy); a shuffled-label adversarial fit of the same
  frame produced ratios of 2263.6x/2429.0x vs. a real fit -- the exact
  kind of test that would catch leakage, and it did not find any.
- adversarial_check: wrong-parity postselection, a shuffled ancilla bit,
  a wrong-sign correction, and the shuffled-label frame fit above all
  failed loudly (large, obviously-wrong errors), exactly as a genuine
  effect requires. Independently reconfirmed on a second real data draw.
- reproducibility: 4 of this project's own 8-draw replication target
  completed as independent real `ionq_simulator` submissions (546
  circuits, 20k shots each), all 4 selecting the identical correction
  candidate on both backends, landing in a 0.0105-0.0192 kcal/mol band.
- Q95 (informational-only vs. the known exact H4 energy, robustness
  envelope across sampled real noise-parameter uncertainty, INCLUDING
  p_readout uncertainty -- the more complete, less flattering of the two
  numbers on record): 0.0491 kcal/mol.
- Real trapped-ion hardware (IonQ qpu.forte-enterprise-1): the circuit
  construction and postselection statistic were spot-checked (a handful
  of real circuits, not the full 21-slot x 13-group sweep needed for a
  real-hardware energy number) and matched the simulator closely. This
  is NOT a full real-hardware energy validation.

Expected auditor verdict: PROMISING / REQUIRES FURTHER CERTIFICATION --
not yet CERTIFIED, because full 8-draw replication isn't complete and
there is no full real-hardware energy validation, even though every hard
gate and the chemical-accuracy bar are currently clean.
"""
from qem_auditor import (
    CircuitSpec,
    Controls,
    Experiment,
    FailureMode,
    NoiseSpec,
    Outputs,
    Replicate,
    ReplicateKind,
    TranspilationStatus,
    UncertaintyCoverage,
    Verdict,
)

EXPERIMENT = Experiment(
    experiment_id="h4_ancilla_qed_conditioned_pec",
    claim="Ancilla-parity leakage detection with conditioned PEC reaches chemical accuracy on H4.",
    description=(
        "Native ancilla-parity leakage detection + conditioned PEC, H4 entanglement "
        "forging (K=6), real ionq_simulator replication draws 0-3."
    ),
    backend="ionq_simulator (aria-1, forte-1)",
    shots=20_000,
    circuit=CircuitSpec(
        circuit_id="h4_ef_ancilla_native_k6",
        native_gate_set="IonQ native (GPi/GPi2/ZZ)",
        transpilation_status=TranspilationStatus.VERIFIED_EQUIVALENT,
        optimization_level=0,
        n_1q_gates=120,
        n_2q_gates=11,
        n_qubits=5,
    ),
    noise=NoiseSpec(
        noise_model="aria-1 / forte-1 named profiles, randomized robustness envelope",
        calibration_source="real measured sweeps; envelope intervals justified from them",
        calibration_uncertainty_propagated=True,
    ),
    controls=Controls(
        ideal_control=True,
        unitary_equivalence=True,
        target_leakage_check=True,
        free_parameter_floor_test=True,
        adversarial_check=True,
        heldout_check=True,
        extrapolation_in_domain=True,
        determinism_check=True,
        reproducibility_checked=True,
    ),
    outputs=Outputs(
        raw_error_kcal=None,  # not meaningful as a single number for a multi-slot pipeline
        mitigated_error_kcal=0.0144,  # mean of the 4 completed draws
        # Genuinely independent submissions: 4 separate real ionq_simulator
        # jobs, 546 circuits each, not resamples of one collection.
        replicates=[
            Replicate(0.01555, ReplicateKind.INDEPENDENT_SUBMISSION, "draw_0"),
            Replicate(0.01535, ReplicateKind.INDEPENDENT_SUBMISSION, "draw_1"),
            Replicate(0.01175, ReplicateKind.INDEPENDENT_SUBMISSION, "draw_2"),
            Replicate(0.01485, ReplicateKind.INDEPENDENT_SUBMISSION, "draw_3"),
        ],
        q95_kcal=0.0491,  # robustness envelope including p_readout uncertainty
        n_replicates_target=8,
        uncertainty=UncertaintyCoverage(
            shot_noise=True,
            method_monte_carlo=True,
            cross_submission=True,
            noise_model=True,
        ),
    ),
    real_hardware_full_validation=False,
    suspected_failure_modes=[FailureMode.UNDER_POWERED],
    notes=(
        "4/8 draws of this project's own replication target completed. Real hardware: "
        "3 circuits run on qpu.forte-enterprise-1 at 2000 shots, retained-parity "
        "fractions 91.5/90.2/90.1% against the simulator's 90.7/91.5% -- consistent, and "
        "a spot-check rather than an energy reconstruction. A full 21-slot x 13-group "
        "sweep is ~273 circuits at ~$25/circuit (cost is per-circuit, not per-shot, "
        "confirmed by a 100->500 shot probe costing the same), so ~$6,825 -- which is "
        "why real_hardware_full_validation stays False."
    ),
)

# Asserted by run_benchmarks.py. The point of pinning this one is the
# opposite of the failure cases: the auditor must keep REFUSING to certify
# the project's own best result while replication and real-hardware
# validation are incomplete, no matter how clean every gate looks.
EXPECTED_VERDICT = Verdict.PROMISING
EXPECTED_PRIMARY_FAILURE_MODE = None
