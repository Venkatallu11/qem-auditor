"""Starting-point files handed to users by the CLI."""

CHECK_TEMPLATE = '''"""Your circuit, for qem-auditor to check.

Define `circuit`. Everything else is optional, and anything you leave out
is simply not checked -- never assumed to have passed.

    circuit            the circuit as designed                    (required)
    observable         what you measure                (enables the ideal control)
    mitigator          your mitigation pipeline        (enables the ideal control)
    submitted_circuit  the circuit as actually SENT    (enables the circuit check)
    amplified_circuit  the noise-amplified arm you BUILD, if your pipeline
                       inserts gates (ZNE folding) -- supplying this is what
                       lets the auditor check the gates survived the compiler
    claim              what you are claiming, one sentence
    backend            where it runs
    replicate_errors   errors from independent re-executions, if you have them

`mitigator` receives an expectation oracle and returns your mitigated value:

    def mitigator(expectation):
        return expectation(circuit, observable)

Run:  qem-auditor check this_file.py
"""
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp

claim = "My mitigated result reaches chemical accuracy."
backend = "aer_simulator"

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.rx(0.3, 0)

observable = SparsePauliOp("ZZ")

# The circuit as it would REALLY be submitted, after transpilation.
submitted_circuit = transpile(circuit, basis_gates=["u", "cx"],
                              optimization_level=0)

# If your pipeline builds a noise-amplified arm (ZNE gate folding), define
# it here as well. This is what lets the auditor check the inserted gates
# survived the compiler -- a check unitary equivalence alone CANNOT make,
# because a fold pair is supposed to leave the unitary unchanged.
#
# amplified_circuit = folded(circuit, folds=2)
# submitted_circuit = transpile(amplified_circuit, basis_gates=["u", "cx"],
#                               optimization_level=0)


def mitigator(expectation):
    """Your pipeline. Replace this with the real one."""
    return expectation(circuit, observable)
'''
