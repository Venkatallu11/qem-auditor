"""Device profiles, because the right method depends on the machine.

Everything here was calibrated against one IBM Eagle chip. That was a
reasonable place to start and a bad place to stop: the recommendation
this engine produces is driven by which error term dominates, and that
term is different on different hardware. A readout-dominated
superconducting device and a gate-limited trapped-ion device are not the
same experiment with different numbers, they are different experiments.

Three architectural facts do most of the work:

**Connectivity.** A superconducting chip connects each qubit to two or
three neighbours, so a circuit that entangles distant qubits pays for
SWAPs that were never in the source. A trapped-ion machine is all-to-all
and pays nothing. Two devices with identical gate error can therefore
run the same circuit at very different effective depths, and the
mitigation answer follows the effective depth, not the written one.

**Readout versus gate error.** On IBM Eagle readout error is roughly ten
times the two-qubit gate error, which is why readout mitigation wins
there and why zero-noise extrapolation -- which cannot reach an error
that does not scale with gate count -- does not. On Quantinuum both are
small and comparable, so neither dominates and the binding constraint
moves to shot noise: the honest advice becomes "take more shots", not
"apply a method".

**Speed against coherence.** Trapped-ion gates are slower by orders of
magnitude and coherence times are longer by orders of magnitude. The
ratio that matters -- circuit duration over T2 -- is what this cares
about, and it is not read off either number alone.

## About these numbers

They are REPRESENTATIVE, dated, and no substitute for your own device's
calibration. Vendor figures move, they are quoted under best conditions,
and they vary qubit to qubit on the same chip. Everything here supports
an ORDERING of methods and an architecture comparison; nothing here
should be quoted as your device's performance.

`profile.replace(...)` takes your measured numbers, and every function
below accepts a profile you built yourself. When a real calibration is
available, use it -- `budget_from_calibration` does not care where the
numbers came from, only that they are yours.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional


class Architecture(Enum):
    """What kind of machine, because it changes which errors dominate."""

    SUPERCONDUCTING = "superconducting"
    TRAPPED_ION = "trapped ion"
    NEUTRAL_ATOM = "neutral atom"


@dataclass(frozen=True)
class DeviceProfile:
    """Representative parameters for one machine.

    `as_of` and `source` are mandatory rather than decorative: a device
    number with no date is a claim with no evidence, and this package
    exists to object to those.
    """

    name: str
    vendor: str
    architecture: Architecture
    two_qubit_error: float
    readout_error: float
    one_qubit_error: float
    qubits: int
    all_to_all: bool
    native_two_qubit: str
    as_of: str
    source: str
    t1_s: Optional[float] = None
    t2_s: Optional[float] = None
    two_qubit_gate_s: Optional[float] = None

    def replace(self, **changes) -> "DeviceProfile":
        """Your measured numbers, on this profile's shape."""
        return replace(self, **changes)

    @property
    def readout_to_gate(self) -> float:
        """How many two-qubit gates' worth of error one measurement costs.

        The single most useful number for choosing a method. Above about
        ten, readout mitigation is the first thing to reach for and
        zero-noise extrapolation cannot help, because readout error does
        not grow when you lengthen the circuit.
        """
        return self.readout_error / self.two_qubit_error

    def routing_multiplier(self, entangling_gates: int,
                           locality: float = 0.5) -> float:
        """Two-qubit gates actually executed, per gate written.

        All-to-all hardware executes what you wrote. On a device with
        nearest-neighbour connectivity, an entangling gate between
        distant qubits becomes a chain of SWAPs, each costing three
        two-qubit gates. `locality` is the fraction of the circuit's
        entangling gates that already act on neighbours; the rest pay.

        Deliberately crude, and it says so: real routing depends on the
        circuit and the compiler. It is here to stop an architecture
        comparison silently pretending that connectivity is free, which
        is the larger error.
        """
        if self.all_to_all:
            return 1.0
        if not 0.0 <= locality <= 1.0:
            raise ValueError(f"locality is a fraction, got {locality}")
        return locality + (1.0 - locality) * 4.0

    def effective_two_qubit_gates(self, written: int,
                                  locality: float = 0.5) -> int:
        return int(round(written * self.routing_multiplier(written, locality)))

    def describe(self) -> str:
        connectivity = ("all-to-all" if self.all_to_all
                        else "nearest-neighbour")
        return (f"{self.name} ({self.vendor}, {self.architecture.value}, "
                f"{self.qubits} qubits, {connectivity})\n"
                f"  two-qubit {self.two_qubit_error:.2%}  "
                f"readout {self.readout_error:.2%}  "
                f"ratio {self.readout_to_gate:.1f}x\n"
                f"  as of {self.as_of}: {self.source}")


#: Representative profiles. Read the module docstring before quoting any
#: of these: they are dated public figures for comparing ARCHITECTURES,
#: not measurements of the machine you are about to run on.
PROFILES = {
    "ibm_eagle": DeviceProfile(
        name="Eagle r3", vendor="IBM", architecture=Architecture.SUPERCONDUCTING,
        two_qubit_error=0.00311, readout_error=0.0293, one_qubit_error=0.00025,
        qubits=127, all_to_all=False, native_two_qubit="ecr",
        t1_s=250e-6, t2_s=120e-6, two_qubit_gate_s=5.4e-7,
        as_of="2026-08",
        source="measured from the qiskit-ibm-runtime FakeKyiv snapshot, the "
               "same numbers examples/real_device_audit.py runs on"),
    "ibm_heron": DeviceProfile(
        name="Heron r2", vendor="IBM", architecture=Architecture.SUPERCONDUCTING,
        two_qubit_error=0.002, readout_error=0.015, one_qubit_error=0.0002,
        qubits=133, all_to_all=False, native_two_qubit="cz",
        t1_s=200e-6, t2_s=150e-6, two_qubit_gate_s=6.8e-8,
        as_of="2026-08",
        source="representative published figures; tunable couplers give a "
               "faster and cleaner two-qubit gate than Eagle"),
    "ionq_aria": DeviceProfile(
        name="Aria", vendor="IonQ", architecture=Architecture.TRAPPED_ION,
        two_qubit_error=0.006, readout_error=0.005, one_qubit_error=0.0006,
        qubits=25, all_to_all=True, native_two_qubit="ms",
        t1_s=100.0, t2_s=1.0, two_qubit_gate_s=6e-4,
        as_of="2026-08",
        source="representative published figures; all-to-all connectivity, "
               "gates ~1000x slower than superconducting and coherence "
               "~10000x longer"),
    "ionq_forte": DeviceProfile(
        name="Forte", vendor="IonQ", architecture=Architecture.TRAPPED_ION,
        two_qubit_error=0.004, readout_error=0.005, one_qubit_error=0.0002,
        qubits=36, all_to_all=True, native_two_qubit="zz",
        t1_s=100.0, t2_s=1.0, two_qubit_gate_s=6e-4,
        as_of="2026-08",
        source="representative published figures; the family the source "
               "project's own hardware runs were taken on"),
    "quantinuum_h2": DeviceProfile(
        name="H2", vendor="Quantinuum", architecture=Architecture.TRAPPED_ION,
        two_qubit_error=0.0015, readout_error=0.003, one_qubit_error=0.00003,
        qubits=56, all_to_all=True, native_two_qubit="zz",
        t1_s=100.0, t2_s=2.0, two_qubit_gate_s=2.5e-4,
        as_of="2026-08",
        source="representative published figures; QCCD architecture with "
               "all-to-all connectivity and mid-circuit measurement"),
    "rigetti_ankaa": DeviceProfile(
        name="Ankaa-3", vendor="Rigetti", architecture=Architecture.SUPERCONDUCTING,
        two_qubit_error=0.015, readout_error=0.03, one_qubit_error=0.001,
        qubits=84, all_to_all=False, native_two_qubit="iswap",
        t1_s=20e-6, t2_s=20e-6, two_qubit_gate_s=7e-8,
        as_of="2026-08",
        source="representative published figures; fast gates, shorter "
               "coherence than the IBM devices above"),
}


def profile(key: str) -> DeviceProfile:
    """Look up a profile, listing the alternatives when the key is wrong."""
    try:
        return PROFILES[key]
    except KeyError:
        raise KeyError(
            f"no profile {key!r}. Available: {', '.join(sorted(PROFILES))}. "
            "Or build a DeviceProfile from your own calibration -- these are "
            "representative figures for comparing architectures, not "
            "measurements of your machine.") from None


def budget_for(device: DeviceProfile, *, two_qubit_gates: int,
               measured_qubits: int, one_qubit_gates: int = 0,
               shots: int = 10_000, locality: float = 0.5):
    """The error budget this circuit would have on this machine.

    Routing is applied first: on nearest-neighbour hardware the circuit
    that runs is longer than the circuit that was written, and budgeting
    the written one would flatter every superconducting device in the
    comparison.
    """
    from .prescribe import budget_from_calibration

    executed = device.effective_two_qubit_gates(two_qubit_gates, locality)
    duration = (executed * device.two_qubit_gate_s
                if device.two_qubit_gate_s else None)
    return budget_from_calibration(
        two_qubit_gates=executed,
        one_qubit_gates=one_qubit_gates,
        measured_qubits=measured_qubits,
        two_qubit_error=device.two_qubit_error,
        one_qubit_error=device.one_qubit_error,
        readout_error=device.readout_error,
        shots=shots,
        circuit_duration_s=duration,
        t2_s=device.t2_s)


def compare(devices, *, two_qubit_gates: int, measured_qubits: int,
            one_qubit_gates: int = 0, shots: int = 10_000,
            locality: float = 0.5) -> list:
    """The same circuit across machines: what dominates, and what survives.

    Returns one row per device, ordered by surviving signal. The point of
    the table is not the ranking -- it is that the DOMINANT ERROR changes
    between rows, and the dominant error is what picks the method.
    """
    from .prescribe import feasibility

    rows = []
    for device in devices:
        executed = device.effective_two_qubit_gates(two_qubit_gates, locality)
        budget = budget_for(device, two_qubit_gates=two_qubit_gates,
                            measured_qubits=measured_qubits,
                            one_qubit_gates=one_qubit_gates, shots=shots,
                            locality=locality)
        survival = feasibility(
            executed,
            {"ecr_error": device.two_qubit_error,
             "readout_error": device.readout_error},
            n_qubits=measured_qubits)
        rows.append({
            "device": device,
            "executed_two_qubit_gates": executed,
            "budget": budget,
            "dominant": budget.dominant.name if budget.is_decisive else None,
            "feasibility": survival,
        })
    rows.sort(key=lambda row: row["feasibility"].survival, reverse=True)
    return rows
