"""What a mitigation method costs to actually submit.

Every other module here prices a method in error: how much of the budget
it reaches, what it assumes, whether it survived its attacks. None of
them priced it in money, and on real hardware that is the binding
constraint far more often than accuracy is.

The rate card below is IonQ's own, from their `GET /jobs/estimate`
endpoint:

    job_cost_minimum   $25.7899   charged once per JOB
    cost_1q_gate       $0.000164  per gate, per shot
    cost_2q_gate       $0.001121  per gate, per shot

    cost = max(job minimum, (n1q*r1 + n2q*r2) * shots * circuits)

Three real quotes from that endpoint reproduce exactly under this
formula, including a 125-circuit job that came back at precisely 125.0x
the one-circuit price -- which is what establishes that the minimum is
per JOB and not per circuit.

TWO REGIMES, AND THE ONE THAT MISLEADS
--------------------------------------
Below the minimum, shots are genuinely free: two real submissions of the
same 5-qubit circuit at 100 and at 500 shots both billed $25.79, and it
is tempting to conclude cost is per-circuit and shots do not matter.
That conclusion is wrong, and this module used to make it. Both jobs
simply sat under the floor. For that circuit the floor stops binding at
806 shots, and at 2,000 shots the same job costs $64.02.

So "shots are free" is a statement about a REGIME, not about a machine,
and the regime has an exact boundary this module computes rather than
assumes. Quoting the flat price outside it understates a real bill by
whatever factor the shot count exceeds the break-even.

WHY GATE COUNTS ARE A PRICE AND NOT ONLY AN ERROR BUDGET
--------------------------------------------------------
Cost is charged per gate per shot, so the circuit that compilation
actually produces is the one that gets billed. The real job here was
written with 20 one-qubit gates and executed 120 of them after native
compilation: the same change that made the error budget worse also
multiplied the price per shot by 2.05x, and moved the break-even from
1,652 shots down to 806.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RateCard:
    """A provider's real pricing, and where it came from."""

    job_minimum_usd: float
    one_qubit_usd: float
    two_qubit_usd: float
    as_of: str
    source: str

    def per_shot(self, one_qubit_gates: int, two_qubit_gates: int) -> float:
        """What one shot of one circuit costs, before the floor."""
        return (one_qubit_gates * self.one_qubit_usd
                + two_qubit_gates * self.two_qubit_usd)

    def gate_cost(self, circuits: int, one_qubit_gates: int,
                  two_qubit_gates: int, shots: int) -> float:
        return circuits * shots * self.per_shot(one_qubit_gates, two_qubit_gates)

    def quote(self, circuits: int, one_qubit_gates: int, two_qubit_gates: int,
              shots: int, one_job: bool = True) -> float:
        """Price a set of circuits.

        `one_job` is not a detail. The minimum is charged per JOB, so
        submitting N circuits separately pays it N times while batching
        them pays it once -- worth real money below the break-even and
        worth nothing above it, where the floor never binds anyway.
        """
        gates = self.gate_cost(circuits, one_qubit_gates, two_qubit_gates, shots)
        if one_job:
            return max(self.job_minimum_usd, gates)
        return circuits * max(self.job_minimum_usd,
                              self.gate_cost(1, one_qubit_gates,
                                             two_qubit_gates, shots))

    def break_even_shots(self, one_qubit_gates: int, two_qubit_gates: int,
                         circuits: int = 1) -> Optional[int]:
        """The shot count where the job minimum stops covering the bill.

        Below it, more shots are free. Above it, cost is linear in shots
        and the flat price people remember from small jobs is an
        understatement. None when the circuit has no billable gates.
        """
        per_shot = self.per_shot(one_qubit_gates, two_qubit_gates) * circuits
        if per_shot <= 0:
            return None
        return math.ceil(self.job_minimum_usd / per_shot)


#: Real, from the provider's own estimate endpoint rather than a price
#: page -- and one account, one backend, one month, which is exactly as
#: far as it should be trusted.
RATE_CARDS = {
    "ionq_forte": RateCard(
        job_minimum_usd=25.7899,
        one_qubit_usd=0.000164,
        two_qubit_usd=0.001121,
        as_of="2026-08",
        source="IonQ GET /jobs/estimate for qpu.forte-1. Three quotes "
               "reproduce exactly under this formula, including a "
               "125-circuit job at precisely 125.0x the one-circuit "
               "price, which is what shows the minimum is per job. Two "
               "real submissions at 100 and 500 shots both billed "
               "$25.79, both under the floor"),
}


@dataclass(frozen=True)
class MethodCost:
    """What one method costs to run, in the units a device charges in."""

    method: str
    circuits: int
    retained: float
    why: str

    def effective_shots(self, shots: int) -> int:
        """Shots that survive to the estimate.

        Post-selection is free in circuits and not free in data: a
        method that keeps 90% of its shots has paid for 100% of them
        and computes its floor from 90%, which widens the floor by
        1/sqrt(0.9).
        """
        return int(shots * self.retained)

    def quote(self, rates: RateCard, shots: int, one_qubit_gates: int,
              two_qubit_gates: int, one_job: bool = True) -> "Quote":
        return Quote(
            cost=self,
            usd=rates.quote(self.circuits, one_qubit_gates, two_qubit_gates,
                            shots, one_job),
            usd_separate_jobs=rates.quote(self.circuits, one_qubit_gates,
                                          two_qubit_gates, shots, False),
            break_even=rates.break_even_shots(one_qubit_gates,
                                              two_qubit_gates, self.circuits),
            shots_paid=shots,
            shots_kept=self.effective_shots(shots))


@dataclass(frozen=True)
class Quote:
    """A price, with what it buys and which regime it is in."""

    cost: MethodCost
    usd: float
    usd_separate_jobs: float
    break_even: Optional[int]
    shots_paid: int
    shots_kept: int

    @property
    def floor_binds(self) -> bool:
        """Are we in the regime where more shots are free?"""
        return self.break_even is not None and self.shots_paid < self.break_even

    @property
    def discarded(self) -> int:
        return self.shots_paid - self.shots_kept

    def describe(self) -> str:
        line = (f"  {self.cost.method:38s} {self.cost.circuits:5d} circuits  "
                f"${self.usd:12,.2f}")
        line += "  at the job floor" if self.floor_binds else "  gate-metered"
        if self.discarded:
            line += f", keeps {self.shots_kept:,}/{self.shots_paid:,} shots"
        return line


def circuits_for(method: str, *, qubits: int = 4, training: int = 5,
                 folds: int = 3, twirls: int = 16,
                 pec_samples: int = 500) -> MethodCost:
    """The circuit count and shot retention of one method by name.

    These are structural: they follow from what the method has to run,
    independently of any device. Names match `prescribe.CATALOGUE` where
    they overlap so a prescription can be priced without a translation
    table.
    """
    table = {
        "unmitigated": MethodCost(
            "unmitigated", 1, 1.0, "the circuit itself"),
        "more shots": MethodCost(
            "more shots", 1, 1.0,
            "the same circuit -- free while the job minimum covers the "
            "bill, and linear in shots the moment it stops"),
        "readout error mitigation (REM)": MethodCost(
            "readout error mitigation (REM)", 1 + 2, 1.0,
            "the circuit, plus all-zeros and all-ones calibration"),
        "REM (tensored)": MethodCost(
            "REM (tensored)", 1 + 2, 1.0,
            "two calibration circuits regardless of width, which is what "
            "the tensored assumption buys"),
        "REM (full confusion matrix)": MethodCost(
            "REM (full confusion matrix)", 1 + 2 ** qubits, 1.0,
            f"one calibration circuit per basis state: 2^{qubits} of them, "
            "which is why the tensored form exists"),
        "zero-noise extrapolation (ZNE)": MethodCost(
            "zero-noise extrapolation (ZNE)", folds, 1.0,
            f"one circuit per noise scale ({folds}) -- and folding MULTIPLIES "
            "the gate count, so on a gate-metered bill the folded copies "
            "cost more than the original rather than the same"),
        "REM then ZNE": MethodCost(
            "REM then ZNE", folds + 2, 1.0,
            "the folds, plus readout calibration shared across them"),
        "Clifford data regression (CDR)": MethodCost(
            "Clifford data regression (CDR)", 1 + training, 1.0,
            f"the circuit, plus {training} near-Clifford training circuits "
            "whose answers are known classically"),
        "vnCDR": MethodCost(
            "vnCDR", folds * (1 + training), 1.0,
            "a full CDR training set at every noise scale, which is the "
            "price of making both assumptions instead of one"),
        "Pauli twirling": MethodCost(
            "Pauli twirling", twirls, 1.0,
            f"{twirls} independently randomised copies; the shots divide "
            "among them rather than adding"),
        "probabilistic error cancellation (PEC)": MethodCost(
            "probabilistic error cancellation (PEC)", pec_samples, 1.0,
            f"{pec_samples} sampled circuits from the quasi-probability "
            "decomposition, and the sample count grows with the error rate"),
        "symmetry verification (post-selection)": MethodCost(
            "symmetry verification (post-selection)", 1, 0.90,
            "no extra circuits at all -- it discards shots rather than "
            "submitting more. The 90% retention is measured: three real "
            "trapped-ion circuits kept 91.5%, 90.2% and 90.1% of their "
            "shots under an ancilla parity check"),
    }
    if method not in table:
        raise KeyError(
            f"no circuit count is recorded for {method!r}. Add one rather "
            "than defaulting to 1: a method priced at one circuit because "
            "nobody looked is exactly the recommendation that blows a "
            "hardware budget.")
    return table[method]


@dataclass(frozen=True)
class Affordability:
    """What a budget can and cannot buy, and the advice that follows."""

    quotes: tuple
    budget_usd: float
    shots: int
    rates: RateCard
    one_qubit_gates: int
    two_qubit_gates: int

    @property
    def affordable(self) -> tuple:
        return tuple(q for q in self.quotes if q.usd <= self.budget_usd)

    @property
    def refused(self) -> tuple:
        return tuple(q for q in self.quotes if q.usd > self.budget_usd)

    def describe(self) -> str:
        single = self.rates.break_even_shots(self.one_qubit_gates,
                                             self.two_qubit_gates)
        lines = [
            f"  budget ${self.budget_usd:,.2f}, {self.shots:,} shots, a circuit "
            f"of {self.one_qubit_gates} one-qubit and {self.two_qubit_gates} "
            f"two-qubit gates",
            f"  ({self.rates.as_of}: {self.rates.source})",
            "",
            f"  one circuit costs ${self.rates.per_shot(self.one_qubit_gates, self.two_qubit_gates):.6f}/shot, "
            f"so the ${self.rates.job_minimum_usd:.2f} job minimum covers it "
            f"up to {single:,} shots.",
        ]
        lines.append(
            f"  At {self.shots:,} shots you are "
            + ("UNDER that line, where more shots are genuinely free."
               if self.shots < single else
               f"OVER it by {self.shots / single:.1f}x, where every shot is "
               "billed and 'shots are free' stops being true."))
        lines.append("")
        for quote in sorted(self.quotes, key=lambda q: q.usd):
            mark = "    " if quote.usd <= self.budget_usd else "  X "
            lines.append(mark + quote.describe().lstrip())
        if self.refused:
            lines.append("")
            lines.append(f"  {len(self.refused)} of {len(self.quotes)} methods "
                         "cost more than the budget. That is a fact about the "
                         "pricing, not about how good they are.")
        batched = [q for q in self.quotes if q.usd_separate_jobs > q.usd * 1.01]
        if batched:
            lines.append("")
            lines.append(
                f"  The minimum is charged per JOB, so batching circuits into "
                f"one submission matters below the break-even: "
                f"{batched[0].cost.method} costs "
                f"${batched[0].usd:,.2f} batched against "
                f"${batched[0].usd_separate_jobs:,.2f} as separate jobs.")
        return "\n".join(lines)


def affordable_methods(methods, budget_usd: float, shots: int,
                       one_qubit_gates: int, two_qubit_gates: int,
                       rates: Optional[RateCard] = None,
                       one_job: bool = True, **kwargs) -> Affordability:
    """Price a list of method names against a real budget.

    The gate counts are required rather than defaulted, because cost is
    charged per gate per shot: a method priced without them is priced
    for a circuit nobody ran. Use the counts the provider reports as
    EXECUTED, not the ones the circuit was written with -- on the real
    job behind this module those differed by 6x on one-qubit gates and
    doubled the price per shot.
    """
    rates = rates or RATE_CARDS["ionq_forte"]
    quotes = tuple(
        circuits_for(name, **kwargs).quote(rates, shots, one_qubit_gates,
                                           two_qubit_gates, one_job)
        for name in methods)
    return Affordability(quotes, budget_usd, shots, rates,
                         one_qubit_gates, two_qubit_gates)
