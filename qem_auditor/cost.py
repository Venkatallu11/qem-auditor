"""What a mitigation method costs to actually submit.

Every other module here prices a method in error: how much of the budget
it reaches, what it assumes, whether it survived its attacks. None of
them priced it in money or in circuits, and on real hardware that turns
out to be the binding constraint far more often than accuracy is.

The numbers below come from real submissions to `qpu.forte-enterprise-1`
in August 2026. Two jobs ran the SAME circuit at 100 and at 500 shots.
The 100-shot job billed $25.79. The 500-shot job billed about the same
-- recorded as roughly $25, with the cents never written down.

That last clause is load-bearing and is carried through the code rather
than rounded away. "The same to the cent" would license a per-shot bound
a hundred times tighter than the evidence supports, which is the exact
move this package exists to catch, so the bound below is computed from
the resolution actually recorded ($1, not $0.01).

That single fact reorganises the whole problem. Cost on that machine is
per-CIRCUIT, not per-shot, so:

  - shots are nearly free once a circuit is paid for, and a budget is
    better spent on fewer circuits with more shots each
  - methods that need many DISTINCT circuits are expensive in a way no
    error budget shows. Post-selection needs no extra circuits at all.
    PEC needs hundreds. They can sit one line apart in a prescription
    ranked on error and differ by three orders of magnitude in price
  - a full 21-slot x 13-group reconstruction is 273 circuits, which at
    the measured rate is about $6,800 whatever the shot count

The methods' circuit multiplicities are structural -- they follow from
what each method has to run, not from any device -- so they are worth
stating once and reusing. What is NOT general is the price: it is one
account, one backend, one month, and it is labelled that way.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Pricing:
    """How a device charges, and where that was learned.

    `per_shot_usd` is an upper BOUND, not a measurement, wherever two
    jobs at different shot counts billed the same: all such a pair can
    say is that the per-shot term is smaller than the resolution of the
    bill. Recording it as exactly zero would be claiming a measurement
    nobody made.
    """

    per_circuit_usd: float
    per_shot_usd_bound: float
    as_of: str
    source: str
    #: The per-shot price where it is actually KNOWN. None means nobody
    #: measured it and `per_shot_usd_bound` is all there is. Keeping
    #: these apart matters: collapsing them into one number either
    #: prices a per-shot machine at zero or inflates a per-circuit one
    #: by a bound nobody was billed.
    per_shot_usd: Optional[float] = None

    def quote(self, circuits: int, shots: int) -> float:
        """What it should cost, using only terms that were measured."""
        return (circuits * self.per_circuit_usd
                + circuits * shots * (self.per_shot_usd or 0.0))

    def upper(self, circuits: int, shots: int) -> float:
        """The most it could cost if an unmeasured term sits at its bound.

        Kept separate from `quote` on purpose. Folding an upper bound
        into a point estimate reports a number larger than anything
        anyone was billed, and then everything downstream quietly
        inherits it.
        """
        if self.per_shot_usd is not None:
            return self.quote(circuits, shots)
        return (circuits * self.per_circuit_usd
                + circuits * shots * self.per_shot_usd_bound)


#: Measured, not published. One account, one backend, one month -- which
#: is exactly as far as it should be trusted. The bound on the per-shot
#: term is (resolution of what was RECORDED) / (difference in shots) =
#: $1.00 / 400, not $0.01 / 400: the second bill was written down as
#: roughly $25 without its cents.
PRICING = {
    "ionq_forte_enterprise": Pricing(
        per_circuit_usd=25.79, per_shot_usd_bound=2.5e-3,
        as_of="2026-08-25",
        source="two real jobs on qpu.forte-enterprise-1 from one account: "
               "100 shots billed $25.79, and 500 shots of the same circuit "
               "billed about the same, recorded to the dollar rather than "
               "the cent -- so the per-shot term is bounded by $1 over a "
               "400-shot difference, not measured"),
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

    def quote(self, pricing: Pricing, shots: int) -> "Quote":
        return Quote(self, pricing.quote(self.circuits, shots),
                     pricing.upper(self.circuits, shots), shots,
                     self.effective_shots(shots))


@dataclass(frozen=True)
class Quote:
    """A price, with what it buys."""

    cost: MethodCost
    usd: float
    usd_upper: float
    shots_paid: int
    shots_kept: int

    @property
    def discarded(self) -> int:
        return self.shots_paid - self.shots_kept

    def describe(self) -> str:
        line = (f"  {self.cost.method:38s} {self.cost.circuits:5d} circuits  "
                f"${self.usd:12,.2f}")
        if self.usd_upper > self.usd * 1.05:
            line += f" (up to ${self.usd_upper:,.2f})"
        if self.discarded:
            line += f"   keeps {self.shots_kept:,}/{self.shots_paid:,} shots"
        return line


#: How many distinct circuits each method submits, for ONE estimate.
#: These are structural: they follow from what the method has to run.
#: Where a method's count depends on a parameter it is a function of it
#: rather than a number, because pretending PEC has a fixed cost is how
#: a prescription recommends something nobody can afford.
def circuits_for(method: str, *, qubits: int = 4, training: int = 5,
                 folds: int = 3, twirls: int = 16,
                 pec_samples: int = 500) -> MethodCost:
    """The circuit count and shot retention of one method by name.

    Names match `prescribe.CATALOGUE` where they overlap, and the
    shootout's implementation names otherwise, so a prescription can be
    priced without a translation table.
    """
    table = {
        "unmitigated": MethodCost(
            "unmitigated", 1, 1.0, "the circuit itself"),
        "more shots": MethodCost(
            "more shots", 1, 1.0,
            "the same circuit; on a per-circuit-priced device this is the "
            "cheapest improvement there is, which is the opposite of the "
            "advice that holds on a per-shot-priced one"),
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
            f"one circuit per noise scale ({folds})"),
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
    pricing: Pricing

    @property
    def affordable(self) -> tuple:
        return tuple(q for q in self.quotes if q.usd <= self.budget_usd)

    @property
    def refused(self) -> tuple:
        return tuple(q for q in self.quotes if q.usd > self.budget_usd)

    def describe(self) -> str:
        lines = [f"  budget ${self.budget_usd:,.2f} at {self.shots:,} shots "
                 f"per circuit, priced at ${self.pricing.per_circuit_usd}/circuit "
                 f"(a bracketed figure is the most it could cost if the "
                 f"unmeasured per-shot term sits at its bound)",
                 f"  ({self.pricing.as_of}: {self.pricing.source})",
                 ""]
        for quote in sorted(self.quotes, key=lambda q: q.usd):
            mark = "    " if quote.usd <= self.budget_usd else "  X "
            lines.append(mark + quote.describe().lstrip())
        if self.refused:
            lines.append("")
            lines.append(f"  {len(self.refused)} of {len(self.quotes)} methods "
                         "cost more than the budget. That is a fact about the "
                         "device's pricing, not about how good they are.")
        cheapest = min(self.quotes, key=lambda q: q.usd)
        headroom = self.budget_usd / max(cheapest.usd, 1e-9)
        lines.append("")
        lines.append(
            f"  Cost here is per-CIRCUIT. The budget buys "
            f"{int(self.budget_usd // self.pricing.per_circuit_usd)} circuits "
            f"at any shot count, so raising shots on {cheapest.cost.method} "
            f"is close to free while adding circuits is not "
            f"({headroom:.1f}x headroom on the cheapest method).")
        return "\n".join(lines)


def affordable_methods(methods, budget_usd: float, shots: int,
                       pricing: Optional[Pricing] = None,
                       **kwargs) -> Affordability:
    """Price a list of method names against a real budget.

    This is the check nobody runs before submitting, and the one whose
    absence costs real money: a prescription ranked purely on which
    error term it reaches will happily recommend a method that needs
    500 circuits on a machine that charges $25.79 for each one.
    """
    pricing = pricing or PRICING["ionq_forte_enterprise"]
    quotes = tuple(circuits_for(name, **kwargs).quote(pricing, shots)
                   for name in methods)
    return Affordability(quotes, budget_usd, shots, pricing)
