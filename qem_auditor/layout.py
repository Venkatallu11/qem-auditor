"""Which qubits to run on, chosen by what is actually hurting you.

The cheapest improvement available to most people is not a mitigation
method. It is running on different qubits. On the fake_kyiv lattice the
two-qubit gate error across neighbouring pairs spans 0.31% to a pair that
is simply dead, and readout error varies by more than 5x between qubits.
Nothing in a mitigation pipeline recovers what a bad placement throws
away, and moving costs nothing but a different `initial_layout`.

The part that is easy to get wrong, and the reason this module takes an
error budget: WHICH property you should optimise depends on which error
is dominating. Picking the lowest-gate-error pair is the obvious move and
it is the wrong one on a device where readout error is 9x the gate error
-- which is the measured case in `examples/real_device_audit.py`. There,
optimising gate error buys almost nothing while optimising readout error
buys most of what is available.

So placements are scored against the budget's own weights. A device where
gate error dominates and a device where readout dominates get different
answers from the same coupling map, which is the whole point.

Stdlib only. The device description is plain numbers, so this works with
any provider's calibration data once it has been read into
`DeviceLayout` -- and `examples/` shows that for IBM's.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Optional

from .prescribe import ErrorBudget, ErrorSource, residual_budget


@dataclass(frozen=True)
class QubitProperties:
    """What a provider publishes per qubit."""

    readout_error: float
    t1_s: Optional[float] = None
    t2_s: Optional[float] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.readout_error <= 1.0:
            raise ValueError(
                f"readout_error={self.readout_error} is not a probability")


@dataclass
class DeviceLayout:
    """A device as this module needs to see it: qubits, edges, and errors.

    `edges` maps an unordered pair to its two-qubit gate error. A pair
    absent from `edges` is not connected, and a circuit needing that
    connection cannot be placed there without routing -- which this
    module does not attempt, because a placement that silently inserts
    swaps is a different circuit and would be scored as though it were
    the one you wrote.
    """

    qubits: dict
    edges: dict
    name: str = "device"

    def __post_init__(self) -> None:
        normalised = {}
        for pair, error in self.edges.items():
            a, b = sorted(pair)
            if a == b:
                raise ValueError(f"edge {pair} connects a qubit to itself")
            for q in (a, b):
                if q not in self.qubits:
                    raise ValueError(
                        f"edge {pair} names qubit {q}, which has no properties")
            normalised[(a, b)] = error
        self.edges = normalised

    def neighbours(self, qubit: int) -> set:
        return {b if a == qubit else a for a, b in self.edges if qubit in (a, b)}

    def gate_error(self, a: int, b: int) -> float:
        pair = tuple(sorted((a, b)))
        if pair not in self.edges:
            raise KeyError(f"qubits {a} and {b} are not connected on {self.name}")
        return self.edges[pair]


@dataclass(frozen=True)
class Placement:
    """A choice of physical qubits, and what it is expected to cost.

    `cost` is in the budget's own units, so it is comparable between
    placements on the same device and meaningless across devices. The
    ratio between two placements' costs is the number worth quoting.
    """

    qubits: tuple
    cost: float
    readout_cost: float
    gate_cost: float
    decoherence_cost: float

    def __str__(self) -> str:
        return f"qubits {self.qubits} (cost {self.cost:.5f})"


def _connected_sets(device: DeviceLayout, size: int,
                    limit: int = 20_000) -> Iterable[tuple]:
    """Every connected set of `size` qubits, up to `limit` of them.

    Exhaustive for the small placements this is used on. The limit is a
    refusal rather than a sample: a truncated search that presented
    itself as a best-of would be claiming an optimum it never looked for.
    """
    if size < 1:
        raise ValueError("a placement needs at least one qubit")
    if size == 1:
        for q in sorted(device.qubits):
            yield (q,)
        return

    seen = set()
    found = 0
    for start in sorted(device.qubits):
        frontier = [(start,)]
        while frontier:
            current = frontier.pop()
            if len(current) == size:
                key = tuple(sorted(current))
                if key not in seen:
                    seen.add(key)
                    found += 1
                    if found > limit:
                        raise OverflowError(
                            f"more than {limit} connected sets of size {size} on "
                            f"{device.name}: this search is exhaustive by design, "
                            "so narrow the candidates rather than trusting a "
                            "truncated best-of")
                    yield key
                continue
            for q in current:
                for n in device.neighbours(q):
                    if n not in current:
                        frontier.append(current + (n,))


def score_placement(qubits: tuple, device: DeviceLayout, budget: ErrorBudget,
                    two_qubit_gates: int = 1, one_qubit_gates: int = 0,
                    circuit_duration_s: Optional[float] = None) -> Placement:
    """Expected error cost of running here, weighted by the budget.

    Each term is the device's contribution to that error source,
    multiplied by how much that source matters in THIS experiment. A
    device property nobody's error budget is sensitive to contributes
    nothing to the score, which is how optimising gate error on a
    readout-dominated device correctly stops looking attractive.
    """
    if len(set(qubits)) != len(qubits):
        raise ValueError(f"placement {qubits} repeats a qubit")
    for q in qubits:
        if q not in device.qubits:
            raise KeyError(f"{device.name} has no qubit {q}")

    readout_share = budget.share(ErrorSource.READOUT)
    gate_share = (budget.share(ErrorSource.GATE_STOCHASTIC)
                  + budget.share(ErrorSource.COHERENT))
    decoherence_share = budget.share(ErrorSource.DECOHERENCE)

    readout = sum(device.qubits[q].readout_error for q in qubits)

    if len(qubits) >= 2:
        # Average over the edges the circuit will actually use. Which
        # edges those are depends on the circuit, so the honest generic
        # answer is the mean over the placement's own connections.
        used = [device.gate_error(a, b) for a, b in combinations(qubits, 2)
                if tuple(sorted((a, b))) in device.edges]
        gate = (sum(used) / len(used) * two_qubit_gates) if used else 0.0
    else:
        gate = 0.0

    decoherence = 0.0
    if circuit_duration_s is not None:
        t2s = [device.qubits[q].t2_s for q in qubits if device.qubits[q].t2_s]
        if t2s:
            decoherence = circuit_duration_s / min(t2s)

    return Placement(
        qubits=tuple(qubits),
        cost=readout_share * readout + gate_share * gate
        + decoherence_share * decoherence,
        readout_cost=readout,
        gate_cost=gate,
        decoherence_cost=decoherence,
    )


def rank_placements(device: DeviceLayout, budget: ErrorBudget, n_qubits: int,
                    **circuit) -> list:
    """Every connected placement, best first."""
    scored = [score_placement(qubits, device, budget, **circuit)
              for qubits in _connected_sets(device, n_qubits)]
    scored.sort(key=lambda p: p.cost)
    return scored


@dataclass(frozen=True)
class LayoutAdvice:
    """Where to run, against where you were going to run.

    `gain` is the ratio of expected costs. It is a prediction from
    calibration data, not a measurement, and the example that uses this
    goes and checks it.
    """

    best: Placement
    current: Optional[Placement]
    worst: Placement
    considered: int
    driven_by: Optional[ErrorSource]
    #: Set when the placement was chosen against what a method will
    #: leave rather than against the raw error.
    after_method: Optional[str] = None

    @property
    def gain(self) -> Optional[float]:
        if self.current is None or self.best.cost <= 0:
            return None
        return self.current.cost / self.best.cost

    @property
    def worth_moving(self) -> bool:
        gain = self.gain
        return gain is not None and gain >= 1.2

    def format_advice(self) -> str:
        lines = [f"  best placement:  {self.best}",
                 f"  worst available: {self.worst}"]
        if self.current is not None:
            lines.insert(1, f"  your placement:  {self.current}")
        driver = self.driven_by.name if self.driven_by else "no dominant source"
        lines.append(f"  chosen for:      {driver}, which dominates this budget")
        if self.after_method:
            lines.append(f"  scored against:  what {self.after_method} leaves "
                         "behind, not the raw error")
        lines.append(f"  searched:        {self.considered} connected placements")
        gain = self.gain
        if gain is None:
            lines.append("  no current placement given, so no comparison is made")
        elif self.worth_moving:
            lines.append(f"  -> moving to {self.best.qubits} is worth about "
                         f"{gain:.2f}x on expected error, and costs nothing but "
                         f"a different initial_layout")
        else:
            lines.append(f"  -> your placement is already within {gain:.2f}x of "
                         "the best available; moving is not worth the churn")
        return "\n".join(lines)


def advise_layout(device: DeviceLayout, budget: ErrorBudget, n_qubits: int,
                  current: Optional[tuple] = None,
                  after_method=None, **circuit) -> LayoutAdvice:
    """The whole recommendation: where to run and whether it is worth moving.

    Pass `after_method` when a mitigation method is already chosen. The
    placement is then scored against what that method will LEAVE, not
    against the raw error -- because a method that removes readout error
    also removes the reason to prefer a low-readout qubit, and the
    optimum moves. Measured: the raw-budget pick wins 13.87 to 36.46
    unmitigated and loses 11.02 to 6.18 once REM is applied.

    The worst placement is reported alongside the best on purpose. A user
    who only sees "use qubits (96, 97)" learns nothing about whether
    placement matters on their device; one who sees best against worst
    can tell at a glance whether this is a free 3x or a rounding error.
    """
    if after_method is not None:
        budget = residual_budget(budget, after_method)
    ranked = rank_placements(device, budget, n_qubits, **circuit)
    if not ranked:
        raise ValueError(
            f"no connected placement of {n_qubits} qubits exists on {device.name}")
    return LayoutAdvice(
        best=ranked[0],
        current=(score_placement(current, device, budget, **circuit)
                 if current is not None else None),
        worst=ranked[-1],
        considered=len(ranked),
        driven_by=budget.dominant if budget.is_decisive else None,
        after_method=after_method.name if after_method is not None else None,
    )
