"""What this circuit reminds the auditor of.

`ledger.py` remembers how methods PERFORMED, keyed on where the error
was. This remembers what was FOUND, keyed on the circuit itself -- so
that a circuit arriving today can be met with "the last three things
shaped like this failed the compiler check, so look there first" instead
of a fresh start every time.

The two are different questions and both are worth asking. A budget says
what will help. A circuit says what went wrong last time, which is the
better predictor of what will go wrong this time: the same ansatz
compiled the same way tends to break the same way, and an auditor that
cannot notice that makes every user rediscover it.

What memory is allowed to do here is REORDER and WARN. It is not allowed
to conclude. A circuit resembling three that were INVALID does not make
this one invalid -- the gates still decide that, on this circuit's own
evidence. Precedent that could convict would be the worst feature in this
package: it would make a method that failed once unable to be shown
working, which is how a tool stops being an auditor and starts being a
reputation system.

So recall produces three things, all of them advisory:

  - identity: this exact circuit has been audited before, and here is
    what happened
  - resemblance: circuits like it, and what failed in them
  - priority: which checks and attacks earned their keep on those, so
    the expensive ones can go first instead of last

Stdlib only. Fingerprints are plain numbers read off a circuit, so any
provider's circuits work once `fingerprint_from_spec` has seen them.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .provenance import hash_json
from .schema import CircuitSpec, Experiment, FailureMode
from .verdict import Verdict


@dataclass(frozen=True)
class CircuitFingerprint:
    """A structural signature: what a circuit IS, not what it is called.

    Names are the obvious key and the wrong one. Two groups running the
    same UCCSD ansatz call it different things, and one group calls two
    different circuits by the same name across a refactor. Structure is
    what actually predicts how something breaks.

    Depth and gate counts are deliberately kept as raw numbers rather
    than bucketed: bucketing decides in advance what counts as similar,
    and `resembles` can make that judgement with the numbers in hand.
    """

    n_qubits: int
    two_qubit_gates: int
    one_qubit_gates: int
    depth: int
    gate_names: tuple = ()
    observable_terms: int = 0
    measurement_bases: int = 1
    family: str = ""

    def __post_init__(self) -> None:
        for name in ("n_qubits", "two_qubit_gates", "one_qubit_gates", "depth"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.n_qubits == 0:
            raise ValueError("a circuit on no qubits has nothing to fingerprint")
        object.__setattr__(self, "gate_names", tuple(sorted(set(self.gate_names))))

    @property
    def digest(self) -> str:
        """Names this exact structure. Two circuits with the same digest
        are structurally identical, whatever they are called."""
        return hash_json({
            "n_qubits": self.n_qubits, "two_qubit_gates": self.two_qubit_gates,
            "one_qubit_gates": self.one_qubit_gates, "depth": self.depth,
            "gate_names": list(self.gate_names),
            "observable_terms": self.observable_terms,
            "measurement_bases": self.measurement_bases,
        })

    def resembles(self, other: "CircuitFingerprint") -> float:
        """How alike two circuits are, on [0, 1].

        Weighted towards the things that predict how a circuit fails.
        Two-qubit gate count and depth dominate because they drive both
        the error budget and the compiler's freedom to rewrite; the gate
        alphabet matters because a circuit built from different
        primitives meets a different transpiler; qubit count matters
        least, since a 4-qubit and a 6-qubit version of the same ansatz
        fail the same way.
        """
        def ratio(a: int, b: int) -> float:
            if a == b:
                return 1.0
            hi = max(a, b)
            return 1.0 - abs(a - b) / hi if hi else 1.0

        alphabet = (len(set(self.gate_names) & set(other.gate_names))
                    / len(set(self.gate_names) | set(other.gate_names))
                    if (self.gate_names or other.gate_names) else 1.0)
        score = (0.30 * ratio(self.two_qubit_gates, other.two_qubit_gates)
                 + 0.25 * ratio(self.depth, other.depth)
                 + 0.20 * alphabet
                 + 0.10 * ratio(self.one_qubit_gates, other.one_qubit_gates)
                 + 0.10 * ratio(self.n_qubits, other.n_qubits)
                 + 0.05 * ratio(self.observable_terms, other.observable_terms))
        # A declared family is a strong hint, but it is the claimant's
        # word, so it adjusts rather than decides. The bonus closes part
        # of the remaining gap rather than adding a flat amount, so it
        # can approach 1.0 and never reach it: circuits of different
        # depth reported as "100% alike" is an over-claim, and it was
        # doing exactly that before this was written as a fraction of
        # what is left rather than a constant.
        if self.family and other.family:
            score = (score + (1.0 - score) * 0.15 if self.family == other.family
                     else score * 0.9)
        return score

    def to_dict(self) -> dict:
        return {"n_qubits": self.n_qubits, "two_qubit_gates": self.two_qubit_gates,
                "one_qubit_gates": self.one_qubit_gates, "depth": self.depth,
                "gate_names": list(self.gate_names),
                "observable_terms": self.observable_terms,
                "measurement_bases": self.measurement_bases, "family": self.family}

    @classmethod
    def from_dict(cls, data: dict) -> "CircuitFingerprint":
        return cls(**{**data, "gate_names": tuple(data.get("gate_names", ()))})


def fingerprint_from_spec(spec: CircuitSpec, depth: Optional[int] = None,
                          observable_terms: int = 0,
                          measurement_bases: int = 1,
                          family: str = "") -> CircuitFingerprint:
    """Build a fingerprint from a record's CircuitSpec.

    `depth` is not on CircuitSpec, so it is taken separately and defaults
    to the two-qubit gate count -- a floor rather than a guess, since a
    circuit cannot be shallower than the gates it must sequence.
    """
    two_qubit = spec.n_2q_gates or 0
    return CircuitFingerprint(
        n_qubits=spec.n_qubits or 1,
        two_qubit_gates=two_qubit,
        one_qubit_gates=spec.n_1q_gates or 0,
        depth=depth if depth is not None else two_qubit,
        gate_names=tuple(g.strip() for g in spec.native_gate_set.split(",") if g.strip()),
        observable_terms=observable_terms,
        measurement_bases=measurement_bases,
        family=family,
    )


@dataclass(frozen=True)
class PastCase:
    """One circuit this auditor has seen, and what it found in it."""

    experiment_id: str
    fingerprint: CircuitFingerprint
    verdict: Verdict
    failed_gates: tuple = ()
    unrun_gates: tuple = ()
    failure_modes: tuple = ()
    attacks_that_fired: tuple = ()
    note: str = ""

    @property
    def digest(self) -> str:
        return hash_json({"experiment_id": self.experiment_id,
                          "circuit": self.fingerprint.digest,
                          "verdict": self.verdict.name,
                          "failed": sorted(self.failed_gates)})

    def to_dict(self) -> dict:
        return {"experiment_id": self.experiment_id,
                "fingerprint": self.fingerprint.to_dict(),
                "verdict": self.verdict.name,
                "failed_gates": list(self.failed_gates),
                "unrun_gates": list(self.unrun_gates),
                "failure_modes": [m.name for m in self.failure_modes],
                "attacks_that_fired": list(self.attacks_that_fired),
                "note": self.note}

    @classmethod
    def from_dict(cls, data: dict) -> "PastCase":
        return cls(
            experiment_id=data["experiment_id"],
            fingerprint=CircuitFingerprint.from_dict(data["fingerprint"]),
            verdict=Verdict[data["verdict"]],
            failed_gates=tuple(data.get("failed_gates", ())),
            unrun_gates=tuple(data.get("unrun_gates", ())),
            failure_modes=tuple(FailureMode[m] for m in data.get("failure_modes", ())),
            attacks_that_fired=tuple(data.get("attacks_that_fired", ())),
            note=data.get("note", ""))


def case_from_audit(exp: Experiment, report, fingerprint: CircuitFingerprint,
                    analysis=None, attacks_that_fired=()) -> PastCase:
    """Build a case from an audit that just happened."""
    return PastCase(
        experiment_id=exp.experiment_id,
        fingerprint=fingerprint,
        verdict=report.verdict,
        failed_gates=tuple(g.name for g in report.gate_results if g.passed is False),
        unrun_gates=tuple(g.name for g in report.gate_results if g.passed is None),
        failure_modes=tuple(d.mode for d in analysis.diagnoses) if analysis else (),
        attacks_that_fired=tuple(attacks_that_fired),
    )


@dataclass(frozen=True)
class Recollection:
    """What memory has to offer about a circuit arriving now.

    Everything here is advisory. `seen_before` and `resembling` are
    reports of what happened; `check_first` and `attacks_first` are
    orderings. None of it is a verdict, and the gates never see it --
    a circuit resembling three that failed is not thereby failing, and
    precedent that could convict would turn an auditor into a reputation
    system where a method that failed once could never be shown working.
    """

    seen_before: tuple = ()
    resembling: tuple = ()
    check_first: tuple = ()
    attacks_first: tuple = ()

    @property
    def is_empty(self) -> bool:
        return not (self.seen_before or self.resembling)

    @property
    def n_similar(self) -> int:
        return len(self.resembling)

    @property
    def worth_relying_on(self) -> bool:
        """Three prior cases are three prior cases.

        Below this the recollection is printed and not acted on, which is
        the same threshold `ledger.py` applies for the same reason.
        """
        return self.n_similar >= 3

    def format_recollection(self) -> str:
        if self.is_empty:
            return ("  Nothing in memory resembles this circuit. That is not "
                    "reassurance -- it is the first time, and the checks below "
                    "are all the evidence there is.")
        lines = []
        if self.seen_before:
            lines.append("  This exact circuit structure has been audited before:")
            for case in self.seen_before:
                lines.append(f"    {case.experiment_id}: {case.verdict.value}")
                if case.failed_gates:
                    lines.append(f"      failed: {', '.join(case.failed_gates)}")
        if self.resembling:
            lines.append(f"  {len(self.resembling)} similar circuit"
                         f"{'s' if len(self.resembling) != 1 else ''} in memory:")
            for case, score in self.resembling[:5]:
                summary = (", ".join(case.failed_gates) if case.failed_gates
                           else "nothing failed")
                lines.append(f"    {case.experiment_id} ({score:.0%} alike): "
                             f"{case.verdict.value} -- {summary}")
        if self.check_first:
            lines.append("  Check these first, they failed most often on circuits "
                         "like this one:")
            for name, hits, total in self.check_first:
                lines.append(f"    {name}: failed {hits}/{total}")
        if self.attacks_first:
            lines.append("  Attacks that earned their keep here before:")
            for name, hits, total in self.attacks_first:
                lines.append(f"    {name}: found something {hits}/{total} times")
        if not self.worth_relying_on:
            lines.append("  Too few prior cases to reorder anything on. Reported "
                         "so you can look, not relied on.")
        lines.append("  Memory advises. The gates still decide this circuit on its "
                     "own evidence.")
        return "\n".join(lines)


@dataclass
class CaseMemory:
    """Everything this auditor has seen, and what it found."""

    cases: list = field(default_factory=list)
    _digests: set = field(default_factory=set, repr=False)

    def remember(self, case: PastCase) -> bool:
        if case.digest in self._digests:
            return False
        self._digests.add(case.digest)
        self.cases.append(case)
        return True

    def __len__(self) -> int:
        return len(self.cases)

    def recall(self, fingerprint: CircuitFingerprint,
               threshold: float = 0.75) -> Recollection:
        """What memory has about a circuit like this one."""
        exact = tuple(c for c in self.cases
                      if c.fingerprint.digest == fingerprint.digest)
        scored = [(c, fingerprint.resembles(c.fingerprint)) for c in self.cases
                  if c.fingerprint.digest != fingerprint.digest]
        similar = tuple(sorted([(c, s) for c, s in scored if s >= threshold],
                               key=lambda cs: -cs[1]))

        pool = list(exact) + [c for c, _ in similar]
        return Recollection(
            seen_before=exact,
            resembling=similar,
            check_first=_rank_by_hits(pool, lambda c: c.failed_gates),
            attacks_first=_rank_by_hits(pool, lambda c: c.attacks_that_fired),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps([c.to_dict() for c in self.cases], indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "CaseMemory":
        memory = cls()
        for entry in json.loads(text):
            memory.remember(PastCase.from_dict(entry))
        return memory

    def save(self, path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path) -> "CaseMemory":
        p = Path(path)
        return cls.from_json(p.read_text()) if p.exists() else cls()


def _rank_by_hits(cases: list, extract) -> tuple:
    """Order things by how often they fired, most productive first.

    Ties break on the name so the ordering is stable across runs. An
    ordering that shuffled between invocations would send a user to a
    different check each time for no reason.
    """
    total = len(cases)
    if not total:
        return ()
    counts = {}
    for case in cases:
        for name in set(extract(case)):
            counts[name] = counts.get(name, 0) + 1
    return tuple(sorted(((name, hits, total) for name, hits in counts.items()),
                        key=lambda row: (-row[1], row[0])))
