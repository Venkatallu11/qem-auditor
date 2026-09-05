"""What a run leaves behind when nobody knows the right answer.

The ledger already records outcomes, but `Observation` needs `raw_error`
and `mitigated_error` -- it needs the TRUTH. That is fine for a benchmark
molecule whose energy is known and useless for the person this project is
most for, who ran something on hardware precisely because nobody knows
what it should give.

So a corpus entry is a different kind of thing, and the difference is the
whole design: **an encounter records what a method DID, never whether it
helped.** Without ground truth, "helped" is unavailable, and a corpus
that inferred it anyway would be manufacturing the conclusion this
package exists to withhold.

What is available, and is recorded:

  * the shot-noise floor, which follows from the counts alone;
  * how far each method MOVED the estimate away from unmitigated;
  * what each method cost in amplified shot noise;
  * whether the claimed uncertainty was below the floor.

None of that ranks methods. What it does is build an expectation, and an
expectation is what turns a single run into evidence: when readout
mitigation on your circuit shifts the answer ten times further than it
has on forty comparable runs, that is worth knowing, and no amount of
staring at one run reveals it. The corpus is an anomaly detector, not a
leaderboard.

Two rules hold it to that:

**It refuses to generalise from too little.** Below `MINIMUM_ENCOUNTERS`
comparable runs, recall returns what it has and declines to summarise.

**It stays on your machine.** Entries are plain JSON in the local store,
written only when you ask. Nothing here transmits anything anywhere, and
the counts themselves are never recorded -- only the derived quantities
above, so a corpus cannot leak the data it learned from.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from statistics import median
from typing import Optional

from .provenance import hash_json

#: Comparable runs needed before the corpus will summarise rather than
#: merely list. Five is not a statistical threshold, it is a refusal to
#: let two coincidences look like a pattern -- the same bar
#: `EvidenceLedger.supports_a_ranking` uses.
MINIMUM_ENCOUNTERS = 5

#: How close two runs must be, in width and shot count, to inform each
#: other. Deliberately coarse: this is "runs like yours", not a join key.
WIDTH_TOLERANCE = 2


@dataclass(frozen=True)
class MethodEffect:
    """What one method did to one estimate. Not whether it was right."""

    method: str
    estimate: float
    shift: float
    amplification: Optional[float] = None

    def to_dict(self) -> dict:
        return {"method": self.method, "estimate": self.estimate,
                "shift": self.shift, "amplification": self.amplification}

    @classmethod
    def from_dict(cls, data: dict) -> "MethodEffect":
        return cls(method=data["method"], estimate=data["estimate"],
                   shift=data["shift"], amplification=data.get("amplification"))


@dataclass(frozen=True)
class Encounter:
    """One analysis of somebody's measurement data, without a known answer.

    Carries no counts and no bitstrings -- only derived quantities. A
    corpus that stored the raw data would be a copy of everyone's
    experiments, and this is meant to be safe to keep.
    """

    n_qubits: int
    settings: int
    shots: int
    floor: float
    unmitigated: float
    effects: tuple = ()
    claimed_uncertainty: Optional[float] = None
    claim_impossible: bool = False
    device: str = "unstated"
    note: str = ""

    def __post_init__(self) -> None:
        if self.shots <= 0:
            raise ValueError("an encounter with no shots records nothing")
        if self.floor < 0:
            raise ValueError("a shot-noise floor cannot be negative")

    @property
    def digest(self) -> str:
        """Content address, so recording one run twice does not make it
        twice as convincing."""
        return hash_json(self.to_dict())[:16]

    def resembles(self, other: "Encounter") -> bool:
        """Is this the kind of run that other one can speak to?

        Width and order-of-magnitude shot count. Not the observable, not
        the device -- those narrow the corpus to nothing long before it
        is large enough to say anything.
        """
        if abs(self.n_qubits - other.n_qubits) > WIDTH_TOLERANCE:
            return False
        smaller, larger = sorted((self.shots, other.shots))
        return larger <= 10 * smaller

    def effect_of(self, method: str) -> Optional[MethodEffect]:
        for effect in self.effects:
            if effect.method == method:
                return effect
        return None

    def to_dict(self) -> dict:
        return {
            "n_qubits": self.n_qubits, "settings": self.settings,
            "shots": self.shots, "floor": self.floor,
            "unmitigated": self.unmitigated,
            "effects": [effect.to_dict() for effect in self.effects],
            "claimed_uncertainty": self.claimed_uncertainty,
            "claim_impossible": self.claim_impossible,
            "device": self.device, "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Encounter":
        return cls(
            n_qubits=data["n_qubits"], settings=data["settings"],
            shots=data["shots"], floor=data["floor"],
            unmitigated=data["unmitigated"],
            effects=tuple(MethodEffect.from_dict(e) for e in data.get("effects", ())),
            claimed_uncertainty=data.get("claimed_uncertainty"),
            claim_impossible=data.get("claim_impossible", False),
            device=data.get("device", "unstated"), note=data.get("note", ""))


def encounter_from_report(report, n_qubits: int, device: str = "unstated",
                          note: str = "") -> Encounter:
    """Turn a `results.ResultsReport` into something the corpus can keep."""
    unmitigated = report.noise.estimate
    effects = []
    for estimate in report.estimates:
        if estimate.method == "unmitigated":
            continue
        amplification = None
        if estimate.floor and report.noise.floor(report.confidence) > 0:
            amplification = estimate.floor / report.noise.floor(report.confidence)
        effects.append(MethodEffect(
            method=estimate.method, estimate=estimate.estimate,
            shift=estimate.estimate - unmitigated, amplification=amplification))
    return Encounter(
        n_qubits=n_qubits, settings=report.noise.settings,
        shots=report.noise.shots, floor=report.noise.floor(report.confidence),
        unmitigated=unmitigated, effects=tuple(effects),
        claimed_uncertainty=report.claimed_uncertainty,
        claim_impossible=report.claim_is_impossible,
        device=device, note=note)


@dataclass(frozen=True)
class Expectation:
    """What comparable runs did, and whether that is enough to say so."""

    method: str
    n: int
    typical_shift: Optional[float] = None
    typical_amplification: Optional[float] = None

    @property
    def worth_relying_on(self) -> bool:
        return self.n >= MINIMUM_ENCOUNTERS

    def surprise(self, shift: float) -> Optional[float]:
        """How many times the typical shift this one is. None when there
        is no typical shift to compare against."""
        if not self.worth_relying_on or not self.typical_shift:
            return None
        return abs(shift) / abs(self.typical_shift)

    def describe(self) -> str:
        if self.n == 0:
            return f"  {self.method}: no comparable runs recorded"
        if not self.worth_relying_on:
            return (f"  {self.method}: {self.n} comparable run"
                    f"{'s' if self.n != 1 else ''}, too few to summarise "
                    f"(need {MINIMUM_ENCOUNTERS})")
        line = (f"  {self.method}: over {self.n} comparable runs it typically "
                f"moved the estimate by {self.typical_shift:+.4g}")
        if self.typical_amplification:
            line += f", costing {self.typical_amplification:.2f}x in shot noise"
        return line


@dataclass
class Corpus:
    """Encounters, deduplicated by content, and what they add up to."""

    encounters: list = field(default_factory=list)
    _seen: set = field(default_factory=set, repr=False)

    def record(self, encounter: Encounter) -> bool:
        """Append unless this exact run is already here. Returns whether
        it was new."""
        if encounter.digest in self._seen:
            return False
        self._seen.add(encounter.digest)
        self.encounters.append(encounter)
        return True

    def __len__(self) -> int:
        return len(self.encounters)

    def similar_to(self, encounter: Encounter) -> list:
        return [other for other in self.encounters
                if other.digest != encounter.digest and other.resembles(encounter)]

    def expectation(self, method: str, like: Encounter) -> Expectation:
        """What comparable runs did with this method."""
        effects = [other.effect_of(method) for other in self.similar_to(like)]
        effects = [effect for effect in effects if effect is not None]
        if not effects:
            return Expectation(method=method, n=0)
        amplifications = [e.amplification for e in effects if e.amplification]
        return Expectation(
            method=method, n=len(effects),
            typical_shift=median(e.shift for e in effects),
            typical_amplification=median(amplifications) if amplifications else None)

    def report_on(self, encounter: Encounter) -> str:
        """What the corpus can say about this run. Often: not enough yet."""
        comparable = self.similar_to(encounter)
        lines = [f"  corpus: {len(self)} encounter"
                 f"{'s' if len(self) != 1 else ''} recorded, "
                 f"{len(comparable)} comparable to this one"]
        if not comparable:
            lines.append("  nothing to compare against yet -- this run becomes "
                         "the first data point for the next one")
            return "\n".join(lines)

        for effect in encounter.effects:
            expectation = self.expectation(effect.method, encounter)
            lines.append(expectation.describe())
            surprise = expectation.surprise(effect.shift)
            if surprise is not None and (surprise > 3 or surprise < 1 / 3):
                lines.append(
                    f"    -> this run moved {effect.shift:+.4g}, "
                    f"{surprise:.1f}x the usual. Worth a look: not wrong, "
                    "but not typical either")
        return "\n".join(lines)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps([e.to_dict() for e in self.encounters], indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "Corpus":
        corpus = cls()
        for entry in json.loads(text):
            corpus.record(Encounter.from_dict(entry))
        return corpus

    def save(self, path) -> None:
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path) -> "Corpus":
        from pathlib import Path
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            return cls.from_json(path.read_text())
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # A corrupt corpus is not worth crashing an audit over, and
            # silently starting fresh is better than pretending to
            # remember something unreadable.
            return cls()
