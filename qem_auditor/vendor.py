"""What the VENDOR says was done to your data.

Every control in this package so far is something the submitter asserts
and the auditor checks against their own data. There is a third party in
any hardware run whose account nobody reads: the provider's own job
record. It states, in machine-readable form, what the service did to the
circuit and to the counts before they reached you.

IonQ's record carries an `error_mitigation` block naming whether
debiasing and symmetry verification were applied, a `compilation` block
naming the gate basis and precision the service compiled to, and the
gate counts of the circuit as EXECUTED rather than as written. All three
can contradict the experiment record.

The contradiction that matters most is the first. A submitter can
believe in good faith that they collected an unmitigated baseline while
the service quietly debiased it, and every mitigation gain measured
against that baseline is then measured against something already
mitigated. No amount of care with the circuit catches that; it is
visible only in the provider's record, which is why this module exists.

The second matters for a different reason. Compilation to a native
basis changes gate counts, and an error budget built on the gates
someone WROTE describes a circuit the machine never ran. On the real
job this module was built against, a 5-qubit circuit executed 120
one-qubit gates against 11 two-qubit ones -- an 11:1 ratio, on hardware
where one-qubit gates are routinely assumed negligible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class VendorRecordError(ValueError):
    """Raised when a job record cannot be read as claimed."""


@dataclass(frozen=True)
class JobRecord:
    """The provider's own account of one submission."""

    job_id: str
    backend: str
    shots: int
    gate_counts: dict
    debiasing: Optional[bool]
    symmetry_verification: Optional[bool]
    gate_basis: Optional[str]
    cost_model: Optional[str]
    execution_ms: Optional[int]
    predicted_ms: Optional[int]
    counts: Optional[dict]

    @property
    def one_qubit_gates(self) -> int:
        return _width_count(self.gate_counts, "1q", _ONE_QUBIT)

    @property
    def two_qubit_gates(self) -> int:
        return _width_count(self.gate_counts, "2q", _TWO_QUBIT)

    @property
    def vendor_mitigated(self) -> bool:
        """Did the SERVICE apply mitigation, whatever the submitter did?"""
        return bool(self.debiasing) or bool(self.symmetry_verification)

    def describe(self) -> str:
        lines = [f"  job {self.job_id} on {self.backend}, {self.shots:,} shots"]
        if self.gate_counts:
            one, two = self.one_qubit_gates, self.two_qubit_gates
            ratio = f", {one / two:.0f}:1" if two else ""
            lines.append(
                f"  as EXECUTED: {one} one-qubit, {two} two-qubit{ratio}"
                + (f" ({self.gate_basis} basis)" if self.gate_basis else ""))
        else:
            # Never fall back to the submitter's own gate counts here.
            # The whole point of this record is that it is the vendor's
            # account, and a batch job's children carry no stats block.
            lines.append("  the job record states no gate counts for this "
                         "job, so the executed circuit cannot be checked "
                         "against the one that was written")
        applied = []
        if self.debiasing:
            applied.append("debiasing")
        if self.symmetry_verification:
            applied.append("symmetry verification")
        lines.append("  vendor-applied mitigation: "
                     + (", ".join(applied) if applied
                        else "none declared in the job record"))
        if self.execution_ms and self.predicted_ms:
            lines.append(f"  execution {self.execution_ms:,}ms against "
                         f"{self.predicted_ms:,}ms predicted "
                         f"({self.execution_ms / self.predicted_ms:.2f}x)")
        return "\n".join(lines)


_ONE_QUBIT = {"gpi", "gpi2", "rx", "ry", "rz", "x", "y", "z", "h", "s", "sx",
              "sdg", "t", "tdg", "u", "u1", "u2", "u3", "virtual_z"}
_TWO_QUBIT = {"zz", "ms", "cx", "cz", "cnot", "ecr", "iswap", "rzz", "xx"}


def _width_count(gate_counts: dict, aggregate: str, names: set) -> int:
    """Gates of one width, without counting any of them twice.

    IonQ's own record carries BOTH an aggregate and a per-gate breakdown
    in the same mapping -- `{"1q": 120, "2q": 11, "GPIq": 40,
    "GPI2q": 80, "ZZq": 11}` -- so summing every key that looks like a
    one-qubit gate returns 240 for a circuit that ran 120. The
    aggregate is authoritative where it exists; the per-gate names are
    for records (a transpiled qiskit circuit's `count_ops`, say) that
    carry no aggregate at all. Mixing the two is the error, so this
    never does.
    """
    if aggregate in gate_counts:
        return int(gate_counts[aggregate])
    return sum(int(n) for gate, n in gate_counts.items()
               if gate.lower().rstrip("q") in names or gate.lower() in names)


def _dig(record: dict, *path):
    node = record
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def ionq_job(record: dict, counts: Optional[dict] = None) -> JobRecord:
    """Read an IonQ job record as returned by the jobs API.

    Accepts either the raw job metadata or a wrapper holding it under
    `raw_job_metadata` / `parent_raw_job_metadata`, which is how these
    records are usually saved alongside the counts they produced.

    A multi-circuit job keeps its mitigation and compilation blocks on
    the CHILD jobs, not the parent, so the parent's empty `output` is
    read from the first child rather than reported as "nothing applied"
    -- reading absence as a declaration of no mitigation would invert
    the one check this module exists to make.
    """
    if not isinstance(record, dict):
        raise VendorRecordError("a job record must be a mapping")
    for wrapper in ("raw_job_metadata", "parent_raw_job_metadata"):
        if wrapper in record:
            counts = counts if counts is not None else _first_counts(record)
            record = record[wrapper]
            break

    output = record.get("output") or {}
    children = record.get("children_raw_job_metadata") or []
    if not output and children:
        output = children[0].get("output") or {}
    mitigation = output.get("error_mitigation")
    compilation = output.get("compilation") or {}

    if mitigation is None:
        debiasing = symmetry = None
    else:
        debiasing = mitigation.get("debiasing")
        symmetry = _dig(mitigation, "symmetry_verification", "applied")

    stats = record.get("stats") or {}
    gate_counts = stats.get("gate_counts") or {}
    if not gate_counts and children:
        gate_counts = _dig(children[0], "stats", "gate_counts") or {}

    return JobRecord(
        job_id=str(record.get("id", "unknown")),
        backend=str(record.get("backend", "unknown")),
        shots=int(record.get("shots") or 0),
        gate_counts=dict(gate_counts),
        debiasing=debiasing,
        symmetry_verification=symmetry,
        gate_basis=compilation.get("gate_basis"),
        cost_model=record.get("cost_model"),
        execution_ms=record.get("execution_duration_ms"),
        predicted_ms=record.get("predicted_execution_duration_ms"),
        counts=counts,
    )


def _first_counts(wrapper: dict) -> Optional[dict]:
    if isinstance(wrapper.get("counts"), dict):
        return wrapper["counts"]
    listed = wrapper.get("counts_list")
    if isinstance(listed, list) and listed and isinstance(listed[0], dict):
        return listed[0]
    return None


@dataclass(frozen=True)
class Contradiction:
    """Where the record and the vendor disagree, and which to believe."""

    field: str
    claimed: str
    vendor: str
    reading: str

    def describe(self) -> str:
        return (f"  {self.field}: record says {self.claimed}, vendor job "
                f"says {self.vendor}\n      -> {self.reading}")


def cross_check(job: JobRecord, *,
                claims_unmitigated: Optional[bool] = None,
                declared_shots: Optional[int] = None,
                declared_two_qubit_gates: Optional[int] = None,
                declared_one_qubit_gates: Optional[int] = None) -> tuple:
    """Compare an experiment's own claims against the provider's record.

    The vendor's record wins every disagreement here, and the reading
    says so. It is the only account of the run written by the party that
    actually performed it.
    """
    found = []

    if claims_unmitigated and job.vendor_mitigated:
        applied = [n for n, on in (("debiasing", job.debiasing),
                                   ("symmetry verification",
                                    job.symmetry_verification)) if on]
        found.append(Contradiction(
            "mitigation", "this is an unmitigated baseline",
            ", ".join(applied) + " was applied by the service",
            "every gain measured against this baseline is measured "
            "against data that was already mitigated, so the improvement "
            "attributed to the method is not the method's alone"))

    if declared_shots is not None and job.shots and job.shots != declared_shots:
        found.append(Contradiction(
            "shots", f"{declared_shots}", f"{job.shots}",
            "the shot-noise floor is computed from the shot count, so the "
            "quoted interval belongs to a different experiment than the "
            "one that ran"))

    for label, declared, executed in (
            ("two-qubit gates", declared_two_qubit_gates, job.two_qubit_gates),
            ("one-qubit gates", declared_one_qubit_gates, job.one_qubit_gates)):
        if declared is None or not executed:
            continue
        if declared != executed:
            found.append(Contradiction(
                label, f"{declared}", f"{executed} after compilation",
                "an error budget built on the gates as written describes a "
                "circuit the machine never ran; compilation to a native "
                "basis is where the two diverge"))
    return tuple(found)


# --- IBM: the mitigation you get without asking -----------------------
#
# IonQ's record says what WAS done, after the fact. IBM's Estimator
# poses a different and sharper problem: it applies error mitigation BY
# DEFAULT, before anyone asks for any, and the defaults are resolved
# server-side from a single `resilience_level` number that most users
# never set.
#
# The rules below are transcribed from the installed
# qiskit-ibm-runtime's own option docstrings, quoted in the tests so a
# version bump that changes them fails loudly rather than leaving this
# module quietly describing last year's behaviour. They are NOT
# observed from a real IBM job -- unlike the IonQ reader, which is
# built against one -- and that difference is the reason the defaults
# are pinned by a test that reads the library rather than by a fixture.

#: EstimatorOptions.resilience_level, when the user sets nothing.
#: "Default: 1" -- and level 1 is "Mitigate error associated with
#: readout errors", which is mitigation nobody in that path asked for.
IBM_DEFAULT_RESILIENCE_LEVEL = 1

#: The level at or above which each option turns itself on, when the
#: user leaves it Unset. From the docstrings: measure_mitigation and
#: measurement twirling are False only at level 0; gate twirling and ZNE
#: are False at levels 0 and 1 and True at level 2; PEC defaults to
#: False at every level and has to be asked for.
_RESOLVES_AT = {
    "readout_mitigation": 1,
    "measurement_twirling": 1,
    "gate_twirling": 2,
    "zne": 2,
}


@dataclass(frozen=True)
class AppliedMitigation:
    """What an IBM Estimator will apply, including what you did not ask for."""

    resilience_level: int
    readout_mitigation: bool
    measurement_twirling: bool
    gate_twirling: bool
    zne: bool
    pec: bool
    #: Which of the above the submitter set explicitly. Everything else
    #: was resolved from the resilience level by the service.
    explicit: tuple
    #: True when the level itself was left to the default.
    level_defaulted: bool

    @property
    def applied(self) -> tuple:
        return tuple(name for name, on in (
            ("readout error mitigation", self.readout_mitigation),
            ("measurement twirling", self.measurement_twirling),
            ("gate twirling", self.gate_twirling),
            ("ZNE", self.zne),
            ("PEC", self.pec)) if on)

    @property
    def unrequested(self) -> tuple:
        """Applied, but never asked for -- the part that surprises people."""
        asked = {
            "readout error mitigation": "readout_mitigation",
            "measurement twirling": "measurement_twirling",
            "gate twirling": "gate_twirling",
            "ZNE": "zne",
            "PEC": "pec",
        }
        return tuple(name for name in self.applied
                     if asked[name] not in self.explicit)

    def describe(self) -> str:
        level = (f"resilience_level {self.resilience_level}"
                 + (" (the default -- nobody set it)" if self.level_defaulted
                    else " (set explicitly)"))
        lines = [f"  {level}"]
        if not self.applied:
            lines.append("  nothing is mitigated: this really is a raw run")
            return "\n".join(lines)
        lines.append("  the service applies: " + ", ".join(self.applied))
        if self.unrequested:
            lines.append("  never requested, resolved from the level: "
                         + ", ".join(self.unrequested))
        return "\n".join(lines)


def _flag(options: dict, *path):
    """Read a possibly-nested option, treating absence as Unset.

    Options arrive as plain dicts from a saved payload and as nested
    option objects from a live Estimator, so both shapes are read the
    same way rather than one being blessed.
    """
    node = options
    for key in path:
        if hasattr(node, key):
            node = getattr(node, key)
        elif isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return None
        if node is None:
            return None
    if isinstance(node, bool):
        return node
    # qiskit-ibm-runtime's sentinel for "let the server decide".
    if type(node).__name__ == "UnsetType":
        return None
    return node if isinstance(node, (int, float, str)) else None


def ibm_estimator_mitigation(options=None) -> AppliedMitigation:
    """What an IBM Estimator run applies, given the options it was handed.

    Pass `estimator.options`, or the options mapping from a saved job
    payload, or nothing at all to see what a bare
    `Estimator(mode=backend)` does.

    The answer to that last case is the point of this function: readout
    error mitigation AND measurement twirling, both on, because
    `resilience_level` defaults to 1 and both resolve to True there. A
    result collected that way is not an unmitigated baseline, however
    firmly the person who collected it believes it is -- and nothing in
    their own code says otherwise, because they never wrote the line
    that turned mitigation on.
    """
    options = options if options is not None else {}
    level = _flag(options, "resilience_level")
    level_defaulted = level is None
    if level_defaulted:
        level = IBM_DEFAULT_RESILIENCE_LEVEL
    level = int(level)

    explicit = []
    resolved = {}
    for name, path in (
            ("readout_mitigation", ("resilience", "measure_mitigation")),
            ("zne", ("resilience", "zne_mitigation")),
            ("pec", ("resilience", "pec_mitigation")),
            ("measurement_twirling", ("twirling", "enable_measure")),
            ("gate_twirling", ("twirling", "enable_gates"))):
        setting = _flag(options, *path)
        if isinstance(setting, bool):
            explicit.append(name)
            resolved[name] = setting
            continue
        # PEC is the one that never turns itself on: "Default: False",
        # with no level that flips it.
        resolved[name] = (False if name == "pec"
                          else level >= _RESOLVES_AT[name])

    return AppliedMitigation(
        resilience_level=level, explicit=tuple(explicit),
        level_defaulted=level_defaulted, **resolved)


def cross_check_ibm(mitigation: AppliedMitigation, *,
                    claims_unmitigated: Optional[bool] = None) -> tuple:
    """Compare an IBM run's claims against what the service will apply.

    Separate from `cross_check` because the evidence is different in
    kind: IonQ's record reports what happened, while this reasons from
    the submitted options and the service's documented resolution of
    them. Same conclusion, weaker evidence, and the two should not be
    presented as though they were the same thing.
    """
    if not claims_unmitigated or not mitigation.applied:
        return ()
    detail = ", ".join(mitigation.applied)
    reading = (
        "every gain measured against this baseline is measured against "
        "data that was already mitigated, so the improvement attributed "
        "to the method is not the method's alone")
    if mitigation.unrequested:
        reading += (
            f". Note that {', '.join(mitigation.unrequested)} was never "
            "requested: it follows from resilience_level"
            + (" being left at its default of "
               f"{IBM_DEFAULT_RESILIENCE_LEVEL}" if mitigation.level_defaulted
               else f" being {mitigation.resilience_level}")
            + ", so nothing in the submitting code says it is on")
    return (Contradiction("mitigation", "this is an unmitigated baseline",
                          f"the service applies {detail}", reading),)
