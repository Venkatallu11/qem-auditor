"""Integrity checks on raw measurement counts.

`integrity.py` asks whether a RECORD is coherent enough to audit. This
module asks a question one level below that: is the raw data the record
was computed from even the thing it claims to be?

The checks here are cheap, universal, and none of them is hypothetical.
The one that motivated the module happened on real trapped-ion hardware
in a sister project: a vendor SDK returned a "histogram" endpoint's RAW
COUNTS, the client code assumed probabilities and multiplied by the shot
count, and every counts table came back inflated by exactly the shot
count -- sums of 4,000,000 where 2,000 shots were requested.

What makes that failure worth a module is what it does downstream. An
expectation value is a WEIGHTED mean, so scaling every count by the same
factor leaves the estimate completely unchanged. The shot-noise bar is
not: it falls as 1/sqrt(N), so counts inflated by k report an interval
sqrt(k) times too tight. At k = 2000 that is a bar 45x too small sitting
under an estimate that looks perfectly correct. Nothing about the
numbers looks wrong. The claim just quietly becomes 45 times stronger
than the data supports.

It was caught by one line of arithmetic -- does the table sum to the
shots we asked for -- and that line is what this module generalises.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


class CountsError(ValueError):
    """Raised when counts cannot support the quantity asked of them."""


#: A counts table whose values are floats summing to about this is
#: probabilities wearing a counts table's clothes -- the exact mistake
#: whose mirror image inflated a real experiment's data by 2000x.
_PROBABILITY_SUM_TOL = 1e-6


@dataclass(frozen=True)
class CountsProblem:
    """One defect, and what it does to the answer.

    `consequence` is not decoration. A problem a caller cannot price is
    a problem they will argue with; saying "the bar comes out 45x too
    tight" ends the argument.
    """

    kind: str
    detail: str
    consequence: str
    setting: Optional[str] = None

    def describe(self) -> str:
        where = f"[{self.setting}] " if self.setting else ""
        return f"{where}{self.detail}\n      -> {self.consequence}"


@dataclass(frozen=True)
class CountsReport:
    """What the raw data is, and where it contradicts what was claimed."""

    problems: tuple
    settings: int
    total_shots: int
    width: Optional[int]

    @property
    def ok(self) -> bool:
        return not self.problems

    def raise_if_broken(self) -> "CountsReport":
        if self.problems:
            raise CountsError(self.describe())
        return self

    def describe(self) -> str:
        if self.ok:
            return (f"counts look sound: {self.settings} setting(s), "
                    f"{self.total_shots} shots, {self.width}-bit outcomes")
        lines = [f"{len(self.problems)} problem(s) with the raw counts:"]
        lines.extend("  - " + problem.describe() for problem in self.problems)
        return "\n".join(lines)


def _is_count(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and float(value).is_integer()


def check_counts(measurements: dict,
                 declared_shots: Optional[int] = None,
                 declared_width: Optional[int] = None) -> CountsReport:
    """Check counts tables before anything is computed from them.

    `measurements` maps a setting name to its counts table, matching
    `results.analyse`. `declared_shots` is what the submitter says was
    requested PER SETTING -- pass it whenever it is known, because it is
    the only check here that can catch a table that is internally
    perfect and still wrong by a constant factor.
    """
    problems = []
    widths = {}
    total = 0

    if not measurements:
        problems.append(CountsProblem(
            "no_settings", "no measurement settings were supplied",
            "there is nothing to estimate from"))
        return CountsReport(tuple(problems), 0, 0, None)

    for name, counts in measurements.items():
        if not counts:
            problems.append(CountsProblem(
                "empty", "the counts table is empty",
                "this setting contributes no information, and any term "
                "routed to it is unmeasured rather than measured as zero",
                name))
            continue

        setting_total = 0
        for bits, value in counts.items():
            if not isinstance(bits, str) or not bits or set(bits) - {"0", "1"}:
                problems.append(CountsProblem(
                    "not_a_bitstring", f"outcome key {bits!r} is not a bitstring",
                    "parities are read off these characters, so a "
                    "non-binary key is either a different encoding or a "
                    "corrupted one -- both give a wrong expectation value",
                    name))
                continue
            widths.setdefault(len(bits), []).append(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                problems.append(CountsProblem(
                    "not_a_number", f"count for {bits} is {value!r}",
                    "a count that is not a number cannot be weighted",
                    name))
                continue
            if value < 0:
                problems.append(CountsProblem(
                    "negative", f"count for {bits} is {value}",
                    "a negative count is not an observation; if this came "
                    "out of a readout correction it is a corrected WEIGHT "
                    "and must not be re-analysed as raw data",
                    name))
                continue
            if not _is_count(value):
                problems.append(CountsProblem(
                    "not_integral", f"count for {bits} is {value}, not a whole number",
                    "shots are counted, not measured; a fractional count "
                    "means these are probabilities or already-processed "
                    "weights, and the shot-noise floor computed from them "
                    "will be a number with no experiment behind it",
                    name))
            setting_total += value

        if setting_total <= 0:
            problems.append(CountsProblem(
                "no_shots", f"the counts sum to {setting_total}",
                "an expectation value needs at least one shot", name))
            continue

        if abs(setting_total - 1.0) < _PROBABILITY_SUM_TOL:
            problems.append(CountsProblem(
                "probabilities", "the counts sum to 1.0",
                "these are probabilities, not counts. The estimate will "
                "come out right and the shot-noise floor will be computed "
                "as though the experiment had a single shot",
                name))

        total += int(setting_total)

        if declared_shots is not None and setting_total != declared_shots:
            ratio = setting_total / declared_shots
            if ratio > 1:
                effect = (f"the estimate is unaffected -- it is a weighted "
                          f"mean -- but the shot-noise bar computed from "
                          f"this table is {math.sqrt(ratio):.1f}x TOO TIGHT")
            else:
                effect = (f"the shot-noise bar computed from this table is "
                          f"{math.sqrt(1 / ratio):.1f}x too wide, and shots "
                          f"that were paid for are missing from the analysis")
            problems.append(CountsProblem(
                "shot_mismatch",
                f"the counts sum to {int(setting_total)} but {declared_shots} "
                f"shots were declared ({ratio:.4g}x)",
                effect, name))

    if declared_width is not None:
        for width in widths:
            if width != declared_width:
                problems.append(CountsProblem(
                    "width_mismatch",
                    f"{width}-bit outcomes where {declared_width} qubits "
                    "were declared",
                    "qubit indices in the observable address positions in "
                    "this string, so a width mismatch reads the wrong "
                    "qubit rather than failing"))
    if len(widths) > 1:
        shown = ", ".join(f"{w} bits in {sorted(set(names))[0]}"
                          for w, names in sorted(widths.items()))
        problems.append(CountsProblem(
            "ragged_width", f"outcomes are not all the same width ({shown})",
            "the same qubit index points at a different qubit in each "
            "table, so terms measured across settings are being combined "
            "from different physical qubits"))

    return CountsReport(tuple(problems), len(measurements), total,
                        min(widths) if widths else None)
