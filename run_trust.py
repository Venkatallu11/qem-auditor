#!/usr/bin/env python3
"""Scores auditors on the QEM-Trust suite.

    python run_trust.py            # this package, plus the constant baselines

The baselines are printed alongside on purpose. A benchmark that only
ever reports the score of the tool that ships with it gives a reader no
way to tell a good score from an easy suite.

Exits non-zero if the built-in auditor is DISQUALIFIED or shows NO SKILL
-- either would mean a change to the gates has stopped the package
separating real results from artifacts, which no unit test on a single
verdict would catch.
"""
import sys

from qem_auditor import Verdict
from qem_auditor.trust import (
    TrustGrade,
    builtin_auditor,
    constant_auditor,
    score,
)

from benchmarks.suite import CASES

BASELINES = [
    ("always-hedge (NOT ESTABLISHED)", Verdict.NOT_ESTABLISHED),
    ("always-condemn (INVALID)", Verdict.INVALID),
    ("always-certify (CERTIFIED UNDER SCOPE)", Verdict.CERTIFIED_UNDER_SCOPE),
]


def main() -> int:
    report = score(builtin_auditor, CASES, "qem-auditor (this package)")
    report.print_report()

    for name, verdict in BASELINES:
        print()
        score(constant_auditor(verdict), CASES, name).print_report()

    if report.grade is TrustGrade.DISQUALIFIED:
        print("\nFAIL: the built-in auditor endorsed a known artifact.", file=sys.stderr)
        return 1
    if report.grade is TrustGrade.NO_SKILL:
        print("\nFAIL: the built-in auditor no longer beats a constant answer.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
