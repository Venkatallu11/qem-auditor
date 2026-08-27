#!/usr/bin/env python3
"""Runs the auditor against every real benchmark case in benchmarks/ and
prints a verdict for each. No mocked data -- every number here traces
back to a real, disclosed experiment in quantum-chemistry-vqe's own
RESEARCH_LEDGER.md.
"""
from qem_auditor import audit

from benchmarks.h4_zne_blowup import EXPERIMENT as ZNE_BLOWUP
from benchmarks.h4_ancilla_qed import EXPERIMENT as ANCILLA_QED

BENCHMARKS = [ZNE_BLOWUP, ANCILLA_QED]


def main():
    for exp in BENCHMARKS:
        report = audit(exp)
        report.print_report()


if __name__ == "__main__":
    main()
