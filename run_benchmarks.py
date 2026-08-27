#!/usr/bin/env python3
"""Runs the auditor against every real benchmark case in benchmarks/ and
checks each one against the verdict that case is known to deserve. No
mocked data -- every number here traces back to a real, disclosed
experiment in quantum-chemistry-vqe's own RESEARCH_LEDGER.md.

This is a regression harness, not a demo: it exits non-zero if any case
audits to something other than its recorded EXPECTED_VERDICT. A change to
the gates that quietly stops flagging the 513x ZNE blowup -- or that
starts certifying a result whose replication isn't finished -- fails here.
"""
import sys

from qem_auditor import audit

from benchmarks import h4_ancilla_qed, h4_zne_blowup

BENCHMARKS = [h4_zne_blowup, h4_ancilla_qed]


def main() -> int:
    failures = []
    for module in BENCHMARKS:
        exp = module.EXPERIMENT
        expected = module.EXPECTED_VERDICT
        report = audit(exp)
        report.print_report()
        if report.verdict is not expected:
            failures.append(
                f"{exp.experiment_id}: expected {expected.value}, got {report.verdict.value}"
            )
            print(f"  MISMATCH: expected {expected.value}")

    print()
    if failures:
        print(f"{len(failures)}/{len(BENCHMARKS)} benchmark(s) audited to the wrong verdict:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"All {len(BENCHMARKS)} benchmark(s) audited to their expected verdicts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
