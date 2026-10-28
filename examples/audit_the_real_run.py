#!/usr/bin/env python3
"""The first full real-hardware run, put through the auditor.

Everything else in this repository audits a record someone wrote, or a
method running against a simulator. This audits the thing the whole
package was built for: a real quantum computer's answer to a real
chemistry question, with real money spent on it.

The run. IonQ `qpu.forte-enterprise-1`, job 01a138e7, submitted 14 Oct
2026 and finished 16 Oct after the device went offline mid-run and came
back on its own. 84 circuits at 1,100 shots. Quoted at $3,882.21.

The claim. A shared-frame correction brings the H4 energy error from
23.84 kcal/mol down to **0.864**, which is inside chemical accuracy
(1 kcal/mol) -- a 27.6x improvement on real hardware.

The point of this file is that the claim is not certified, and the
reasons are arithmetic rather than opinion. Nothing here needs qiskit:
the inputs are a residual list, three energies and a bootstrap.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qem_auditor.calibration import audit_groups, audit_residuals  # noqa: E402
from qem_auditor.cost import RATE_CARDS  # noqa: E402

FIXTURE = (Path(__file__).resolve().parent.parent / "tests" / "fixtures"
           / "ionq_full_run.json")

#: Chemical accuracy. The bar the field agreed on long before this run.
CHEMICAL_ACCURACY = 1.0

#: The forte-1 simulator rehearsal of these same 84 circuits, run before
#: the hardware and reported at this chi2/dof. Its per-residual rows
#: were not published, so only the summary figure is used.
REHEARSAL_CHI2_DOF = 2.68


def rule(title):
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)


def main() -> int:
    run = json.loads(FIXTURE.read_text())["full_run"]
    energies = run["energies_kcal_per_mol"]
    boot = run["bootstrap"]["shared_frame"]

    rule("THE RUN")
    print(f"  job {run['job_id']} on {run['backend']}")
    print(f"  {run['circuits']} circuits, {run['shots_per_circuit']:,} shots "
          f"each, quoted ${run['cost_estimate_usd']:,.2f}")
    print(f"  vendor error mitigation: {run['vendor_debiasing']}")
    print(f"  ancilla retention {run['mean_ancilla_retention']:.3f} mean, "
          f"{run['min_ancilla_retention']:.3f} worst circuit")

    rule("THE CLAIM, AT FACE VALUE")
    raw = energies["raw"]["err_kcal"]
    corrected = energies["shared_frame"]["err_kcal"]
    print(f"  raw error            {raw:8.3f} kcal/mol")
    print(f"  no-frame correction  {energies['no_frame']['err_kcal']:8.3f}")
    print(f"  shared frame         {corrected:8.3f}   <- the claim")
    print(f"  improvement          {raw / corrected:8.1f}x")
    print(f"\n  {corrected:.3f} is inside chemical accuracy "
          f"({CHEMICAL_ACCURACY} kcal/mol). Taken alone, that is a pass.")

    rule("1. THE POINT ESTIMATE IS NOT THE EXPERIMENT")
    low, high = boot["err_kcal_boot_q05"], boot["err_kcal_boot_q95"]
    print(f"  bootstrap over 20 resamples: {boot['err_kcal_boot_mean']:.3f} "
          f"+- {boot['err_kcal_boot_std']:.3f} kcal/mol")
    print(f"  90% of the resampled draws land in [{low:.3f}, {high:.3f}]")
    straddles = low <= CHEMICAL_ACCURACY <= high
    print(f"\n  Does that interval straddle {CHEMICAL_ACCURACY} kcal/mol? "
          f"{'YES' if straddles else 'no'}")
    print(f"  The bootstrap MEAN ({boot['err_kcal_boot_mean']:.3f}) is on the "
          "wrong side of the bar,")
    print(f"  while the point estimate ({corrected:.3f}) is on the right one. "
          "Both are this")
    print("  run; they disagree about whether it passed.")
    print("\n  Chemical accuracy is CONSISTENT WITH this data. It is not")
    print("  established BY it. That distinction is the whole verdict.")

    rule("2. THE ERROR BARS ARE NOT THE RIGHT SIZE")
    audit = audit_residuals([r["z"] for r in run["residuals"]],
                            fitted_parameters=run["fitted_parameters"])
    print(audit.describe())
    print()
    print("  And the headline hides two different problems:")
    by_group = {}
    for row in run["residuals"]:
        by_group.setdefault(row["group"], []).append(row["z"])
    grouped = audit_groups(by_group,
                           fitted_parameters=run["fitted_parameters"])
    for key in sorted(grouped.groups):
        one = grouped.groups[key]
        kind = ("systematic shift" if abs(one.mean_z) > 1.0
                else "scatter")
        print(f"    group {key}  chi2/dof {one.chi2_per_dof:5.2f}  "
              f"mean z {one.mean_z:+.2f}   mostly {kind}")
    print("\n  Group 0 is off in one direction; group 2 is the noisiest.")
    print("  More shots fixes neither, and would make chi2/dof worse.")

    rule("3. THE REHEARSAL PREDICTED PART OF IT, AND SAID SO FIRST")
    hardware_excess = audit.chi2_per_dof - 1.0
    model_excess = REHEARSAL_CHI2_DOF - 1.0
    print(f"  simulator rehearsal, same 84 circuits: chi2/dof "
          f"{REHEARSAL_CHI2_DOF:.2f}")
    print(f"  hardware:                              chi2/dof "
          f"{audit.chi2_per_dof:.2f}")
    print(f"\n  Measured as excess over 1.0, the model accounted for "
          f"{model_excess / hardware_excess:.0%} of it.")
    print(f"  The other {1 - model_excess / hardware_excess:.0%} appeared only "
          "on hardware -- which is what")
    print("  this run was designed to measure, and it did.")

    rule("4. WHAT THE NEXT RUN WOULD COST")
    rates = RATE_CARDS["ionq_forte"]
    per_1q = run["total_one_qubit_gates"] // run["circuits_submitted"]
    per_2q = run["total_two_qubit_gates"] // run["circuits_submitted"]
    ours = (run["total_one_qubit_gates"] * rates.one_qubit_usd
            + run["total_two_qubit_gates"] * rates.two_qubit_usd
            ) * run["shots_per_circuit"]
    quoted = run["cost_estimate_usd"]
    print(f"  IonQ quoted  ${quoted:,.2f}")
    print(f"  this card    ${ours:,.2f}   ({abs(ours - quoted) / quoted:.2%} low, "
          f"${quoted - ours:.2f} unexplained)")
    print(f"\n  The card was built from two $25.79 jobs. This is a bill "
          f"{quoted / 25.79:.0f}x larger.")
    print("  The 0.18% gap is not rate drift: the rates measured two months")
    print("  later make the agreement slightly worse, not better.")
    print()
    print("  So would more shots settle the claim? This is where section 2")
    print("  earns its place. Shot noise falls as 1/sqrt(N); the unmodelled")
    print(f"  term -- {audit.unmodelled_in_sigmas:.2f}x the shot noise, measured "
          "above -- does not fall")
    print("  at all. They add in quadrature, so the TOTAL barely moves:")
    print()
    print(f"    {'shots/circuit':>14s}  {'cost':>11s}   {'total bar':>10s}   "
          "vs now")
    unmodelled = audit.unmodelled_in_sigmas          # in units of today's shot noise
    base = (1.0 + unmodelled ** 2) ** 0.5
    for factor in (1, 2, 4, 16):
        shots = run["shots_per_circuit"] * factor
        cost = (run["total_one_qubit_gates"] * rates.one_qubit_usd
                + run["total_two_qubit_gates"] * rates.two_qubit_usd) * shots
        total = ((1.0 / factor) + unmodelled ** 2) ** 0.5
        print(f"    {shots:14,}  ${cost:10,.2f}   {total:10.3f}   "
              f"{total / base:.3f}x")
    print()
    print(f"  Sixteen times the shots -- ${16 * ours:,.0f} -- buys a bar "
          f"{1 - ((1 / 16) + unmodelled ** 2) ** 0.5 / base:.1%} narrower.")
    print("  The money does not buy the claim. Finding the missing error does,")
    print("  and group 0's systematic shift is where to start looking.")

    rule("THE READING")
    print("  Three things are established by this run:")
    print("    - the circuits run correctly on real hardware")
    print(f"    - the correction reduces the error {raw / corrected:.0f}x, "
          "which is large and real")
    print("    - the noise model predicted the RAW error well (23.84 vs 24.06)")
    print()
    print("  One thing is not:")
    print("    - that the result is inside chemical accuracy. The point")
    print("      estimate says yes, the bootstrap straddles the bar, and the")
    print(f"      error bars are {audit.inflation:.1f}x too tight on top of that.")
    print()
    print("  An auditor that reported only the 0.864 would be reporting the")
    print("  most flattering of three numbers this run produced. It is the")
    print("  right number to be proud of and the wrong one to certify.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
