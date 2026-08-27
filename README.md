# qem-auditor

An AI-assisted **auditor** for quantum error-mitigation claims — not an
oracle that guesses the corrected answer, but a system whose job is to
design experiments, run them, check them, try to break them, and refuse
to declare success without evidence.

## The idea

Most "AI for quantum error mitigation" work tries to make a model
produce a better-corrected result. This project does the opposite: the
AI is never allowed to decide whether a result is trustworthy. That
judgment is made by a small set of plain, inspectable Python gates —
ideal-control checks, target-leakage checks, adversarial/negative
controls, reproducibility across independent replicates, and an honest
uncertainty bar — applied to a structured experiment record. An LLM can
propose experiments, interpret literature, and write up what happened in
prose, but it never gets to grade its own work.

```
VERDICT = INVALID                              a hard gate actively failed
VERDICT = NOT ESTABLISHED                       not enough evidence yet
VERDICT = PROMISING / REQUIRES CERTIFICATION    clean so far, not fully proven
VERDICT = CERTIFIED UNDER SCOPE                 every gate passed, full replication done
```

## Why this, and why now

This came directly out of a 40+ iteration research project
([quantum-chemistry-vqe](https://github.com/Venkatallu11/quantum-chemistry-vqe))
building a noise-mitigation pipeline for VQE chemistry on real IonQ
hardware. That project's history is, honestly, a library of failure
modes: a compiler silently cancelling gates, a ZNE extrapolator that
amplified pure shot noise 513x on a model with zero real noise to
correct, PEC results that looked good on one draw and didn't replicate,
a joint-frame optimizer with catastrophic local minima, and — eventually
— a result that survived every adversarial control thrown at it. Most of
those failures were caught by hand, one at a time, over weeks. This
project turns that manual discipline into reusable, automated gates.

## Status: Phase 1

This is early. What exists right now:

- `qem_auditor/schema.py` — the `Experiment` record: controls, outputs,
  nothing else. No verdict field — verdicts are computed, never asserted
  by whoever creates the record.
- `qem_auditor/gates.py` — five gates: ideal control, target leakage,
  adversarial controls, reproducibility (against this project's own
  established 8-replicate convention), and a chemical-accuracy bar.
- `qem_auditor/verdict.py` — combines gate results into one verdict. Hard
  gates can force `INVALID` on their own; no amount of good replication
  data overrides a broken ideal control.
- `benchmarks/` — two **real** cases, not synthetic test fixtures, pulled
  directly from `quantum-chemistry-vqe`'s own research ledger:
  - `h4_zne_blowup.py` — the real 513x ZNE blowup. Expected and actual
    verdict: `INVALID`.
  - `h4_ancilla_qed.py` — the project's current best real result (ancilla
    -parity leakage detection + conditioned PEC). Expected and actual
    verdict: `PROMISING / REQUIRES FURTHER CERTIFICATION` — deliberately
    *not* `CERTIFIED`, because full 8-draw replication and a full
    real-hardware energy validation aren't done yet, even though every
    hard gate is currently clean.

Run it:

```bash
python run_benchmarks.py
```

## Roadmap (not yet built)

- **Experiment planner**: given the current evidence, propose the next
  experiment that most reduces uncertainty per unit cost — not "run
  more shots," but a real value-of-information calculation.
- **Adversarial agent**: an LLM-driven agent whose only job is trying to
  falsify a claim (synthetic negative controls, held-out noise models,
  seed perturbations) before the auditor will pass it.
- **Backend adapters**: generalize beyond IonQ (Aer, IBM, Quantinuum)
  and beyond H4 — the `Experiment` schema is already backend-agnostic.
- **Hypothesis ledger**: track competing explanations over time
  (`P(H_i | data)`), not just pass/fail per experiment.

None of this is built yet on purpose — the gates need to be trustworthy
on real historical data first, which is what Phase 1 is for.

## Relationship to quantum-chemistry-vqe

This is a separate project, not a fork. The H4/IonQ work stays where it
is; this repo reuses its real, disclosed results as the first benchmark
suite for auditing claims, nothing more.

## License

MIT
