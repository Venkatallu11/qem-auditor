# qem-auditor

An AI-assisted **auditor** for quantum error-mitigation claims — not an
oracle that guesses the corrected answer, but a system whose job is to
design experiments, run them, check them, try to break them, propose the
next experiment, and refuse to declare success without evidence.

The AI never decides whether a result is trustworthy. It proposes
experiments, interprets literature, and writes up what happened; plain,
inspectable Python decides what passed. Most AI-for-QEM work tries to
make a model produce a better-corrected number. This does the opposite:
it tries to prove that a number deserves to be believed.

## The idea

Judgment is made by a small set of plain Python gates — ideal-control
checks, circuit-equivalence checks, target-leakage checks,
adversarial/negative controls, extrapolation-domain checks, determinism
checks, replication across genuinely independent submissions, tail-risk
checks, and an honest uncertainty bar — applied to a structured
experiment record.

```
VERDICT = INVALID RECORD        the record contradicts itself; unauditable
VERDICT = INVALID               a hard gate actively failed
VERDICT = REFUTED               the claim's own evidence contradicts it
VERDICT = CONFLICT              independent measurements disagree
VERDICT = NOT ESTABLISHED       required evidence was never collected
VERDICT = VALID UNDER MODEL     holds, but only under the models tested
VERDICT = PROMISING             clean so far, not fully proven
VERDICT = CERTIFIED UNDER SCOPE every gate passed, replication complete
```

Three rules do most of the work:

- A hard gate that **failed** forces `INVALID` no matter how good the
  rest of the evidence looks.
- A hard gate that was never **run** is not a pass — silence is not
  evidence, so it caps the verdict at `NOT ESTABLISHED`.
- A claim is graded **before** replication is demanded. It takes more
  evidence to bless a claim than to withhold a blessing.

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

## Status: v0.3 — the autonomous auditor

v0.2 went from **auditing experiments supplied to it** to **generating and
executing the cheapest experiment capable of falsifying the claim**. v0.3
lets it do that **on its own**, optionally with a language model widening
the search — while keeping the model strictly a proposer.

```
qem_auditor/
  schema.py         the experiment record
  record.py         read/write records as JSON
  integrity.py      is this record even readable?
  gates.py          14 gates, each from a real disqualification
  verdict.py        gate results -> one verdict
  failure_modes.py  why it failed, and the cheapest fix
  hypothesis.py     competing explanations, Bayesian, carried across experiments
  planner.py        what to run next, by information gain per dollar
  claim.py          what has been shown, what has not, what closes the gap
  adversary.py      generates falsification experiments  (v0.2)
  executor.py       runs them; never pretends about what it could not run  (v0.2)
  power.py          how much evidence would actually be enough  (v0.2)
  active_design.py  when more data cannot help, and what to run instead  (v0.2)
  provenance.py     content-addressed evidence bundles  (v0.2)
  blind.py          audit without seeing the answer  (v0.2)
  llm.py            provider-agnostic model access (v0.3)
  llm_scientist.py  model proposals, validated by the same code (v0.3)
  agent.py          the loop, running by itself (v0.3)
  report.py         console and self-contained HTML reports (v0.3)
  api.py            the Auditor facade
  cli.py            python -m qem_auditor
  adapters/         execute controls instead of trusting them (optional, needs qiskit)
benchmarks/         6 real QEM-Trust cases
examples/           end-to-end verification, and the full adversarial loop
tests/              291 tests
```

### The loop

```
claim -> audit -> what can still be wrong -> generate adversaries
      -> execute -> formal audit -> belief update -> next experiment
```

The division of labour is what keeps a proposer in the loop without
letting it grade itself:

| | says |
|---|---|
| the proposer | "this attack should distinguish H1 from H2, and here is what each outcome would mean" |
| the executor | runs the attack |
| the gates | what actually happened |

The proposer commits to what each outcome means **before** anything runs,
so it cannot reinterpret a bad result afterwards. `AdversarialScientist`
has no API for issuing a verdict, and a test keeps it that way.

### Running it autonomously

```bash
qem-auditor investigate my_experiment.json --qiskit --html report.html
```

It audits, works out what could still be wrong, proposes attacks, executes
the ones it can, folds the results back in, and decides whether to
continue — stopping when nothing informative remains to buy, and saying
why.

On a real ZNE claim submitted at `optimization_level=3`, unaided:

```
round 1: NOT ESTABLISHED | 10 attacks | 2 falsified, 2 survived, 0 not run
    FALSIFIED by T_compiler
    FALSIFIED by T_compiler+T_extrapolation
round 2: INVALID
stopped: INVALID: the claim is disqualified, and no further attack changes that
```

The agent decides only **whether to keep going**. Every verdict comes from
the gates, it has no method containing the word "certify", and a test
asserts an agent run cannot upgrade an unproven claim.

### Adding a language model (optional, and free)

The auditor needs no model. One only widens the set of attacks
considered — a local model is plenty:

```bash
ollama serve && ollama pull llama3.1
export QEM_LLM_PROVIDER=openai
export QEM_LLM_BASE_URL=http://localhost:11434/v1
export QEM_LLM_MODEL=llama3.1
```

Anything speaking `/v1/chat/completions` works (Ollama, LM Studio,
llama.cpp, vLLM, Groq, OpenRouter, Together), as does the Anthropic
Messages API. With nothing configured, the deterministic grammar runs and
reaches the same verdicts.

**The model is a proposer and structurally cannot be anything else.**
Every proposal meets the same validation a hand-written one does:

| the model proposes | what happens |
|---|---|
| an attack with different genuine/artifact outcomes | accepted, at a lower discrimination than a hand-written one |
| an attack predicting the same thing either way | **rejected** — it will confirm whatever you already believe |
| an attack missing its statistic | **rejected**, naming the gaps |
| `"verdict": "PASS"`, `"ideal_control": true` | those fields **stripped**; the legitimate part kept |
| a hypothesis with no observable consequence | **rejected** — no experiment could ever address it |

Rejections are reported, never silently dropped. A model that proposes
six attacks of which two are non-diagnostic has told you something useful
about itself, and hiding that would make the auditor's own behaviour
unauditable.

### The failure grammar

Nine transformations, each from a failure this project actually suffered,
and they compose:

`T_label` `T_sign` `T_seed` `T_calibration` `T_compiler`
`T_extrapolation` `T_shot` `T_leakage` `T_correlation`

An attack is only an attack if it predicts **different** outcomes under
"genuine" and "artifact". `Prediction` refuses to be constructed
otherwise — an experiment both hypotheses predict identically will
confirm whatever you already believe.

Composition is deliberate: the H4 robustness envelope was `T_calibration`
composed with a coherent-error transformation, and it behaved nothing like
either alone (per-instance coherent bias gave Q95 = 827 kcal/mol, the same
magnitude applied per gate *type* gave 0.21). An auditor that only tests
one transformation at a time misses interactions of exactly that kind.

A **self-reported** passing control does not close a question. Only a
control the auditor measured does — which is the whole reason an adversary
exists.

### When more data cannot help

The most expensive experiment is one that cannot answer its question at
any sample size. `active_design.py` computes the Fisher information
`F = JᵀΣ⁻¹J`; a near-zero eigenvalue names a direction the experiment is
blind to, and blindness is not cured by repetition.

On the H4-shaped case — a design sensitive only to `p_ZZ − p_GPi2` — it
reports the blind direction and then ranks candidates by information along
it per dollar:

```
   1.939  zz_only_calibration       adds 50 along the weak direction at $25.79
 0.07326  full_sweep                adds 500 at $6,825.00
       0  more_shots_same_circuit   adds 0 along the weak direction for free
```

Free is not the same as useful.

### How much evidence is enough

`UNDER_POWERED` as a label tells a researcher nothing. `power.py` returns
power, `required_n`, `minimum_additional` and expected cost — and refuses
to size a sample when **σ does not match the claim's uncertainty scope**.
That refusal is the project's most expensive historical mistake restated
statistically: the 8-seed bootstrap bars had a within-submission σ near
0.0015 kcal/mol while independent submissions of the identical circuit set
differed by 3.27.

It also sizes against the 95% *upper bound* on σ rather than the point
estimate, because a sample sd from 4 draws is not σ — it is a noisy guess
at σ, inflated ~3.0x at n=4 and ~1.8x at n=8.

A finding worth recording: powering against the scale the draws actually
differ on returns `required_n = 8`, independently rederiving the project's
own 8-draw convention from the statistics rather than from habit.

### Blind mode

An auditor evaluated on records whose expected verdict sits in the same
file is being graded on a task it can see the answer to — the target-leakage
failure its own benchmarks encode. A `BlindChallenge` withholds every
outcome quantity while keeping the methodology visible, and `reveal()`
refuses until a decision is committed.

All six benchmarks are answered correctly blind, the flagship ancilla-QED
case included: the auditor withholds certification and names the missing
evidence without ever seeing 0.0144 or the Q95.

### Provenance

Content-addressed evidence bundles, so "can you reproduce 0.018?" is
answerable. A digest names an exact combination of circuit, counts,
calibration, seeds, backend, environment and analysis version; `diff` says
*which* input moved, which a bare mismatch cannot. `is_reproducible` flags
an unset `PYTHONHASHSEED` and a dirty working tree — both make a recorded
commit fail to identify the code that ran.

Stated plainly: these are content hashes, not signatures. They detect
change; they do not authenticate.

## Install

```bash
pip install -e .                 # core: no dependencies at all
pip install -e ".[adapters]"     # plus qiskit, to execute controls
```

## Using it on your own experiment

No Python required for the basic path:

```bash
qem-auditor template > my_experiment.json   # a blank record to fill in
qem-auditor validate my_experiment.json     # is it readable and self-consistent?
qem-auditor audit my_experiment.json        # the verdict, why, and what to run next
qem-auditor attack my_experiment.json       # what would falsify this claim?
qem-auditor blind my_experiment.json        # audit with the outcome hidden, then reveal
qem-auditor investigate my_experiment.json  # run the loop autonomously
qem-auditor audit my_experiment.json --html report.html   # a shareable report
qem-auditor audit my_experiment.json --json # machine-readable, for CI
```

(`python -m qem_auditor ...` works identically.)

Exit codes are meant for CI, so a claim cannot quietly regress: `0` certified,
`1` anything else, `2` the record could not be read.

From Python:

```python
from qem_auditor import Auditor

result = Auditor().audit("my_experiment.json")
result.verdict            # Verdict.NOT_ESTABLISHED
result.failure_modes      # [FailureMode.UNDER_POWERED, ...]
result.next_experiment    # the cheapest missing evidence
print(result.render())
```

### Grading a record vs verifying a claim

With no adapter, the auditor grades the record **as written** — it checks
whether you *claim* the ideal control passed. That is useful for your own
work and worthless against a stranger's, because a gate that trusts the
claimant is not a gate.

With an adapter it executes the controls it can:

```python
from qem_auditor.adapters.qiskit_adapter import QiskitAdapter

auditor = Auditor(adapter=QiskitAdapter())
auditor.verify_fold_survival(exp, base=base_circuit, submitted=submitted_circuit)
auditor.verify_ideal_control(exp, circuit, observable, my_mitigation_pipeline)
auditor.verify_determinism(exp, my_pipeline)
result = auditor.audit(exp)
```

Every control carries `Provenance` — `SELF_REPORTED` or `MEASURED` — and
`CERTIFIED UNDER SCOPE` requires that everything the auditor *could* have
checked, it did. Three controls are mechanizable today:

| control | what the auditor runs | the failure it catches |
|---|---|---|
| fold survival | builds both circuits, transpiles, compares unitary **and** gate count | fold pairs optimized back out before submission |
| ideal control | runs *your* mitigation against a noiseless model with shot noise | an estimator that amplifies statistical noise (the 513x) |
| determinism | runs the identical computation N times and diffs | hash-order nondeterminism tipping a nonconvex solver |

Target leakage, adversarial design, and free-parameter floors are
procedural or domain-specific and still rest on honest reporting. The
auditor says so rather than papering over it —
`independent_verification` reports exactly which controls were measured
and which were taken on trust.

The fold-survival check is worth a note. A ZNE fold pair is *supposed* to
leave the unitary unchanged, so a unitary-equivalence check passes happily
while the transpiler removes the pairs — leaving an "amplified" arm with
the same noise as the unamplified one, and an extrapolation fitting a
slope through a variable that never varied. Both halves have to be checked
at once. `examples/verify_zne_claim.py` demonstrates this against the real
Qiskit transpiler: `optimization_level=3` cancels the folds and audits to
`INVALID`; `optimization_level=0` preserves them.

Run it:

```bash
python run_benchmarks.py                    # the 6 real cases; non-zero exit on a wrong verdict
python run_audit_loop.py                    # the closed loop, replayed on real history
python examples/verify_zne_claim.py         # end-to-end verification (needs qiskit)
python examples/adversarial_loop.py         # propose, execute, judge (needs qiskit)
python examples/autonomous_audit.py         # the agent, unattended (needs qiskit)
python -m unittest discover -s tests -t .   # full suite
```

Installing nothing runs everything except the adapters, which skip.

### The record

Everything is built against `quantum-chemistry-vqe`'s 12,000-line
research ledger. Three distinctions in that history a naive schema gets
wrong, each of which cost that project a real result:

**Bootstrap resampling is not replication.** The identical circuit set was
submitted to IonQ's simulator twice, independently, and the two energies
differed by 3.27 kcal/mol — three times the project's own reproducibility
bar. Its standard "8-seed mean ± std" bars never showed this, because they
resample *one* submission's counts. `Replicate.kind` makes the difference
explicit, and only independent submissions count toward a replication
target.

**An uncertainty bar is only as broad as what it varied.** The project's
0.115 kcal/mol headline was real under one fixed noise model and disowned
once the noise model's own parameters were allowed to vary (Q95 = 51.22,
~205x over target). `UncertaintyCoverage` tracks four independent axes
rather than a single scale — the joint Schmidt frame propagated
noise-model uncertainty but never cross-submission drift, and the earlier
headline numbers had exactly the opposite gap. Neither is a superset of
the other.

**"Beats this baseline" and "reaches chemical accuracy" are different
claims.** The joint Schmidt frame is an 88.8% MSE reduction that still
does not reach chemical accuracy. Grading it against an absolute bar
would call a real result a failure.

### The QEM-Trust benchmark

Six real cases, not synthetic fixtures. Each is pinned to both its verdict
and its primary failure mode, so a change that stops catching one fails
loudly.

| case | verdict | why |
|---|---|---|
| abstract-gate folding | `INVALID` | compiler optimized the inserted gates back out; the executed circuit was not the designed one |
| 513x ZNE blowup | `INVALID` | held-out validation tested interpolation while production extrapolated |
| 0.115 one-off PEC | `NOT ESTABLISHED` | every hard gate clean, chemical accuracy met — and the replicates were bootstrap resamples |
| cross-fitted manifold | `REFUTED` | median improved, 2/32 trials blew up to ~2100 kcal/mol |
| joint Schmidt frame | `VALID UNDER MODEL` | 88.8% MSE reduction, adversarially validated, no cross-submission drift in the bars |
| ancilla-QED + PEC | `PROMISING` | the current best result, correctly refused certification |

The third row is the one worth staring at. Every hard gate passes and it
clears chemical accuracy, and the auditor still refuses — on nothing more
than the observation that a single submission's bootstrap bar cannot speak
to reproducibility. That refusal happens *before* the expensive robustness
study that later disowned the number.

### Explaining why, not just that

Not "ZNE failed" but:

```
EXTRAPOLATION_INSTABILITY (confidence 0.90)
  the estimator is used outside the domain its held-out validation
  tested; also: on a model with zero real noise to correct, mitigation
  amplified the error 513x (0.0652 -> 33.4800 kcal/mol) -- the failure is
  numerical conditioning of the estimator, not hardware noise
  remedy: quantify the estimator's conditioning at the production point
          before any further hardware spend
```

The classifier is never shown what the record's author suspected;
agreement is reported separately, as a check on the classifier rather
than an input to it.

### Choosing the next experiment

The planner ranks candidates by expected information gain per dollar, and
is allowed to say stop. It derives candidates from the audit's own gaps —
a gate that never ran *is* the missing experiment, with a known procedure
and a known cost. It distinguishes two questions that get conflated:
closing a record gap changes what a result can be **cited as**; running a
discriminating experiment changes which hypothesis is **true**. A free
experiment is often worth running for the first reason while gaining
nothing on the second.

Saying stop is a first-class output. Real hardware here costs ~$25.79 per
circuit, dominated by per-circuit overhead rather than shots (confirmed by
a 100→500 shot probe that cost the same), so a full 21-slot × 13-group
energy reconstruction is ~273 circuits ≈ $6,825. "More shots" was
historically the wrong answer anyway: in the measured variance budget,
shot noise was 0.0037 against the method's own Monte Carlo at 2.11.

## Roadmap (not yet built)

- **Adversarial agent**: an LLM-driven agent whose only job is trying to
  falsify a claim — generating the negative controls, held-out noise
  models, and seed perturbations itself, rather than checking that a
  human ran them. The gates already encode what must be survived; what
  is missing is a proposer.
- **More executable attacks**: six of the nine transformations still need
  a domain hook (the claimant's own fitting code). `T_label`, `T_sign` and
  `T_shot` are the tractable next three — they need a fit-and-reconstruct
  interface the schema does not yet define.
- **Noisy-backend adapters**: the Qiskit adapter uses exact statevectors
  plus sampled shot noise, which is what the ideal control needs. Running
  claims against Aer noise models, IBM, or Quantinuum is the same
  interface with a different expectation oracle.
- **Likelihoods from a simulator**: hypothesis likelihoods are currently
  supplied per observation. Generating `P(D|H_i)` from a predictive model
  would make the Bayesian layer much stronger, and would let the proposer
  check that a new hypothesis has measurable consequences before it
  becomes an experiment candidate.
- **Learned priors**: with enough records across molecules, backends and
  methods, `P(mitigation succeeds | circuit features)` becomes
  estimable — useful for pricing an experiment before running it. This
  needs volume the project does not yet have, and is deliberately last.

## Relationship to quantum-chemistry-vqe

This is a separate project, not a fork. The H4/IonQ work stays where it
is; this repo reuses its real, disclosed results as the first benchmark
suite for auditing claims, nothing more.

## License

MIT
