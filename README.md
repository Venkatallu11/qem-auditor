# qem-auditor

**Can I trust this quantum error-mitigation result?**

An auditor for quantum error-mitigation claims. It does not try to produce
a better-corrected number — it tries to work out whether a number deserves
to be believed, by designing experiments that would destroy it and
reporting what survived.

The AI never decides. It proposes experiments, widens the search, and
writes prose; plain, inspectable Python decides what passed. An AI that
could talk itself into `CERTIFIED` would be the exact failure this exists
to prevent.

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

- A hard gate that **failed** forces `INVALID`, however good the rest looks.
- A hard gate that was never **run** is not a pass — silence is not
  evidence, so it caps the verdict at `NOT ESTABLISHED`.
- A claim is graded **before** replication is demanded. It takes more
  evidence to bless a claim than to withhold a blessing.

---

## Quick start

```bash
pip install -e .                 # core: no dependencies at all
pip install -e ".[adapters]"     # plus qiskit, to execute controls
```

Bring a circuit, get a verdict — no record to write:

```bash
qem-auditor check --template > my_circuit.py   # a starting point
qem-auditor check my_circuit.py                # the verdict
```

You define `circuit`, and optionally `observable`, `mitigator`,
`submitted_circuit` and `amplified_circuit`. Gate counts, qubits and basis
are read off the circuit itself, and the auditor **executes** what it can:

```
EXECUTED BY THE AUDITOR
  [PASS] unitary_equivalence: submitted circuit implements the intended unitary
  [PASS] ideal_control: mitigation is not pathologically conditioned (1.4x)
  [PASS] determinism_check: 3 identical runs produced identical results

OUTSIDE WHAT THIS CHECK CAN ESTABLISH
  target_leakage: whether the known answer influenced tuning -- procedural,
                  and not visible in a circuit
  adversarial: needs your own fitting code to shuffle and refit
  free_parameter_floor: needs your method's parameters
  reproducibility: the auditor cannot run your submissions for you
```

**Those two lists are the point.** The auditor never fills in a control it
did not measure, and it separates *your method failed this* from *nobody
could check this from a circuit*. Certification is structurally
unreachable from artifacts alone — correct, not a limitation to route
around.

One subtlety, because it cost the source project a real result: if your
pipeline inserts gates deliberately (ZNE folding), define
`amplified_circuit` too. A fold pair is *supposed* to leave the unitary
unchanged, so an equivalence check passes happily while the transpiler
removes the pairs. The auditor has to be told the gates were deliberate
before it can check they survived.

### Everything else

```bash
qem-auditor validate rec.json      # readable and self-consistent?
qem-auditor audit rec.json         # verdict, why, and what to run next
qem-auditor attack rec.json        # what would falsify this claim?
qem-auditor blind rec.json         # audit with the outcome hidden, then reveal
qem-auditor investigate rec.json   # run the loop autonomously
qem-auditor audit rec.json --html report.html   # a shareable report
```

Exit codes are meant for CI: `0` certified, `1` anything else, `2`
unreadable.

From Python:

```python
from qem_auditor import Auditor

result = Auditor().audit("my_experiment.json")
result.verdict            # Verdict.NOT_ESTABLISHED
result.failure_modes      # [FailureMode.UNDER_POWERED, ...]
result.next_experiment    # the cheapest missing evidence
print(result.render())
```

---

## Why this, and why now

This came out of a 40+ iteration research project
([quantum-chemistry-vqe](https://github.com/Venkatallu11/quantum-chemistry-vqe))
building a noise-mitigation pipeline for VQE chemistry on real IonQ
hardware. That project's history is, honestly, a library of failure modes:
a compiler silently cancelling gates, a ZNE extrapolator that amplified
pure shot noise 513x on a model with zero real noise to correct, PEC
results that looked good on one draw and didn't replicate, a joint-frame
optimizer with catastrophic local minima, and — eventually — a result that
survived every adversarial control thrown at it.

Most of those were caught by hand, one at a time, over weeks.
**Every gate here traces to one of them**, and the docstrings name which,
so a reader can check each gate against the failure it exists to catch.

### Three distinctions a naive schema gets wrong

Each cost that project a real result.

**Bootstrap resampling is not replication.** The identical circuit set was
submitted to IonQ's simulator twice, independently, and the two energies
differed by 3.27 kcal/mol — three times the project's own reproducibility
bar. The standard "8-seed mean ± std" bars never showed it, because they
resample *one* submission's counts. Only independent submissions count
toward a replication target.

**An uncertainty bar is only as broad as what it varied.** The 0.115
kcal/mol headline was real under one fixed noise model and disowned once
the noise model's own parameters were allowed to vary (Q95 = 51.22, ~205x
over target). `UncertaintyCoverage` tracks four independent axes rather
than one scale — the joint Schmidt frame propagated noise-model
uncertainty but never cross-submission drift, and earlier headline numbers
had exactly the opposite gap.

**"Beats this baseline" and "reaches chemical accuracy" are different
claims.** The joint Schmidt frame is an 88.8% MSE reduction that still
does not reach chemical accuracy. Grading it against an absolute bar would
call a real result a failure.

---

## The QEM-Trust benchmark

Six real cases from that ledger, not synthetic fixtures. Each pinned to
both its verdict and its primary failure mode, so a change that stops
catching one fails loudly.

| case | verdict | why |
|---|---|---|
| abstract-gate folding | `INVALID` | compiler optimized the inserted gates back out |
| 513x ZNE blowup | `INVALID` | held-out validation tested interpolation while production extrapolated |
| 0.115 one-off PEC | `NOT ESTABLISHED` | every hard gate clean, chemical accuracy met — and the replicates were bootstrap resamples |
| cross-fitted manifold | `REFUTED` | median improved, 2/32 trials blew up to ~2100 kcal/mol |
| joint Schmidt frame | `VALID UNDER MODEL` | 88.8% MSE reduction, adversarially validated, no cross-submission drift in the bars |
| ancilla-QED + PEC | `PROMISING` | the current best result, correctly refused certification |

The third row is worth staring at. Every hard gate passes and it clears
chemical accuracy, and the auditor still refuses — on nothing more than
the observation that a single submission's bootstrap bar cannot speak to
reproducibility. That refusal lands *before* the expensive robustness
study that later disowned the number.

```bash
python run_benchmarks.py    # non-zero exit if any case audits wrongly
```

---

## The loop

```
claim -> audit -> what can still be wrong -> generate adversaries
      -> execute -> formal audit -> belief update -> next experiment
```

| | says |
|---|---|
| the proposer | "this attack should distinguish H1 from H2, and here is what each outcome would mean" |
| the executor | runs the attack |
| the gates | what actually happened |

The proposer commits to what each outcome means **before** anything runs,
so it cannot reinterpret a bad result afterwards. `AdversarialScientist`
has no API for issuing a verdict, and a test keeps it that way.

```bash
qem-auditor investigate my_experiment.json --qiskit --html report.html
```

On a real ZNE claim submitted at `optimization_level=3`, unaided:

```
round 1: NOT ESTABLISHED | 10 attacks | 2 falsified, 2 survived, 0 not run
    FALSIFIED by T_compiler
    FALSIFIED by T_compiler+T_extrapolation
round 2: INVALID
stopped: INVALID: the claim is disqualified, and no further attack changes that
```

The agent decides only **whether to keep going**. Every verdict comes from
the gates, it has no method containing "certify", and a test asserts an
agent run cannot upgrade an unproven claim.

---

## The failure grammar

Nine transformations, each from a failure this project actually suffered,
and they compose:

| | needs | status |
|---|---|---|
| `T_compiler` `T_extrapolation` `T_seed` | a backend adapter | **executable** |
| `T_label` `T_sign` `T_shot` | `fit` / `reconstruct` / `goodness_of_fit` | **executable** |
| `T_leakage` | `free_parameters` / `evaluate_at` | **executable** |
| `T_calibration` | `noise_parameters` / `evaluate_under_noise` | **executable** |
| `T_correlation` | `fit_without_structure` | **executable** |

All nine execute. The three optional capabilities are a handful of lines
each, and each unlocks an attack that would otherwise be taken on trust.

An attack is only an attack if it predicts **different** outcomes under
"genuine" and "artifact". `Prediction` refuses to be constructed otherwise
— an experiment both hypotheses predict identically will confirm whatever
you already believe.

Composition is deliberate: the H4 robustness envelope was `T_calibration`
composed with a coherent-error transformation, and it behaved nothing like
either alone (per-instance coherent bias gave Q95 = 827 kcal/mol, the same
magnitude applied per gate *type* gave 0.21).

A **self-reported** passing control does not close a question. Only a
control the auditor measured does — which is the whole reason an adversary
exists.

### Attacking your own fitting code

Implement three methods and half the grammar becomes executable against
your real pipeline:

```python
class MyPipeline:
    def fit(self, data): ...                  # your correction model
    def reconstruct(self, fit, data): ...     # the quantity you claim
    def goodness_of_fit(self, fit, data): ... # chi2/dof; lower is better

auditor.run_attacks(exp, reconstructor=MyPipeline(), fit_data=data)
```

The auditor never looks inside your fit. It corrupts the **data** going in
and watches whether the model noticed:

| attack | what it does | falsifies your claim if |
|---|---|---|
| `T_label` | permutes labels within each (slot, draw) | the shuffled fit is about as good |
| `T_sign` | negates every measured value | the model fits the negation as well |
| `T_shot` | resamples shot noise vs. subsamples your own MC draws | your own sampling dominates |

Three more become executable if you expose a little more. Each is
optional and checked for at runtime, so nobody implements machinery for
an attack that does not apply to their method:

```python
    # T_leakage
    def free_parameters(self): ...      # name -> (floor, nominal)
    def evaluate_at(self, name, value, data): ...

    # T_calibration
    def noise_parameters(self): ...     # name -> (low, high)
    def evaluate_under_noise(self, params, data): ...

    # T_correlation
    def fit_without_structure(self, data): ...
    def reconstruct_without_structure(self, fit, data): ...
```

`T_leakage` is worth a note on what it measures. The obvious test — "does
the estimate converge on the exact answer?" — is unusable, because an
auditor holding the exact answer does not need to be an auditor. The
usable statistic is whether the estimate stops responding to the **data**:
the auditor feeds real and *scrambled* measurements at each parameter
value and watches the gap between them collapse. That catches the
disqualified CDR behaviour without any ground truth at all.

**Why goodness-of-fit and not accuracy.** The auditor does not have the
true answer and should not be trusted with it if it did. What it can do is
destroy real structure while leaving the model's flexibility intact and
ask whether the model noticed. That comparison needs no ground truth,
which is exactly why it can be trusted.

`examples/attack_a_pipeline.py` runs two pipelines that look equally good
— and the over-parameterised one looks *better*, at `chi2/dof = 0`:

```
GENUINE  (per-label scale)     chi2/dof 1.03
  [SURVIVED ] T_label   1.03 -> 531 (515x worse when shuffled)

FLEXIBLE (per-measurement)     chi2/dof 0.00
  [FALSIFIED] T_label   0 -> 0 (1.0x -- absorbs shuffled data as readily)
```

Fit quality alone would have picked the wrong one.

**A limitation found by testing, not reasoned about in advance**: with only
**two** labels there is one non-identity permutation, so every group gets
the same swap — a systematic relabelling that any model with one free
parameter per label absorbs exactly. The attack reports *cannot judge*
rather than a pass. Three or more labels are needed.

---

## Deciding what to run next

### When more data cannot help

The most expensive experiment is one that cannot answer its question at
any sample size. `active_design.py` computes the Fisher information
`F = J^T Sigma^-1 J`; a near-zero eigenvalue names a direction the
experiment is blind to, and blindness is not cured by repetition.

On the H4-shaped case — a design sensitive only to `p_ZZ - p_GPi2` — it
finds the blind direction and ranks candidates by information along it per
dollar:

```
   1.939  zz_only_calibration       adds 50 along the weak direction at $25.79
 0.07326  full_sweep                adds 500 at $6,825.00
       0  more_shots_same_circuit   adds 0 along the weak direction for free
```

Free is not the same as useful.

### How much evidence is enough

`power.py` returns power, `required_n`, `minimum_additional` and expected
cost — and **refuses to size a sample when σ does not match the claim's
uncertainty scope**. That refusal is the project's most expensive
historical mistake restated statistically: the 8-seed bootstrap bars had a
within-submission σ near 0.0015 kcal/mol while independent submissions of
the identical circuit set differed by 3.27.

It sizes against the 95% *upper bound* on σ rather than the point
estimate, because a sample sd from 4 draws is not σ — it is a noisy guess
at σ, inflated ~3.0x at n=4.

A finding worth recording: powering against the scale the draws actually
differ on returns `required_n = 8`, independently rederiving the project's
own 8-draw convention from the statistics rather than from habit.

---

## Trust machinery

### Blind mode

An auditor evaluated on records whose expected verdict sits in the same
file is being graded on a task it can see the answer to — the
target-leakage failure its own benchmarks encode. A `BlindChallenge`
withholds every outcome quantity while keeping the methodology visible,
and `reveal()` refuses until a decision is committed.

All six benchmarks are answered correctly blind, the flagship ancilla-QED
case included: the auditor withholds certification and names the missing
evidence without ever seeing 0.0144 or the Q95.

### Provenance

Content-addressed evidence bundles, so "can you reproduce 0.018?" is
answerable. A digest names an exact combination of circuit, counts,
calibration, seeds, backend, environment and analysis version; `diff` says
*which* input moved, which a bare mismatch cannot. `is_reproducible` flags
an unset `PYTHONHASHSEED` and a dirty working tree.

Stated plainly: these are content hashes, **not signatures**. They detect
change; they do not authenticate.

### Grading a record vs verifying a claim

With no adapter, the auditor grades the record **as written** — it checks
whether you *claim* the ideal control passed. That is useful for your own
work and worthless against a stranger's, because a gate that trusts the
claimant is not a gate.

Every control carries `Provenance` — `SELF_REPORTED` or `MEASURED` — and
`CERTIFIED UNDER SCOPE` requires that everything the auditor *could* have
checked, it did.

---

## Adding a language model (optional, and free)

The auditor needs no model. One only widens the set of attacks considered
— a local model is plenty:

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

| the model proposes | what happens |
|---|---|
| an attack with different genuine/artifact outcomes | accepted, at lower discrimination than hand-written |
| an attack predicting the same thing either way | **rejected** — it will confirm whatever you already believe |
| an attack missing its statistic | **rejected**, naming the gaps |
| `"verdict": "PASS"`, `"ideal_control": true` | those fields **stripped**; the legitimate part kept |
| a hypothesis with no observable consequence | **rejected** — no experiment could ever address it |

Rejections are reported, never silently dropped. A model that proposes six
attacks of which two are non-diagnostic has told you something useful
about itself, and hiding that would make the auditor's own behaviour
unauditable.

---

## Layout

```
qem_auditor/
  schema.py         the experiment record
  record.py         read/write records as JSON
  integrity.py      is this record even readable?
  gates.py          14 gates, each from a real disqualification
  verdict.py        gate results -> one of 8 verdicts
  failure_modes.py  why it failed, and the cheapest fix
  adversary.py      generates falsification experiments
  executor.py       runs them; never pretends about what it could not run
  reconstruct.py    the interface that lets it attack your fitting code
  hypothesis.py     competing explanations, Bayesian, across experiments
  planner.py        what to run next, by information gain per dollar
  power.py          how much evidence would actually be enough
  active_design.py  when more data cannot help, and what to run instead
  provenance.py     content-addressed evidence bundles
  blind.py          audit without seeing the answer
  claim.py          what has been shown, what has not, what closes the gap
  agent.py          the loop, running by itself
  llm.py            provider-agnostic model access (optional)
  report.py         console and self-contained HTML reports
  frontdoor.py      bring a circuit, get a verdict
  api.py / cli.py   the entry points
  adapters/         execute controls instead of trusting them (needs qiskit)
benchmarks/         6 real QEM-Trust cases
examples/           4 runnable end-to-end demonstrations
tests/              407 tests
```

Run it:

```bash
python run_benchmarks.py                    # the 6 real cases
python run_audit_loop.py                    # the closed loop on real history
python examples/attack_a_pipeline.py        # genuine vs flexible (no qiskit)
python examples/verify_zne_claim.py         # end-to-end verification (qiskit)
python examples/adversarial_loop.py         # propose, execute, judge (qiskit)
python examples/autonomous_audit.py         # the agent, unattended (qiskit)
python -m unittest discover -s tests -t .   # full suite
```

Installing nothing runs everything except the adapters, which skip.

---

## What this is not

- **Not a full audit.** All nine transformations execute, but only against
  a pipeline that exposes the relevant interface. Implement three methods
  and six run; implement three more and all nine do. What was not run is
  reported as not run, never as passed.
- **Not an authenticator.** Provenance detects change, not forgery.
- **Not a source of truth about your method.** Target leakage, adversarial
  design and free-parameter floors are procedural and still rest on honest
  reporting. The auditor names them rather than papering over them.

## Roadmap

- **Tail resolution**: the calibration envelope defaults to 32 draws, so
  its Q95 is the second-largest deviation and estimates the tail coarsely.
  The H4 robustness studies used 29-97 expensive draws. A near-threshold
  ratio should be read as "run more draws", not as a pass.
- **Noisy-backend adapters**: the Qiskit adapter uses exact statevectors
  plus sampled shot noise, which is what the ideal control needs. Aer
  noise models, IBM or Quantinuum are the same interface with a different
  expectation oracle.
- **Likelihoods from a simulator**: hypothesis likelihoods are supplied
  per observation. Generating `P(D|H_i)` from a predictive model would
  strengthen the Bayesian layer and let the proposer check a new
  hypothesis has measurable consequences before it becomes a candidate.

## Relationship to quantum-chemistry-vqe

A separate project, not a fork. The H4/IonQ work stays where it is; this
repo reuses its real, disclosed results as the first benchmark suite for
auditing claims. The figures in `benchmarks/` are transcribed from that
ledger with iteration and task cited, not linked — an upstream correction
would need a manual re-sync.

## License

MIT
