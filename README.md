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

### Scoring an auditor, not an experiment

`run_benchmarks.py` pins qem-auditor's verdicts against themselves. That
is a useful regression test and it is circular as evidence: it shows the
auditor is stable, not that it discriminates. `qem_auditor.trust` points
the other way — it takes *any* auditor (this package's, a competitor's,
an LLM's, a person's) and scores it on the same six records.

```bash
python run_trust.py    # this package, plus the constant baselines
```

Three things make the score mean something:

**Errors are not symmetric, so accuracy is not the metric.** Condemning a
sound result wastes someone's month. Certifying an artifact puts a wrong
number into the literature. The report names those separately, and one of
them — endorsing a result the suite knows to be broken — disqualifies on
its own, at any aggregate score. That is the hard-gate rule from
`verdict.py` turned on the auditor itself.

**Hedging scores zero by construction, not by disapproval.** The headline
number is not credit; it is credit measured against the best a *constant*
answer achieves on the same suite. An auditor that replies `NOT
ESTABLISHED` to everything is never dangerously wrong and still scores
`-0.029`. So does every other constant answer, by definition.

**The suite reports its own resolution.** Six cases cannot support "94%
accurate". Every report carries the Wilson interval on its own exact-match
rate: six for six is `[0.61, 1.00]` — consistent with being wrong a third
of the time. The top grade is named `SUITE SATURATED` rather than
anything flattering, because a perfect score on six cases is a statement
about the suite's resolution, not the auditor's reliability.

```
  exact          6/6   95% CI [0.61, 1.00]
  credit         1.000
  best constant  0.433 (always INVALID)
  SKILL          +1.000
  attribution    5/5 causes named correctly
  GRADE: SUITE SATURATED (this suite cannot resolve further)
```

Verdict and diagnosis are scored on separate axes. An auditor that stops
a bad result for the wrong reason still stopped it, and folding the two
together would let a good diagnosis mask a false endorsement.

### Minimal pairs, and what saturating a suite means

The first thing the scoring layer reported about qem-auditor was `GRADE:
SUITE SATURATED` — 6/6 on the disclosed cases. That is a statement about
the suite, not the tool. Six cases had left six gates never once observed
failing, `mitigation_benefit` never observed running at all,
`independent_verification` never observed passing, and no case anywhere
near the top verdict — so an auditor that *could not produce*
`CERTIFIED UNDER SCOPE` scored the same as one that could.

`benchmarks/constructed.py` adds six **minimal pairs**: twelve records,
two at a time, identical except in one stated respect that is supposed to
move the verdict. Credit is all-or-nothing across a pair — half a pair is
what guessing looks like.

| pair | the one difference | verdicts |
|---|---|---|
| failed vs unrun | the ideal control failed, or was never run | `INVALID` / `NOT ESTABLISHED` |
| bootstrap vs independent | replicates resample one submission, or are eight | `NOT ESTABLISHED` / `CERTIFIED` |
| self-reported vs measured | who checked the passing controls | `PROMISING` / `CERTIFIED` |
| benefit vs none | under device noise the mitigation helps, or doesn't | `PROMISING` / `CERTIFIED` |
| model-only vs full scope | what the identical bars were computed over | `VALID UNDER MODEL` / `CERTIFIED` |
| absolute vs relative | the same numbers, claimed two ways | `REFUTED` / `CERTIFIED` |

The last one is the sharpest: 4.00 → 1.10 kcal/mol is a real 3.6x
reduction *and* a clear miss of chemical accuracy. The evidence refutes
one claim and supports the other. An auditor that grades the number
instead of the claim must get one of them wrong.

**Do the pairs earn their place?** Measured, not asserted.
`number_reading_auditor` is a baseline that reads the results table and
nothing else — roughly how most claims actually get assessed:

| | disclosed 6 | with the pairs |
|---|---|---|
| exact | 2/6 | 6/18 |
| skill | **+0.206** | +0.144 |
| pairs solved | — | **0/6** |
| grade | PARTIAL SKILL | **DISQUALIFIED** |

On the disclosed cases it looks like a passable auditor. Every pair holds
the numbers fixed and moves something it cannot see, so it solves none of
them — and endorses two records the suite knows to be broken.

**What the pairs do not do**, stated plainly: they do not break
qem-auditor's saturation. It scores 18/18 and 6/6 pairs. That is close to
guaranteed and it is not a compliment — a constructed case's truth
follows from the record by the same lattice this package implements, so
the tool that defines the rules keeps passing cases derived from them.
Constructed cases can only be hard for an auditor that reasons some other
way. Breaking this package's saturation needs *disclosed* cases where the
follow-up work disagreed with the verdict, and those are found by doing
experiments, not by writing records.

Constructed and disclosed cases are scored under separate labels and
never blended into one headline, because a tool that scores well only on
constructed cases has learned the schema, and a blended number would let
that look like having learned the physics.

Scoring your own auditor is four lines:

```python
from qem_auditor.trust import Answer, score
from benchmarks.suite import CASES

def my_auditor(experiment):        # sees the record, never the answer key
    return Answer(my_verdict_for(experiment))

score(my_auditor, CASES, "mine").print_report()
```

### A live audit, end to end

Everything above is a record *transcribed* from an experiment someone
already ran and already understood. `examples/live_h2_audit.py` is the
other thing: it runs the experiment now and hands the auditor numbers it
measured itself.

Real chemistry — H2 at 0.735 Å in STO-3G, two qubits, exact ground state
`-1.857275030 Ha` obtained by diagonalising the Hamiltonian rather than
from this package, so the verdicts can be checked against truth
afterwards. Two zero-noise-extrapolation protocols are audited, and the
auditor is never told which is which:

| | folds | fit | raw → mitigated | improvement |
|---|---|---|---|---|
| **A** | 1,3,5 | linear | 21.95 → 3.97 kcal/mol | 5.53x |
| **B** | 1,3,5,7,9 | quartic | 21.95 → 4.62 kcal/mol | 4.75x |

Read as a results table, both pass and B looks only slightly worse. Both
also win 8/8 paired trials on "does mitigation help under noise". A
reviewer would sign off on both.

What the auditor did instead — running the pipelines itself, not reading
about them:

- **On B's noiseless control it measured 20.0x error amplification.** With
  no physical noise to correct, that is the quartic fit amplifying shot
  noise. Nobody told it B was the aggressive one; it found the
  ill-conditioning by executing the pipeline against an exact model.
- **It ran the held-out check in the direction production uses** —
  hold out the lowest fold, fit only the ones above it, predict downward
  — with the tolerance set to the error each protocol claims. A costs
  5.12 kcal/mol to predict a fold it *did* measure, against the 3.97 it
  claims for the answer it didn't. B costs 8.15 against 4.62. Both fail.
- **B's spread and tail failed too**: 1/8 trials catastrophic, replicates
  disagreeing by 4.03 kcal/mol.

Both land `INVALID`, and A is the interesting one: it is the sober
protocol, it passes the noiseless control at a benign 1.9x, its
replicates are tight — and its extrapolation still cannot predict a
held-out point as accurately as it claims to predict the answer. That is
a real finding about a real run, not a fixture.

```bash
python examples/live_h2_audit.py    # ~13s, needs qiskit-aer
```

### The auditor made a prediction. It was right.

Of that run the auditor said, and refused to certify partly because of it:

> **CALIBRATION_MISMATCH** — the stated uncertainty never varied the
> assumed noise parameters, so it cannot speak to how far they sit from
> the true ones — a result under one fixed noise model predicts little
> about hardware

That is falsifiable, so `examples/real_device_audit.py` falsifies it
rather than repeating it. Nothing about the protocol changes — same
circuit, same folds, same fit, same shots, same seeds. The only thing
swapped is the noise: out goes the depolarizing model this project
invented, in comes IBM's **measured** calibration of qubits 119 and 120
on `fake_kyiv`, a 127-qubit Eagle processor (ECR error 0.31%, readout
error 2.93%, T1 387/258 µs). The pair was chosen by lowest gate error,
not by which one flatters the result, and a test asserts that.

**Protocol A's 5.53x improvement becomes 1.14x.**

The mechanism is not subtle once isolated, and the example isolates it by
switching each error off in turn:

| noise present | raw | mitigated | gain |
|---|---|---|---|
| gate errors only | 3.24 | 1.38 | 2.34x |
| gate + decoherence | 6.45 | 0.86 | **7.54x** |
| gate + **readout** | 33.43 | 30.31 | **1.10x** |
| all three, as measured | 36.46 | 30.95 | 1.18x |

Gate errors and decoherence both scale with the number of gates, so
folding amplifies them and extrapolation removes them — ZNE does its job,
and does it *better* with decoherence present. Readout error happens
**once, at measurement**, however many gates were folded. It does not
scale with the fold factor, so no extrapolation in that factor can reach
it. On this device readout error is 9x the two-qubit gate error, and ZNE
is structurally unable to touch the dominant term.

```bash
python examples/real_device_audit.py    # ~17s, needs qiskit-aer
```

The calibration is pinned in the example so it runs on `qiskit-aer`
alone; with `pip install 'qem-auditor[devices]'` a test compares the
pinned copy against the live snapshot, because a pinned number that
drifts from its source is a transcription claiming to be a measurement.

### Nine methods, two noise models, one auditor

ZNE is one method among many, so `benchmarks/methods.py` implements nine
and `examples/method_shootout.py` audits all of them. Each gets the same
access to the device — circuits in, counts out — and none holds the exact
answer. Two of the nine are there to be refused rather than ranked.

| method | invented noise | measured `fake_kyiv` |
|---|---|---|
| unmitigated | 21.43 ± 0.89 | 36.29 ± 1.02 |
| REM (readout) | 21.18 ± 0.94 | **6.08 ± 1.84** |
| ZNE (fold 1,3,5) | **4.24 ± 1.26** | 31.10 ± 1.27 |
| REM + ZNE | 3.95 ± 1.27 | **1.15 ± 1.35** |
| symmetry verification | 11.23 ± 0.93 | 6.88 ± 0.82 |
| CDR (Clifford regression) | 0.92 ± 0.84 | 1.29 ± 0.79 |
| PEC (model inversion) | 1.48 ± 0.93 | 16.96 ± 1.07 |
| dressed identity | 21.43 ± 0.89 | 36.29 ± 1.02 |
| oracle peek (fraud) | *0.43 ± 0.02* | *0.73 ± 0.02* |

Median error in kcal/mol over 16 seeds, **with the run-to-run spread** —
because without it these are fingerprints of one seeding rather than
measurements. This table quoted bare medians for a while, which was the
same over-claiming the package objects to everywhere else: REM+ZNE's
spread on the measured device *exceeds its median*, and it is not
distinguishable from CDR on any single run. The claims below are the ones
that survive their own spread.

Median error in kcal/mol over 8 independent runs. Four findings, and only
the first was expected:

**ZNE and REM change places.** ZNE is the best single method under the
invented noise and nearly useless under the measured one; REM is useless
under the invented noise and the largest single win under the measured
one. *Ranking mitigation methods on one noise model ranks nothing.*

**CDR is the only method that barely moves between them** (1.77 → 1.78).
It learns the noise map from data instead of assuming its structure. PEC,
which assumes the structure, is the one that collapses — 2.03 → 17.13 —
which is `CALIBRATION_MISMATCH` happening rather than being warned about.

**On accuracy alone the fraud wins both tables** — and it is worse than
that. On the measured device the fraud (0.737) and the best real method
(0.732) are **not distinguishable**: 0.005 apart against a spread of
0.72, so whichever printed first did so by luck. `power.compare` says it
would take about **143 runs each** to separate them. A leaderboard ranked
on error does not merely crown the fraud; it has nothing to crown it
*over*. What catches it is
`data_sensitivity`: scramble the outcome labels and every honest method's
answer moves as much as the raw estimate does (ratios 0.82–1.12), while
the fraud moves 2% as far (**0.020**). The bar sits in an empty gap, not
tuned against either side. That is `T_label` from the failure grammar,
pointed at a mitigation method.

**Two frauds need two detectors.** The `dressed identity` runs every
circuit ZNE runs and returns the raw value anyway — pipelines really do
end up here, via an extrapolation whose coefficients collapse to (1,0,0)
or a flag that silently turned the correction off. It *passes* the
scramble attack, because it genuinely reads the data; it just does
nothing with it. The improvement gate is what refuses it, and both
tables land it at `REFUTED`.

The auditor also refuses the method with the best median. **REM+ZNE
scores 1.56 on the measured noise and ranges 0.34 → 5.51 across runs** —
a 16x lottery — while CDR's median is 0.2 worse and far steadier. On the
single run a real experiment gets, the median winner is the one you can
least rely on. That is what `tail_risk` and `reproducibility` are for.

**A printed order is a claim.** `power.rank_with_ties` groups the methods
the runs cannot actually separate, so the table stops asserting an
ordering it did not earn:

```
not separated by these runs:
  oracle peek (fraud) = REM + ZNE = CDR
  REM (readout) = REM (tensored) = symmetry verification
  unmitigated = dressed identity
    0 apart -- and no number of runs establishes that two methods
    are identical, so none is quoted
```

The last line is the honest form of a question people ask constantly.
"Are these the same?" is not something a finite experiment answers, so
the report declines to name a run count rather than inventing one.

```bash
python examples/method_shootout.py            # ~3m, needs qiskit-aer
python examples/method_shootout.py --quick    # ~1m45, what CI runs
```

### From a refusal to a better experiment

An auditor that only says no wastes the time of the honest people it
exists to serve. `qem_auditor.prescribe` is the other half: given where
the error actually comes from, what is the best available thing to do?

The reason this can be more than folklore is that the shootout above
**measured** it. The organising idea is **scaling** — folding multiplies
the number of gates, so it multiplies every error that grows with gate
count and leaves untouched every error that does not:

| error source | scales as | so it can be reached by |
|---|---|---|
| shot noise | 1/√N — mitigation *amplifies* it | more shots, nothing else |
| readout | **constant under folding** | REM, symmetry post-selection, CDR |
| gate stochastic | with gate count | ZNE, PEC, CDR |
| decoherence | with duration | ZNE, CDR, a shorter circuit |
| ansatz | constant | **nothing** — change the circuit |

Every recommendation starts from an error budget, and the budget can be
had two ways. On a simulator you switch each noise source off and watch.
On hardware you can't — there is no exact answer to compare against,
which is why the experiment is being run — so `budget_from_calibration`
estimates it from published calibration data and your own gate counts.

On the measured `fake_kyiv` numbers the two agree that readout dominates
(estimate 75%, ablation 82%), which is what licenses using the cheap one
where the expensive one is impossible.

Feed that budget in and the advice is unsurprising only in hindsight:

```
  -> Clifford data regression (CDR)
       because: reaches READOUT, GATE_STOCHASTIC, DECOHERENCE, 93% of the error here
  -> REM then ZNE
  -> readout error mitigation (REM)

  Will NOT help here, and why:
  x  zero-noise extrapolation (ZNE): reaches only 9% of the error here.
     READOUT is outside what it can act on: unchanged by folding --
     extrapolation cannot reach it
  x  probabilistic error cancellation (PEC): withheld: its correctness rests
     entirely on the assumed noise model being the real one, and that has not
     been verified here. When the assumption fails this method does not
     degrade, it inverts -- measured at 2.03 -> 17.13 kcal/mol
```

**The `will NOT help` list is half the value.** ZNE is the method everyone
reaches for first, and on this device it is the one that cannot work.

Then `examples/prescribe_for_circuit.py` closes the loop: it runs the
recommendations *and* the methods it demoted, and reports whether the
advice held. It did — best recommended 1.56 kcal/mol against 30.95 for
the demoted one — and running it found two bugs in the prescriber on its
first pass. Shot noise was being computed as an absolute error while every
other term was a fraction, which inflated its share; and a "ceiling" was
being quoted from an *estimated* budget, which produced methods beating
their own bound by 2.5x. An estimate earns an ordering, not a number, and
now says so.

```bash
python examples/prescribe_for_circuit.py    # ~30s, needs qiskit-aer
```

### The cheapest fix is usually not a method — it is different qubits

`qem_auditor.layout` picks where to run. Measured on `fake_kyiv`, same
circuit, same shots, only `initial_layout` changed:

| placement | measured error |
|---|---|
| best available | **14.1 kcal/mol** |
| median | 36.6 |
| worst | **330.0** |

A **23x range**, free. A user unlucky in their layout pays that while
concluding their method is weak.

The part that is easy to get wrong: **which qubit property to optimise
depends on which error dominates.** Picking the lowest-gate-error pair is
the obvious move and it is wrong on a readout-dominated device — this
project's own device audit hand-picked qubits `(119, 120)` by gate error
and paid 36.46 kcal/mol where a budget-aware pick paid 13.87. So
placements are scored against the error budget's own weights, and the
same coupling map gives opposite answers for different budgets.

And a second-order effect worth knowing, because it inverts the answer:

| placement | raw | with REM |
|---|---|---|
| lowest readout `(96, 97)` | **13.87** | 11.02 |
| lowest gate error `(119, 120)` | 36.46 | **6.18** |

REM removes the readout error that made the first pair attractive and
leaves the gate error, where that pair is 2.2x worse. **Choosing qubits
and choosing a method are one decision, not two** — so `advise_layout`
takes an `after_method` and scores against what that method will *leave*.

The search is exhaustive and says so. On the 127-qubit `fake_kyiv`
lattice it enumerates every connected placement — 144 pairs, 7,104
ten-qubit sets — in under a fifth of a second, and at twelve qubits it
**refuses** rather than return a truncated best-of, because a search that
quietly stopped looking would be claiming an optimum it never found.
Pass `candidates=` to narrow to a region; the answer is then exhaustive
over the region you named, and the report says which region that was.

### Before mitigation: is it the circuit you said it was?

An 18-qubit phase oracle arrived from outside this project — a 64x64 logo
bitmap, a specification naming 1097 marked pixels, a depth target of 726,
and a submission file asserting `"verified": true`. Three findings, in
the order they have to come:

1. **It did not parse.** `mcx` and `mcz` are not OPENQASM 2.0 gates and
   are not in `qelib1.inc`.
2. **It did not compute its specification.** Shimmed so it would run, it
   marked **1 of 4096** basis states — the wrong one — and left its
   ancillas entangled with the input on *every* input, so it was not a
   phase oracle at all.
3. **Only then is mitigation a question**, and the answer is no.

`qem_auditor.reversible` is the check that was missing. An oracle, an
arithmetic block, a reversible subroutine — the parts of a program most
likely to carry a written specification — are built from gates that only
permute basis states and add phases. So they can be evaluated one basis
state at a time, tracking a bitstring and a sign: 4096 cheap evaluations
instead of 262,144 amplitudes. Exhaustive, exact, and cheap enough to be
mandatory.

```
inputs checked:   4096 (exhaustive, exact)
specification:    1097 marked
circuit marks:    1
ancillas not restored: 4096 of 4096 (100.0%)
-> the circuit does NOT implement its specification
```

The report never quotes accuracy alone. That circuit scores **73.2%**,
and so does one that does nothing at all, because the specification marks
a quarter of the space.

**What correct costs.** Cutting the marked set into 17 disjoint
rectangles and then 111 disjoint cubes gives an ancilla-free oracle —
one multi-controlled Z per cube, nothing to uncompute, verified on all
4096 inputs:

| | depth | 2-qubit gates | marks |
|---|---|---|---|
| as uploaded | 591 | 465 | 1 of 1097 |
| claimed target | 726 | 2,200 | — |
| correct, same 18-qubit width | 7,060 | 5,898 | **1097** |

**The depth target was met by not implementing the function.**

And then `feasibility` refuses to rank methods for it. On `fake_kyiv`,
5,898 two-qubit gates at 0.31% error survive one shot in 10⁸:

```
-> no method mitigates this: about 2e+17 shots to see the signal at all.
   Getting under 1305 two-qubit gates is the prerequisite, and it is a
   compilation problem, not a mitigation one.
```

Two smaller gaps the same circuit exposed, both now closed: **predicate
observables** (`predicate_expectation`), because "did we land in the
marked set" is diagonal and measurable in one setting but has no useful
Pauli expansion at 1097 states; and **tensored readout mitigation**,
because full REM needs 2ⁿ calibration circuits and refuses at seven
qubits — 262,144 of them here, against two at any width. It is
registered under its own name, not as full REM made cheaper: it assumes
readout errors factorise, which discards exactly the crosstalk between
neighbouring resonators.

Measured on `fake_kyiv` over 6 seeds, the two are **not distinguishable**:
means 0.05 kcal/mol apart against a spread of 1.69, and `power.compare`
puts the separating run count at about 18,000 each. An earlier draft of
this section quoted a single seed — 4.92 against 4.76 — which read as
"tensored is slightly worse" and was the same over-claiming from one run
that this package objects to everywhere else. What the runs support is
that the cheap method costs nothing measurable here, not that it costs a
little.

### Every audit makes the next one better

The catalogue is frozen: it knows what happened on two noise models and
one molecule, and it would still say so after a thousand real audits had
disagreed. `qem_auditor.ledger` is the part that accumulates. Each
measured outcome is appended, and later prescriptions cite what actually
happened on budgets like yours.

Three rules keep it from laundering guesses into evidence:

- **Content-addressed.** The same run recorded twice does not become two
  data points — a duplicate is caught by what it says, not by whether
  someone remembered to deduplicate.
- **Small samples say they are small.** Below five observations the
  corpus reports what it saw and declines to rank on it; the mechanism
  ordering stands. And it only reorders when it has enough observations
  for *every* method being compared, otherwise the best-studied method
  wins rather than the best one.
- **Disagreement is surfaced, not absorbed.** If measured outcomes stop
  matching what a method claims — either a method that should work and
  doesn't, or one that shouldn't and does — that is reported as a finding
  about this package's own mechanism table.

It is a plain JSON file: inspectable, diffable, deletable. A recommender
that improves in ways nobody can read is not an improvement anybody
should accept.

```bash
python examples/better_next_time.py    # ~15s, needs qiskit-aer
```

### What this circuit reminds the auditor of

The ledger remembers how methods *performed*, keyed on where the error
was. `qem_auditor.memory` remembers what was *found*, keyed on the
**circuit itself** — so a circuit arriving today is met with "the last
three things shaped like this failed the compiler check, look there
first" instead of a fresh start.

Both questions are worth asking. A budget says what will help; a circuit
says what went wrong last time, which is the better predictor of what
will go wrong this time — the same ansatz compiled the same way tends to
break the same way.

Circuits are keyed on **structure, not names**. Two groups call the same
ansatz different things and one group calls two circuits the same across
a refactor, so the fingerprint is gate counts, depth, gate alphabet and
observable shape — weighted towards the things that predict how a circuit
fails. Recall on the third audit of a family:

```
  2 similar circuits in memory:
    h2_ucc_tuesday (98% alike): INVALID -- unitary_equivalence
    h2_ucc_monday  (96% alike): INVALID -- unitary_equivalence
  Check these first, they failed most often on circuits like this one:
    unitary_equivalence: failed 2/2
  Attacks that earned their keep here before:
    T_compiler: found something 2/2 times
```

The expensive check goes first instead of last.

**Memory advises. It never convicts.** A circuit resembling three that
were `INVALID` is not thereby invalid — the gates decide on this
circuit's own evidence, and the example ends by auditing a clean circuit
that memory associates *only* with failures and watching it certify.
Precedent that could convict would be the worst feature in this package:
a method that failed once could never be shown working, and an auditor
would have become a reputation system. It holds by construction rather
than by care, because the gates are never handed the memory at all.

```bash
python examples/memory_pays_off.py    # ~1s, no dependencies
```

### Your circuit, not ours

Everything above runs on H2 or an Ising chain — this project's own
systems. The question that matters is whether any of it works on a
circuit nobody involved has seen, and until recently the answer was
**no, silently**.

The measurement layer assumed every term of the observable was all-Z or
all-X. True of H2 in its encoding, true of the Ising chain, false of most
things. And it did not check: an operator with a term like `XYZ` had its
bases popped arbitrarily from a set and was measured in one of them,
returning a number that was **wrong rather than absent** — while the
docstring directly above it said that silently averaging would be wrong.

`qem_auditor.estimation` replaces that with general Pauli estimation:
terms grouped into commuting sets that share a circuit, each qubit
rotated into the basis its term needs, anything unmeasurable refused by
name. It is checked against exact statevectors on random mixed-basis
observables.

So a hardware-efficient ansatz — three qubits, a five-term observable
including `XYZ`, three measurement settings worked out automatically:

| method | error | gain | sensitivity |
|---|---|---|---|
| oracle peek (fraud) | *0.0014* | *50x* | **0.020** |
| CDR | **0.0033** | 20.6x | 1.230 |
| REM + ZNE | 0.0109 | 6.2x | 0.765 |
| REM | 0.0133 | 5.1x | 1.203 |
| unmitigated | 0.0678 | 1.00x | 1.000 |
| dressed identity | 0.0678 | 1.00x | 1.000 |

Eight of nine methods run, with CDR's training circuits **generated**
rather than demanded. The ninth — symmetry post-selection — refuses,
because whether a state obeys a checkable symmetry is something a person
asserts and no error budget reveals. Refusing is the correct answer, not
a gap.

**And this found a real flaw in the fraud detector.** Scrambling a
method's data scrambled its *calibration* too, so a calibrated method
re-fit to the garbage and partly compensated: CDR's fitted slope flipped
from +1.24 to −0.32 while its target flipped sign as well, the two
cancelled, and it scored **0.390** — below the floor, next to the fraud,
for reading scrambled data twice rather than for not reading it.
Scrambling only the experiment's measurement clears it to **1.230** while
the fraud stays at 0.020.

Getting that right needed a second correction: a *folded* copy of the
experiment is still the experiment. Classifying it as calibration by
comparing circuit objects made ZNE look perfectly data-independent, so a
method now declares which of its measurements are calibration instead of
anything guessing.

```bash
python examples/bring_your_own_circuit.py    # ~40s, needs qiskit-aer
```

### Does any of it generalise?

Everything above was measured on H2 in STO-3G: two qubits, two CX gates.
One system. So `benchmarks/tfim.py` adds a transverse-field Ising chain —
Hamiltonian **constructed from a formula here** rather than transcribed
from anywhere, and with **depth as a knob**.

**The readout finding did not survive, and that is the mechanism working.**

| Trotter steps | 2q gates | readout share |
|---|---|---|
| 1 | 6 | 15.0% |
| 2 | 12 | 34.3% |
| 4 | 24 | 43.1% |
| 8 | 48 | 32.8% |

On H2 readout was **82%** of the error. Here it never exceeds 43%.
Readout error is charged once per measured qubit however many gates ran;
gate error is charged per gate. H2's budget was readout-heavy because H2
has two CX gates — not because readout dominates in general. The finding
was real; the generalisation would have been false.

And the auditor noticed, because it reasons from the budget rather than
from a rule. Same tool, different system, different advice — right both
times.

| method | TFIM (4 spins, 4 steps) | sensitivity |
|---|---|---|
| oracle peek (fraud) | *0.0082* | **0.020** |
| REM + ZNE | **0.1581** | 0.539 |
| CDR | 0.1801 | 1.090 |
| REM | 0.2272 | 0.536 |
| ZNE | 0.3463 | 1.142 |
| PEC | 0.3491 | 1.041 |
| unmitigated | 0.4095 | 1.000 |
| dressed identity | 0.4095 | 1.000 |

**What held on both:** the fraud tops the accuracy table and is caught
anyway; the dressed identity returns *exactly* the unmitigated value;
REM+ZNE is the best honest method; PEC underperforms wherever its assumed
model isn't the real one; symmetry verification correctly refuses where
no symmetry exists.

**What didn't:** readout dominance, and the size of the gains — 2.6x here
against 23x on H2.

The second system also found two bugs. REM's confusion matrix was
hardcoded 4×4 — fine on the only system it had ever run on, and a crash
on a four-qubit one. And the first CDR training set gave every training
circuit the **same** exact value, so the regression had slope zero and
returned a constant — which happened to beat every real method. The
scramble attack flagged it at `0.000` and was very nearly dismissed as a
false positive on a method known to be legitimate.

```bash
python examples/second_system.py    # ~80s, needs qiskit-aer
```

### All of it, from one command

The above were four capabilities and, for a while, four *libraries*:
`AuditResult.consult` and `.recalled` were never populated by anything,
the CLI had no way to ask, and nothing persisted. Someone running
`qem-auditor audit` got a verdict and none of the guidance.

Now the verdict arrives with its context and its remedy:

```bash
qem-auditor audit run.json --calibration device.json
```

```
NOT ESTABLISHED (1)
  - ideal_control: ideal/noiseless control did NOT recover a sane result

WHAT THIS REMINDS THE AUDITOR OF
  This exact circuit structure has been audited before:
    monday_run: INVALID -- failed: ideal_control
  Check these first: ideal_control failed 1/1
  Memory advises. The gates still decide this circuit on its own evidence.

WHAT TO DO ABOUT IT
  READOUT  82.8%  |  GATE_STOCHASTIC 10.0%  |  SHOT_NOISE 7.2%
  -> Clifford data regression (CDR), reaches 93% of the error here
  x  zero-noise extrapolation (ZNE): READOUT is unchanged by folding --
     extrapolation cannot reach it
```

Without `--calibration` it says so rather than going quiet, because an
error budget is not something it can invent:

```
NO REMEDY OFFERED
  No error budget was supplied, and one is not something this can
  invent. Pass --calibration with your device's published error rates
  and your own gate counts, and the verdict comes back with what to do.
```

`qem-auditor remember` shows what the corpus holds, and
`remember --circuit run.json` recalls against one circuit. A corpus that
silently steers recommendations and cannot be read is the thing this
package refuses everywhere else.

**The library is quiet; the command accumulates.** Importing a package
should not start writing files in someone's home directory, so
`Auditor()` learns nothing across calls unless handed a `Store`. The CLI
does open one — a tool that forgets between invocations is not much of a
tool — and prints where, the first time it creates anything.
`--no-store` opts out, `--store DIR` or `$QEM_AUDITOR_STORE` moves it.

The prescription also refuses to prescribe. When shot noise dominates it
says take more shots and warns that extrapolating first makes that term
*worse*. When the ansatz cannot represent the answer it recommends no
mitigation at all — that error is not noise, and removing noise more
precisely recovers a wrong answer more precisely.

---

**This run also found a bug in the auditor.** The gates separate "failed"
from "never run" with `is False`, which is exact — and `numpy.bool_` is
*equal* to `False` without *being* it. So
`extrapolation_in_domain = error <= tolerance`, the most natural line
anyone doing quantum work would write, stored a value that read as *not
recorded*. A measured failure disappeared and the verdict softened from
`INVALID` to `NOT ESTABLISHED` with nothing to show it had happened.
Control values are now normalised at the boundary, and anything
ambiguous — `1`, `0.0`, `"no"` — is refused rather than guessed at.

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

### Running under real device noise

The expectation source is the pluggable piece, so adding a backend is a
new source rather than a new adapter:

```python
from qem_auditor.adapters.qiskit_adapter import QiskitAdapter
from qem_auditor.adapters.sources import AerNoiseSource

auditor = Auditor(adapter=QiskitAdapter(source=AerNoiseSource(noise_model)))
```

That unlocks a control the noiseless path structurally cannot run:

| | asks |
|---|---|
| `ideal_control` | does this method **break** when there is no noise? |
| `mitigation_benefit` | does it **help** when there is? |

Passing one says nothing about the other. A do-nothing mitigator clears
the ideal control trivially — it cannot amplify noise it never touches —
and fails the benefit check outright. `examples/check_under_noise.py`
shows exactly that: both pass the ideal control, only one does anything.

**Two invariants worth knowing.**

The ideal control always runs **noiseless**, whatever source the adapter
was built with. Its entire content is "does this break with no noise to
correct", and running it through device noise would quietly turn it into a
different, much weaker check.

And `AerNoiseSource` verifies the noise actually took effect before
returning anything. Aer's `method="automatic"` silently returns a pure
state even with a noise model attached, as does a model whose basis the
circuit never hits — either way the numbers come back noiseless and
labelled noisy. Failing loudly is better.

`mitigation_benefit` needs the exact answer to be computable, which holds
in simulation and not on hardware large enough to matter. The auditor says
so rather than pretending otherwise.

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
  gates.py          15 gates, each from a real disqualification
  verdict.py        gate results -> one of 8 verdicts
  failure_modes.py  why it failed, and the cheapest fix
  prescribe.py      what to do about it: error budget -> ranked advice
  estimation.py     measuring any Pauli observable, not just ours
  reversible.py     does the circuit compute what its author said?
  layout.py         which qubits to run on, weighted by the budget
  ledger.py         the corpus that makes each audit inform the next
  memory.py         what this circuit reminds the auditor of
  store.py          where an auditor keeps what it has learned
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
  trust.py          scores an AUDITOR: skill, pairs, one fatal error
  report.py         console and self-contained HTML reports
  frontdoor.py      bring a circuit, get a verdict
  api.py / cli.py   the entry points
  adapters/         execute controls instead of trusting them (needs qiskit)
    sources.py      where expectation values come from: noiseless, or Aer noise
benchmarks/         6 real QEM-Trust cases
  suite.py          the same cases, scoreable by any auditor
  methods.py        9 mitigation methods to be audited, 2 of them frauds
  tfim.py           a second physical system, with depth as a knob
  constructed.py    6 minimal pairs: one difference, opposite verdicts
examples/           13 runnable end-to-end demonstrations
tests/              699 tests
```

Run it:

```bash
python run_benchmarks.py                    # the 6 real cases
python run_trust.py                         # score auditors against them
python run_audit_loop.py                    # the closed loop on real history
python examples/attack_a_pipeline.py        # genuine vs flexible (no qiskit)
python examples/verify_zne_claim.py         # end-to-end verification (qiskit)
python examples/adversarial_loop.py         # propose, execute, judge (qiskit)
python examples/autonomous_audit.py         # the agent, unattended (qiskit)
python examples/check_under_noise.py        # noiseless vs noisy (qiskit-aer)
python examples/live_h2_audit.py            # a real H2 run, audited live (qiskit-aer)
python examples/real_device_audit.py        # the same claim on measured IBM calibration
python examples/method_shootout.py --quick  # 9 methods, 2 noise models, audited
python examples/prescribe_for_circuit.py    # the fix, prescribed and then checked
python examples/better_next_time.py         # qubit choice, and the growing corpus
python examples/memory_pays_off.py          # circuit memory (no dependencies)
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
- **More sources**: IBM and Quantinuum are new `ExpectationSource`
  implementations, not new adapters — `exact`, `sampled`, and a
  `noiseless_twin` for the ideal control.
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
