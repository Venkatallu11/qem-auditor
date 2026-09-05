# Case study: an 18-qubit oracle that marked one state

An 18-qubit phase oracle was submitted to a Classiq logo challenge. It
came with a specification naming **1097 marked pixels**, a depth target
of **726**, and a submission file asserting `"verified": true`.

It marks **one** of 4096 basis states. The wrong one.

It was our own submission. That is the useful part: nothing here needed a
hostile reviewer or an adversary. It needed one exhaustive check that
nobody had run, on a circuit that compiles, runs, and produces plausible
output.

Every number below is reproduced by `examples/audit_an_oracle.py`, and
the commands to re-derive them from the original file are at the end.

---

## What arrived

| | |
|---|---|
| Qubits | 18 — `q[0:6]` x, `q[6:12]` y, `q[12:18]` ancillas |
| Specification | union of two rectangles and two disks over a 64×64 grid |
| Claimed marked pixels | 1097 |
| Depth target | 726, with 2200 two-qubit gates |
| Submission file | `"verified": true` |

The pixel count is the one thing that checked out. The four shapes really
do cover 1097 of the 4096 grid points.

---

## Finding 0: it does not parse

```
QASM2ParseError: 'mcx' is not defined in this scope
```

`mcx` and `mcz` are **not OPENQASM 2.0 gates** and are not in
`qelib1.inc`. The file compiles nowhere standard. They read as though
they should exist — most frameworks provide something by those names —
and they do not.

This cost an hour. `qem-auditor` now names it and the remedy in one line
when it refuses such a file.

Everything below required shimming those two gates so the circuit could
be read at all.

---

## Finding 1: it marks one pixel, and the wrong one

Exhaustive over all 4096 inputs, exact, no sampling:

```
inputs checked:   4096 (exhaustive, exact)
specification:    1097 marked
circuit marks:    1
ancillas not restored: 4096 of 4096 (100.0%)
specified but unmarked: 1097 of 4096 (26.8%)
marked but not specified: 1 of 4096 (0.0%)
-> the circuit does NOT implement its specification
```

The single marked state is `x=63, y=63`, which the specification does not
mark.

**Accuracy reads 73.2%.** So does a circuit that does nothing at all,
because the specification marks about a quarter of the space. This is why
the report never quotes accuracy without the counts beside it.

---

## Finding 2: three independent defects

### The phase gate is aimed at the wrong qubits

```qasm
mcz q[12],q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11];
```

The controls include all twelve **coordinate** qubits as well as the
predicate flag. The phase therefore fires only when every coordinate bit
is 1 — `x=63, y=63` — whatever the predicate computed. It should be a
phase on the flag alone: `z q[12]`.

That single line explains the entire "marks 1 pixel" result.

### The ancillas are never uncomputed

```qasm
x q[12];x q[13];x q[14];x q[15];x q[16];x q[17];
```

This *flips* the ancillas. It does not run the computation backwards.
They stay entangled with the input on **all 4096 inputs**, so the circuit
is not a phase oracle at all.

This is the defect that would have been hardest to find in the wild.
Inside amplitude amplification, entangled ancillas destroy the
interference the algorithm runs on. The symptom is a Grover search that
never converges — not an error message, not a crash. You would blame the
algorithm, the hardware, or the noise.

### Two of four shapes were never implemented

The file says so itself:

```
// For now, skip disk predicates (they're complex)
// The synthesis engine will handle this optimally.
```

The two disks are **357 of the 1097 pixels**. The synthesis engine was
not, in fact, going to handle it.

---

## Finding 3: the predicate logic is independently wrong

Repairing the phase gate and the uncompute — leaving the predicate
arithmetic exactly as written — gives:

| | |
|---|---|
| Ancillas dirty | **0** of 4096 (was 4096) |
| Circuit marks | 256 pixels |
| Square+Bar target | 740 pixels |
| **Of those, correct** | **10** |
| False positives | 246 |

So even with both structural defects fixed, the comparator logic gets
**10 of 740** pixels right.

The comments show where it goes astray:

```
// x[5] is the 32's bit. Since 26 < 32, x[5] = 0 always for x <= 26
// So x <= 26 iff q[5] = 0
```

The first line is true; the second does not follow from it. `x = 27`
through `31` also have bit 5 clear, so the condition is necessary and not
sufficient — and `x <= 26` is exactly the bound the Square depends on.

Sixty lines further down, working the Bar's `x <= 49`, the derivation is
abandoned in the open:

```
// This is getting complex. Alternative:
// For Bar: x in [26,49]. Let me use simpler bounds:
// 49 = 110001, 26 = 011010
// x >= 26 AND x <= 49
```

The requirement is restated and then **no gate is ever emitted for it**.
Tracing the ancilla that was supposed to hold it makes this exact — here
is every line in the file that touches `q[13]`:

```
 53: cx q[5],q[13];              // computed, for the SQUARE's x <= 26
 87: ccx q[12],q[13],q[16];      // read, for the Square
 95: x q[12];x q[13];...         // the bogus "uncompute"
142: ccx q[12],q[13],q[16];      // read again, for the BAR
146: x q[12];x q[13];...
```

Line 53 is the only computation. The Bar at line 142 reads whatever the
Square left behind, inverted by line 95. It is not a wrong value for
`x <= 49` — it is the Square's `x <= 26`, negated, standing in for a
condition that was never computed.

And line 53 does not compute what its own comment says. `cx q[5],q[13]`
leaves `q[13] = q[5]`, but the comment reads `// q[13] = NOT q[5]`. The
negation needs an `x q[13]` that is not there, so the flag is **true
exactly when `x >= 32`** — the opposite of the `x <= 26` it stands for.

So all four shapes fail, for four different reasons:

| shape | why |
|---|---|
| Square | its `x <= 26` flag is inverted (missing `x`), and the bound is necessary-not-sufficient anyway |
| Bar | its `x <= 49` is never computed; it reads the Square's stale flag |
| Disk 1 | not implemented |
| Disk 2 | not implemented |

That is the mechanism behind the 10-of-740 number above.

---

## What a correct implementation costs

Cutting the marked set into **17 disjoint rectangles**, then **111
disjoint cubes**, gives an oracle that is one multi-controlled Z per
cube. Because the cubes are disjoint, at most one matches any input, so
the phases cannot double-count — and it needs **no ancillas at all**,
which removes the entire failure mode the submission died of. There is
nothing to uncompute, so there is no uncompute to get wrong.

Verified on all 4096 inputs by the same call that audited the original.

| | depth | 2-qubit gates | marks | qubits |
|---|---|---|---|---|
| As uploaded | 591 | 465 | **1** of 1097 | 18 |
| Claimed target | 726 | 2,200 | — | 18 |
| Correct, ancilla-free | 25,827 | 17,340 | **1097** | 12 |
| Correct, ancilla-assisted | 7,060 | 5,898 | **1097** | 18 |

**The depth target was met by not implementing the function.**

One honest caveat: the correct versions here are the simplest
obviously-right constructions, not optimised ones. A serious synthesis
lands below 5,898. But the gap to 726 is a factor of eight at the same
width, and it is not closed by cleverness alone.

---

## Can any hardware run a correct one?

No. Not on anything available, and not by a small margin.

A correct implementation at 5,898 two-qubit gates, across the machines in
`qem_auditor.devices`:

| machine | executed gates | survival | shots to see the signal |
|---|---|---|---|
| Quantinuum H2 | 5,898 | 1.4×10⁻⁴ | 5×10⁸ |
| IonQ Forte | 5,898 | 5.0×10⁻¹¹ | 4×10²¹ |
| IBM Heron r2 | 14,745 | 1.2×10⁻¹³ | 7×10²⁶ |
| IonQ Aria | 5,898 | 3.5×10⁻¹⁶ | 7×10³¹ |
| IBM Eagle r3 | 14,745 | 6.6×10⁻²¹ | 2×10⁴¹ |
| Rigetti Ankaa-3 | 14,745 | 9.5×10⁻⁹⁸ | 10¹⁹⁴ |

Two things stand out.

**Connectivity is worth four orders of magnitude here.** The
superconducting rows execute 14,745 gates for a circuit that writes
5,898, because distant entangling gates become SWAP chains. The
trapped-ion rows execute what was written.

**Even the best case is hopeless.** Quantinuum H2 needs 5×10⁸ shots. The
useful output is not "use CDR" — it is a gate budget: **get under 3,031
two-qubit gates on H2, or 1,305 on Eagle**, and that is a compilation
problem, not a mitigation one.

---

## What this says about the workflow

**Nothing here throws an exception.** The circuit compiles. It runs. It
returns a distribution. Every failure is silent, and the loudest signal —
`"verified": true` — was written by the same process that produced the
defect.

**Simulation would not have caught it.** An ideal simulation and a noisy
simulation of the *same wrong circuit* agree with each other. The
disagreement a sim-vs-sim pipeline looks for never appears. A
ground-truth check on sampled counts cannot help either: resolving a
1097-state marked set inside a 4096-state space from 4096 shots is not
a statistics problem anyone can win.

**Enumeration can, and cheaply.** These circuits only permute basis
states and add phases, so each input can be evaluated by tracking a
bitstring and a sign — 4096 cheap steps rather than a 262,144-amplitude
state vector. That is why the check is affordable enough to be mandatory
rather than optional.

**And it belongs first.** Auditing mitigation on this circuit would have
produced a precise, careful, entirely worthless answer.

---

## Reproduce it

```bash
pip install -e ".[adapters]"
python examples/audit_an_oracle.py
```

To run the check on your own oracle:

```python
from qem_auditor.reversible import preflight_gate

verdict = preflight_gate(circuit, predicate=my_spec, n_inputs=4096,
                         encode=my_encoding, ancillas=range(12, 18))
# {"verdict": "GO" | "BLOCK" | "SKIP", "reason": ..., "report": ...}
```

The predicate is yours and is never inferred. An auditor that writes both
sides of the comparison is not auditing anything.
