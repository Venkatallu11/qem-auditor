# Contributing

The bar here is unusual, so it is worth stating before you spend an
evening on a patch.

## The one rule

**A claim has to carry its evidence.** Not an argument that it should
hold — a number, from a run, that someone else can reproduce. This
applies to code comments, to the README, and to this file.

Three real examples of that rule biting the project itself:

- The pricing model said cost was per-circuit, from two bills that
  agreed. Both were under a per-job minimum nobody had looked up. At
  2,000 shots the real answer was 2.5× what the model said.
- The shootout printed "the fraud scores 0.020 against 0.6–1.1 for
  everything real." True when written; a later method scored 0.280. It
  now computes that range from the rows it just printed.
- A held-out check validated three extrapolators with a straight-line
  fit none of them uses. It passed, and certified nothing.

None of these were caught by review. Two were caught by tests, one by a
reader asking a blunt question. Write the test that would have caught
yours.

## What gets a patch merged

1. **Tests that could fail.** A test asserting what the code obviously
   does is scaffolding, not evidence. The good ones here encode a
   failure someone actually hit.
2. **CI green in both conditions.** `core (no dependencies)` runs the
   suite with no qiskit at all; `adapters (qiskit)` runs it with qiskit
   plus every example end to end. The core must stay dependency-free —
   the thing deciding whether a result is trustworthy should not need a
   stack to run.
3. **Comments that say why, not what.** The code says what.

## What gets a patch turned down

- A method added to the catalogue without a circuit count in
  `cost.py`. A method priced at one circuit because nobody looked is
  how a prescription blows a hardware budget.
- A number quoted from a previous run beside a number read from this
  one. Half measured and half remembered is worse than either.
- Softening a verdict because it looks harsh. An unrun hard gate is not
  a pass; that is the whole design.
- Anything that lets the AI layer decide a verdict. It proposes
  experiments and writes prose. Plain Python decides what passed.

## Running it

```bash
pip install -e ".[adapters,devices]" qiskit-aer
python -m unittest discover -s tests -t . -v     # the full suite
python -m unittest discover -s tests -t .        # quieter
```

To check the dependency-free condition the way CI does, install without
extras in a clean environment and run the same command. The
qiskit-dependent tests skip and nothing fails — deliberately not quoted
as a skip count here, because that number moves with every test added
and a stale one in a file about stale numbers would be embarrassing.

The examples are executable documentation and CI runs every one of
them. If you change behaviour an example prints, run that example and
paste the new output in your pull request.

## Reporting something wrong

A bug report that says "this number looks wrong to me" is welcome and
has already improved this project more than once. You do not need to
find the cause. Say what you ran, what it printed, and what you
expected — if the answer turns out to be that the tool was right, that
exchange belongs in the README too.
