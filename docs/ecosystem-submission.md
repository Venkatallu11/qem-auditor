# Qiskit Ecosystem submission

## Outcome: accepted, 2026-09-18

qem-auditor is a Qiskit Ecosystem member. The entry is
`resources/members/qemauditor_66544776.toml` upstream, carrying
`submission_number = 1355`, and it landed in `main` on 2026-09-18.

The mechanism this document predicted held exactly: the submission issue
came first (#1355), `qiskit-bot` wrote the member file and opened the
pull request from it, and a maintainer merged. Nothing here was written
by hand upstream, which is what generated the `uuid` and the badge.

Two fields came back different from what was proposed below, and the
accepted values are the ones that count:

| field | proposed here | accepted entry |
|---|---|---|
| category | Tooling | **Noise Management** |
| labels | error mitigation, research, quantum information, chemistry, AI/LLM | error mitigation, AI/LLM, benchmarking, **quantum information**, research |

`chemistry` was dropped and `benchmarking` added, which is the better
description: the benchmark suite is the part a stranger meets first, and
the chemistry is a test system rather than a subject. `maturity` and
`status` came back as `experimental` / `Very Early Project`, which is
what was asked for and what the project can support.

**The badge**, generated from the first segment of the entry's uuid:

```markdown
[![Qiskit Ecosystem](https://img.shields.io/endpoint?url=https://qiskit.github.io/ecosystem/b/66544776)](https://qisk.it/e)
```

That form -- `[![Qiskit Ecosystem](<url>)](https://qisk.it/e)` -- is the
one their own `badge_md` generator emits, not a guess. Once their daily
workflow adds a `[badge]` block to the member file, a `qisk.it/e-66544776`
short link resolves to the same image; the long form above does not wait
on it. The badge also carries membership STATUS: if the project stops
meeting the criteria it changes to "under revision", and to "alumni" if
it is removed. It is a live signal, not a trophy.

---

You are already an Ecosystem member: `quantum-chemistry-vqe` is
`resources/members/quantumche_e52f2069.toml` upstream, carrying
`submission_number = 1224`. This documents how that happened, so
qem-auditor takes the same path rather than a guessed one.

## Nobody writes the member file, and nobody opens the PR

Member entries ARE added by pull request -- "Add QuantumUQ (#1331)",
"Add SBD eigensolver (#1353)" -- but every one of those commits is
authored by `qiskit-bot`, not by the project's owner.

The proof is in the numbering. QuantumUQ's member file records
`submission_number = 1330`, and the pull request that added it was
**#1331**: the next number along. The issue comes first, the bot opens
the pull request from it, a maintainer merges.

So the sequence is:

    you open a submission issue        -> issue #N
    qiskit-bot writes the member file
    qiskit-bot opens the pull request  -> PR #N+1
    a maintainer merges

A hand-written member file in a hand-opened pull request skips the step
that generates the `uuid`, the badge URL, and the GitHub and PyPI
statistics -- which is why the file cannot be written by hand and why
this repository is not the place a pull request could come from. A pull
request adding a project to the Ecosystem has to change files in the
Ecosystem repository, so it can only come from a fork of that
repository; it cannot originate here.

**The door: https://qisk.it/add-to-ecosystem**

## Eligibility, checked against their stated criteria

| criterion | status |
|---|---|
| Builds on, interfaces with, or extends the Qiskit SDK | yes — the adapters execute controls through Qiskit and Aer |
| Compatible with Qiskit 2.0 or newer | yes — CI installs the current release and the full suite passes on 2.5.2 |
| OSI-approved licence | MIT |
| Adheres to Qiskit's Code of Conduct | `CODE_OF_CONDUCT.md` |
| Maintainer activity in the last 6 months | yes |

The `adapters` extra previously floored Qiskit at `>=1.0`, which was a
compatibility claim nothing here verified. It now says `>=2.0`, which is
what CI actually tests.

## The form, filled in

**Project name**

    qem-auditor

**Description** (under 135 characters)

    Audits quantum error-mitigation claims: runs the methods, attacks
    each one, and reports what the evidence actually supports.

**Contact email**

    alluvenkat11@gmail.com

**Category**

    Tooling

**Labels** (up to 5)

    error mitigation, research, quantum information, chemistry, AI/LLM

Your existing `quantum-chemistry-vqe` entry uses `AI/LLM`, which is a
real label even though it is absent from the issue form's dropdown --
worth asking for in a comment if the form will not offer it.

**Interface/API**

    Python, Command-line interface (CLI)

**Stability and support expectations**

    experimental

Chosen deliberately over `production-ready`. The API has moved this
month, and claiming stability this package has not demonstrated would be
the exact failure it audits other people for.

**Qiskit Pattern steps**

    Optimize, Execute, Post-process

Map is left out: this does not build circuits from a problem. It advises
on transpilation and placement (Optimize), runs mitigation methods and
their controls (Execute), and grades the result (Post-process).

**GitHub repository**

    https://github.com/Venkatallu11/qem-auditor

**Home page** — none

**Documentation** — none separate; the README is the documentation

**Package URLs** — none yet; not published to PyPI

## Worth doing before submitting

Publishing to PyPI is optional for membership, but the member entries
show that a package unlocks the version-compatibility and download
fields their tooling fills in automatically. Installing straight from
GitHub works today; a release would make the listing more informative.
