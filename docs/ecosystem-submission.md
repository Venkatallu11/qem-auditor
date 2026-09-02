# Qiskit Ecosystem submission

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
