# Qiskit Ecosystem submission

The Qiskit Ecosystem does **not** take pull requests for joining. Its
`CONTRIBUTING.md` says so directly, and the member files under
`resources/members/` confirm it: they carry a generated `uuid`, GitHub
star counts, PyPI download figures and a badge URL, none of which a
contributor writes by hand. A hand-authored member file would be the
wrong door.

The door is a submission issue: **https://qisk.it/add-to-ecosystem**

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

**Category**

    Tooling

**Labels** (up to 5)

    error mitigation, research, utility-scale, quantum information, chemistry

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
