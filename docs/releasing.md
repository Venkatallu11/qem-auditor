# Releasing to PyPI

Publishing is irreversible in a way most things here are not. A version
number on PyPI can be yanked but never replaced, and a project name is
claimed for good. So the release path is built to make the two failures
that actually happen impossible rather than unlikely: publishing a
version nothing in the repository names, and publishing from a tree that
was never tested.

## How it works

`.github/workflows/publish.yml` publishes on a `v*` tag using **PyPI
Trusted Publishing** (OIDC). There is no API token in this repository,
none in its secrets, and none in anyone's shell history. GitHub proves
to PyPI that this workflow, in this repository, in the `pypi`
environment, is what is asking. A token that does not exist cannot leak.

Before publishing, the workflow:

1. **Checks the tag against `pyproject.toml`.** A `v0.2.0` tag on a tree
   declaring `0.1.0` fails the build rather than putting a version on
   PyPI that nothing names.
2. **Runs the full suite** with adapters and devices installed. The
   suite is green on `main` before any tag exists, but a release is the
   one build where "it passed earlier" is not good enough.
3. **Runs `twine check --strict`**, so a README that renders badly on
   PyPI is caught before it is published rather than after.

## One-time setup — a human has to do this

Trusted publishing has to be authorised from the PyPI account, and
nobody else can do it. At <https://pypi.org/manage/account/publishing/>,
add a **pending publisher**:

| field | value |
|---|---|
| PyPI project name | `qem-auditor` |
| Owner | `Venkatallu11` |
| Repository name | `qem-auditor` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

"Pending" is the right kind for a project that does not exist on PyPI
yet: the first successful publish creates it and converts the pending
publisher into a real one.

## Cutting a release

```bash
# 1. The version lives in exactly one place.
#    Edit pyproject.toml: version = "0.1.0"

# 2. Tag it, matching that version exactly.
git tag v0.1.0
git push origin v0.1.0
```

That is the whole release. The workflow builds, tests, checks and
publishes. Watch the run; if the tag/metadata check fails, delete the
tag (`git tag -d v0.1.0 && git push origin :v0.1.0`), fix the version,
and tag again — nothing has been published at that point.

`workflow_dispatch` runs the same build **without** publishing, for
inspecting the artifacts first.

## Versioning

`0.x` while the API is still moving, which it is — and the Qiskit
Ecosystem entry says `maturity = experimental` for the same reason. The
PyPI classifier says `Development Status :: 3 - Alpha`. Those three
statements have to keep agreeing with each other; a package that calls
itself experimental in one registry and beta in another is making a
claim it has not earned in whichever one flatters it.

## After the first publish

The README's install instructions can then say `pip install
qem-auditor`, and the Qiskit Ecosystem entry picks up version and
download fields automatically — its tooling reads them from PyPI, which
is why `packages = []` sits in the member file today.

Do not add the `pip install` line before the first release succeeds. A
README that tells people to install something that is not there is a
small lie of exactly the kind this project exists to catch.
