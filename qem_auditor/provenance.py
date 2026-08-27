"""Tying a claim to the artifacts that produced it.

"Can you reproduce 0.018?" is only answerable if something records which
circuit, which counts, which calibration, which seeds and which analysis
version produced it. Without that, a number and a rerun that disagrees
leave no way to tell whether the method is unstable, the input changed,
or the analysis moved underneath it -- and this project spent real
iterations on exactly that ambiguity.

An `EvidenceBundle` is a content-addressed manifest: every input is
hashed, the hashes are hashed, and the resulting digest names that exact
combination. Two bundles with the same digest describe the same
experiment; two that differ tell you precisely which input moved.

What this does and does not give you:

- It **detects** change. Any altered input changes the digest.
- It does **not** authenticate. These are SHA-256 content hashes, not
  signatures: they prove a bundle is self-consistent, not that a
  particular person produced it. Someone who rewrites the artifacts and
  recomputes the digest gets a valid bundle. Guarding against that needs
  signing, which is deliberately out of scope here rather than half-done.

Stdlib only: hashlib and json.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

MANIFEST_VERSION = 1


class ProvenanceError(ValueError):
    """A bundle that cannot be built or verified as asked."""


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_text(text: str) -> str:
    return hash_bytes(text.encode("utf-8"))


def hash_json(obj: Any) -> str:
    """Hash a structure canonically.

    `sort_keys` is load-bearing: without it the same counts dictionary
    hashes differently depending on insertion order, which would make
    every bundle spuriously unique and the whole mechanism useless. This
    is the same class of bug as the hash-order nondeterminism that tipped
    this project's nonconvex solver into different local optima.
    """
    return hash_text(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                default=str))


def hash_file(path: str | Path, chunk: int = 1 << 20) -> str:
    p = Path(path)
    if not p.is_file():
        raise ProvenanceError(f"not a file: {p}")
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_commit(repo: str | Path = ".") -> Optional[str]:
    """The current commit, or None outside a repo or with git unavailable.

    None rather than an exception: analysis code legitimately runs outside
    a checkout, and a bundle that records "no commit" is more useful than
    one that refuses to exist.
    """
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def git_is_dirty(repo: str | Path = ".") -> Optional[bool]:
    """Whether the working tree has uncommitted changes.

    A commit hash on a dirty tree does not identify the code that ran, so
    a bundle records this and `verify` reports it. A result produced from
    uncommitted work is not reproducible from the commit it names.
    """
    try:
        out = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return bool(out.stdout.strip())


def environment_fingerprint(packages: tuple[str, ...] = ("qiskit", "numpy")) -> dict:
    """Enough of the environment to explain a numerical difference.

    Includes PYTHONHASHSEED explicitly: an unset hash seed is exactly what
    made this project's identical-seed reruns land in different local
    optima 40-50% of the time, and a bundle that omitted it would be
    unable to explain the difference afterwards.
    """
    versions = {}
    for name in packages:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "unset"),
        "packages": versions,
    }


@dataclass
class EvidenceBundle:
    """An immutable manifest of everything that produced a result."""

    experiment_id: str
    claim: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)
    """name -> sha256. Circuits, raw counts, calibration data, analysis inputs."""

    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None
    seeds: dict[str, int] = field(default_factory=dict)
    backend_id: str = ""
    backend_calibration_timestamp: str = ""
    analysis_version: str = ""
    environment: dict = field(default_factory=dict)
    manifest_version: int = MANIFEST_VERSION

    def manifest(self) -> dict:
        """The canonical structure the digest is computed over."""
        return {
            "manifest_version": self.manifest_version,
            "experiment_id": self.experiment_id,
            "claim": self.claim,
            "artifacts": dict(sorted(self.artifacts.items())),
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "seeds": dict(sorted(self.seeds.items())),
            "backend_id": self.backend_id,
            "backend_calibration_timestamp": self.backend_calibration_timestamp,
            "analysis_version": self.analysis_version,
            "environment": self.environment,
        }

    @property
    def digest(self) -> str:
        """Names this exact combination of inputs."""
        return hash_json(self.manifest())

    @property
    def short_digest(self) -> str:
        return self.digest[:12]

    @property
    def is_reproducible(self) -> tuple[bool, list[str]]:
        """Whether this bundle could actually be reproduced from what it records."""
        problems = []
        if not self.artifacts:
            problems.append("no artifacts recorded -- nothing to reproduce from")
        if self.git_commit is None:
            problems.append("no git commit: the analysis code is not identified")
        elif self.git_dirty:
            problems.append(
                "the working tree was dirty, so the recorded commit does not identify "
                "the code that ran")
        if self.environment.get("pythonhashseed", "unset") == "unset":
            problems.append(
                "PYTHONHASHSEED was unset: hash-order effects can change results "
                "between runs without any input changing")
        if not self.seeds:
            problems.append("no random seeds recorded")
        return (not problems), problems

    def diff(self, other: "EvidenceBundle") -> dict[str, tuple[Any, Any]]:
        """Exactly which inputs differ. The question a failed reproduction
        needs answered, and the one a bare digest mismatch cannot."""
        mine, theirs = self.manifest(), other.manifest()
        changed: dict[str, tuple[Any, Any]] = {}
        for key in sorted(set(mine) | set(theirs)):
            a, b = mine.get(key), theirs.get(key)
            if a == b:
                continue
            if key == "artifacts" and isinstance(a, dict) and isinstance(b, dict):
                for name in sorted(set(a) | set(b)):
                    if a.get(name) != b.get(name):
                        changed[f"artifacts.{name}"] = (a.get(name), b.get(name))
                continue
            changed[key] = (a, b)
        return changed

    def to_json(self, indent: int = 2) -> str:
        return json.dumps({**self.manifest(), "digest": self.digest},
                          indent=indent, sort_keys=True, default=str)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceBundle":
        p = Path(path)
        if not p.exists():
            raise ProvenanceError(f"no such bundle: {p}")
        data = json.loads(p.read_text())
        recorded = data.pop("digest", None)
        data.pop("manifest_version", None)
        bundle = cls(**data)
        if recorded is not None and bundle.digest != recorded:
            raise ProvenanceError(
                f"bundle at {p} does not match its own digest: recorded {recorded[:12]}, "
                f"computed {bundle.digest[:12]}. The manifest was edited after it was "
                f"written.")
        return bundle


def build_bundle(experiment_id: str, claim: str = "",
                 artifact_paths: dict[str, str | Path] | None = None,
                 artifact_objects: dict[str, Any] | None = None,
                 seeds: dict[str, int] | None = None,
                 backend_id: str = "",
                 backend_calibration_timestamp: str = "",
                 analysis_version: str = "",
                 repo: str | Path = ".") -> EvidenceBundle:
    """Build a bundle, hashing files and in-memory objects alike."""
    artifacts: dict[str, str] = {}
    for name, path in (artifact_paths or {}).items():
        artifacts[name] = hash_file(path)
    for name, obj in (artifact_objects or {}).items():
        if name in artifacts:
            raise ProvenanceError(
                f"artifact {name!r} given both as a file and an object -- one name, "
                f"one artifact, or the digest means nothing")
        artifacts[name] = hash_json(obj)
    return EvidenceBundle(
        experiment_id=experiment_id,
        claim=claim,
        artifacts=artifacts,
        git_commit=git_commit(repo),
        git_dirty=git_is_dirty(repo),
        seeds=dict(seeds or {}),
        backend_id=backend_id,
        backend_calibration_timestamp=backend_calibration_timestamp,
        analysis_version=analysis_version,
        environment=environment_fingerprint(),
    )
