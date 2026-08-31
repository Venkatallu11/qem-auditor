"""Where an auditor keeps what it has learned.

`memory.py` and `ledger.py` each know how to save themselves, and until
this existed nothing called them. The capability was real and unreachable:
an auditor that could remember, wired to nothing that would make it.

This bundles the two behind one directory and one save, so "every audit
makes the next one better" is something that happens rather than
something a user could arrange if they wrote the plumbing themselves.

One deliberate asymmetry. The library never writes to disk unless a store
is handed to it -- importing a package should not create files in
someone's home directory, and an auditor that silently accumulated a
record of everything it had been shown would be a surprising thing to
have installed. The CLI does open a store by default, because a tool
that forgets everything between invocations is not much of a tool. So the
default is: quiet as a library, accumulating as a command, and the report
says which one is in force.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .ledger import EvidenceLedger, Observation
from .memory import CaseMemory, CircuitFingerprint, PastCase, case_from_audit

#: Overridable so a project can keep its corpus beside the project rather
#: than in a home directory shared with unrelated work.
ENV_VAR = "QEM_AUDITOR_STORE"
DEFAULT_DIRECTORY = "~/.qem-auditor"

MEMORY_FILE = "memory.json"
LEDGER_FILE = "ledger.json"


def default_directory() -> Path:
    return Path(os.environ.get(ENV_VAR, DEFAULT_DIRECTORY)).expanduser()


@dataclass
class Store:
    """A directory holding what past audits found, and how methods did.

    Both halves are plain JSON. Someone who wants to know what this thing
    believes can read it, diff it, edit it, or delete it -- which is the
    whole reason neither half is a model.
    """

    directory: Optional[Path] = None
    memory: CaseMemory = field(default_factory=CaseMemory)
    ledger: EvidenceLedger = field(default_factory=EvidenceLedger)

    @classmethod
    def open(cls, directory=None) -> "Store":
        """Load a store, creating nothing until something is saved."""
        path = Path(directory).expanduser() if directory else default_directory()
        return cls(directory=path,
                   memory=CaseMemory.load(path / MEMORY_FILE),
                   ledger=EvidenceLedger.load(path / LEDGER_FILE))

    @classmethod
    def ephemeral(cls) -> "Store":
        """A store that accumulates within a session and writes nothing.

        For tests, and for anyone who wants the benefit of memory across
        one run without leaving a trace afterwards.
        """
        return cls(directory=None)

    @property
    def persistent(self) -> bool:
        return self.directory is not None

    def save(self) -> bool:
        """Write both halves. Returns False for an ephemeral store."""
        if self.directory is None:
            return False
        self.directory.mkdir(parents=True, exist_ok=True)
        self.memory.save(self.directory / MEMORY_FILE)
        self.ledger.save(self.directory / LEDGER_FILE)
        return True

    # -- what an audit puts in ----------------------------------------
    def remember_audit(self, experiment, report, fingerprint: CircuitFingerprint,
                       analysis=None, attacks_that_fired=()) -> bool:
        return self.memory.remember(case_from_audit(
            experiment, report, fingerprint, analysis, attacks_that_fired))

    def record_outcome(self, observation: Observation) -> bool:
        return self.ledger.record(observation)

    def summarise(self) -> str:
        where = str(self.directory) if self.persistent else "in memory only"
        return (f"  store: {where}\n"
                f"    {len(self.memory)} circuit"
                f"{'s' if len(self.memory) != 1 else ''} remembered, "
                f"{len(self.ledger)} method outcome"
                f"{'s' if len(self.ledger) != 1 else ''} recorded")
