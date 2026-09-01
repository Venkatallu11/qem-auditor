"""MCP server: the auditor as a live service.

Six tools, and deliberately none of them talks to a quantum computer.
Listing backends, submitting jobs, reading queues and estimating vendor
cost are served by `quantum-verifier`; a second implementation of that
surface would make two projects that are each worse than one. What is
here is the half that project does not do:

    quantum-verifier  ->  will the right circuit survive this hardware?
    qem-auditor       ->  is it the right circuit, and is the number real?

Every tool is a judgement with its reasoning attached, and several of
them can answer "no". `check_circuit_computes` returns BLOCK on a circuit
that does not implement its own specification. `check_feasibility`
refuses to rank methods for an experiment with no signal left.
`recommend_mitigation` disqualifies a method that does not read its own
data, however accurate it looks -- which matters, because in this
project's own suite the deliberate fraud is not distinguishable from the
best real method on accuracy.

Run:  python mcp_server.py          (needs: pip install -e ".[service]")

The tool functions live in `qem_auditor/service.py` and are tested
directly, so this file stays a thin binding over code that is already
checked.
"""
from __future__ import annotations

import json
import sys

from qem_auditor.service import TOOLS

DESCRIPTIONS = {
    "check_circuit_computes":
        "Exhaustively and exactly check whether a circuit marks the states its "
        "author says it does. Runs before any simulation, because an ideal and "
        "a noisy simulation of the same wrong circuit agree with each other. "
        "Returns GO, BLOCK, or SKIP.",
    "check_feasibility":
        "Is there any signal left for a mitigation method to improve? Ask "
        "before ranking methods: none of them creates an estimate.",
    "recommend_mitigation":
        "Which method to use, why, and what the runs do not settle. "
        "Disqualifies methods that do not read their own data, and returns "
        "statistically inseparable methods as a tier rather than an order.",
    "compare_two_methods":
        "Can these runs tell two methods apart? If not, how many runs would.",
    "falsify_claim":
        "Is the effect the mechanism or the apparatus? Compare a circuit "
        "against its entanglement-free control.",
    "budget_from_device":
        "Where the error comes from, from calibration numbers alone. Supports "
        "an ordering of methods, never a quoted ceiling.",
}


def build_server():
    """Construct the MCP server. Imported lazily so this module stays
    importable -- and testable -- without the MCP SDK installed."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("qem-auditor")
    for name, function in TOOLS.items():
        server.add_tool(function, name=name, description=DESCRIPTIONS[name])
    return server


def describe() -> dict:
    """The tool manifest, without needing the SDK. Used by `--list` and by
    the tests, so the advertised surface cannot drift from the real one."""
    return {"name": "qem-auditor",
            "tools": [{"name": name, "description": DESCRIPTIONS[name]}
                      for name in TOOLS]}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--list" in argv:
        print(json.dumps(describe(), indent=2))
        return 0
    try:
        build_server().run()
    except ImportError:
        print("the MCP SDK is not installed: pip install -e \".[service]\"",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
