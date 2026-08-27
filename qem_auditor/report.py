"""Presenting a verdict so it cannot be skimmed into meaning more than it does.

Two renderers over the same content. The design constraint in both is the
same and is not decoration: **what has been shown and what has not get
equal visual weight**. A report that renders supporting evidence in
confident green and trails off into a grey limitations paragraph is how a
0.115 kcal/mol number becomes a headline, and this project exists because
that happened.

So: no green ticks on a claim that is not certified, no progress bar that
implies a claim is most of the way to true, and the licence line -- what
the verdict actually permits someone to do -- sits directly under the
verdict rather than in a footnote.

Stdlib only. The HTML is a single self-contained file with no external
requests, so it can be opened from a filesystem or attached to an email.
"""
from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Optional

from .claim import compile_claim
from .schema import Experiment
from .verdict import AuditReport, Verdict, audit

# How each verdict should be read. Neutral, factual words -- nothing that
# congratulates a claim for surviving.
VERDICT_TONE = {
    Verdict.INVALID_RECORD: ("unreadable", "#b91c1c"),
    Verdict.INVALID: ("disqualified", "#b91c1c"),
    Verdict.REFUTED: ("contradicted", "#b91c1c"),
    Verdict.CONFLICT: ("inconsistent", "#c2410c"),
    Verdict.NOT_ESTABLISHED: ("untested", "#a16207"),
    Verdict.MODEL_CONDITIONAL: ("conditional", "#a16207"),
    Verdict.PROMISING: ("incomplete", "#1d4ed8"),
    Verdict.CERTIFIED_UNDER_SCOPE: ("established in scope", "#15803d"),
}

_BOX = {"tl": "┌", "tr": "┐", "bl": "└", "br": "┘", "h": "─", "v": "│"}


def _rule(width: int = 72) -> str:
    return _BOX["h"] * width


def render_console(exp: Experiment, investigation=None,
                   report: Optional[AuditReport] = None) -> str:
    """A terminal report. Support and limits at equal weight, always."""
    report = report or audit(exp)
    claim = compile_claim(exp, report)
    tone, _ = VERDICT_TONE[report.verdict]

    out: list[str] = []
    out.append(_BOX["tl"] + _rule(70) + _BOX["tr"])
    title = f" {exp.experiment_id} "
    out.append(_BOX["v"] + title.ljust(70) + _BOX["v"])
    out.append(_BOX["bl"] + _rule(70) + _BOX["br"])
    out.append("")
    out.append(f"CLAIM     {exp.claim or '(none stated)'}")
    out.append(f"VERDICT   {report.verdict.value}  [{tone}]")
    out.append(f"LICENCE   {claim.licence}")

    if investigation is not None:
        out.append("")
        out.append(f"INVESTIGATION  {len(investigation.rounds)} round(s), "
                   f"{investigation.total_falsified} attack(s) falsified the claim")
        out.append(f"STOPPED        {investigation.stopped_because}")

    shown = [g for g in report.gate_results if g.passed is True]
    not_shown = [g for g in report.gate_results if g.passed is False]
    untested = [g for g in report.gate_results if g.passed is None]

    # Deliberately in this order and with matching headers: the reader
    # should not be able to absorb the first list without the second.
    out.append("")
    out.append(f"ESTABLISHED ({len(shown)})")
    out += [f"  + {g.name}: {g.reason}" for g in shown] or ["  (nothing)"]

    out.append("")
    out.append(f"NOT ESTABLISHED ({len(not_shown) + len(untested)})")
    out += [f"  - {g.name}: {g.reason}" for g in not_shown]
    out += [f"  ? {g.name}: {g.reason}" for g in untested]
    if not not_shown and not untested:
        out.append("  (nothing outstanding)")

    if report.integrity_violations:
        out.append("")
        out.append("RECORD INTEGRITY")
        out += [f"  ! {v}" for v in report.integrity_violations]

    analysis = claim.failure_analysis
    if analysis and analysis.diagnoses:
        out.append("")
        out.append("WHY")
        for d in analysis.diagnoses:
            out.append(f"  {d.mode.name} ({d.confidence:.2f})")
            out.append(f"    {d.evidence}")
            if d.remedy:
                out.append(f"    remedy: {d.remedy}")

    if claim.next_experiment:
        cost = (f"${claim.next_experiment.cost_usd:,.2f}"
                if claim.next_experiment.cost_usd else "no cost")
        out.append("")
        out.append("NEXT EXPERIMENT")
        out.append(f"  {claim.next_experiment.description}  ({cost})")

    if claim.pass_criterion:
        out.append("")
        out.append(f"PASS CRITERION  {claim.pass_criterion}")
    return "\n".join(out)


_CSS = """
:root{--bg:#ffffff;--fg:#111827;--muted:#6b7280;--line:#e5e7eb;--card:#f9fafb;
--shown:#15803d;--missing:#b45309;--bad:#b91c1c}
@media (prefers-color-scheme:dark){:root{--bg:#0b0f16;--fg:#e5e7eb;--muted:#9ca3af;
--line:#1f2937;--card:#111827;--shown:#4ade80;--missing:#fbbf24;--bad:#f87171}}
*{box-sizing:border-box}
body{margin:0;padding:2rem 1rem;background:var(--bg);color:var(--fg);
font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:52rem;margin:0 auto}
h1{font-size:1.35rem;margin:0 0 .25rem;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:.9rem;margin:0 0 1.5rem}
.verdict{border:1px solid var(--line);border-left:4px solid var(--accent);
border-radius:.5rem;padding:1rem 1.25rem;background:var(--card);margin-bottom:1.5rem}
.verdict .v{font-size:1.1rem;font-weight:650;color:var(--accent)}
.verdict .tone{color:var(--muted);font-weight:400;font-size:.85rem;margin-left:.5rem}
.verdict .licence{margin-top:.5rem;font-size:.92rem}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:1.25rem}
@media (max-width:40rem){.cols{grid-template-columns:1fr}}
section{border:1px solid var(--line);border-radius:.5rem;padding:1rem 1.15rem;
background:var(--card);margin-bottom:1.25rem}
h2{font-size:.78rem;text-transform:uppercase;letter-spacing:.07em;
color:var(--muted);margin:0 0 .75rem;font-weight:600}
ul{margin:0;padding-left:1.1rem}
li{margin:.35rem 0}
li code{font-size:.85em;color:var(--muted)}
.shown li::marker{color:var(--shown)}
.missing li::marker{color:var(--missing)}
.bad li::marker{color:var(--bad)}
.diag{border-top:1px solid var(--line);padding-top:.75rem;margin-top:.75rem}
.diag:first-of-type{border-top:0;padding-top:0;margin-top:0}
.mode{font-weight:650}
.conf{color:var(--muted);font-weight:400;font-size:.85em}
.remedy{color:var(--muted);font-size:.9em;margin-top:.25rem}
.rounds{width:100%;border-collapse:collapse;font-size:.9rem}
.rounds th{text-align:left;color:var(--muted);font-weight:600;font-size:.78rem;
text-transform:uppercase;letter-spacing:.05em;padding:.35rem .5rem .35rem 0}
.rounds td{padding:.35rem .5rem .35rem 0;border-top:1px solid var(--line)}
footer{color:var(--muted);font-size:.82rem;margin-top:2rem;border-top:1px solid var(--line);
padding-top:1rem}
"""


def _li(items, cls):
    if not items:
        return '<ul><li style="list-style:none;color:var(--muted)">(none)</li></ul>'
    rows = "".join(f"<li><code>{html.escape(n)}</code> — {html.escape(r)}</li>"
                   for n, r in items)
    return f'<ul class="{cls}">{rows}</ul>'


def render_html(exp: Experiment, investigation=None,
                report: Optional[AuditReport] = None) -> str:
    """A single self-contained HTML page. No external requests."""
    report = report or audit(exp)
    claim = compile_claim(exp, report)
    tone, colour = VERDICT_TONE[report.verdict]

    shown = [(g.name, g.reason) for g in report.gate_results if g.passed is True]
    failed = [(g.name, g.reason) for g in report.gate_results if g.passed is False]
    untested = [(g.name, g.reason) for g in report.gate_results if g.passed is None]

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>Audit — {html.escape(exp.experiment_id)}</title>",
        f"<style>{_CSS}</style></head><body><main>",
        f"<h1>{html.escape(exp.experiment_id)}</h1>",
        f"<p class='sub'>{html.escape(exp.claim or '(no claim stated)')}</p>",
        f"<div class='verdict' style='--accent:{colour}'>",
        f"<div class='v'>{html.escape(report.verdict.value)}"
        f"<span class='tone'>{html.escape(tone)}</span></div>",
        f"<div class='licence'>{html.escape(claim.licence)}</div></div>",
    ]

    if investigation is not None:
        rows = "".join(
            f"<tr><td>{r.number}</td><td>{html.escape(r.verdict.value)}</td>"
            f"<td>{len(r.attack_plan.attacks) if r.attack_plan else 0}</td>"
            f"<td>{len(r.falsified)}</td></tr>"
            for r in investigation.rounds)
        parts.append(
            "<section><h2>Investigation</h2><table class='rounds'>"
            "<tr><th>Round</th><th>Verdict</th><th>Attacks</th><th>Falsified</th></tr>"
            f"{rows}</table>"
            f"<p class='remedy'>Stopped: {html.escape(investigation.stopped_because)}</p>"
            "</section>")

    # Side by side, same size, same styling weight. The layout is the
    # argument: neither column is the summary.
    parts.append(
        "<div class='cols'>"
        f"<section><h2>Established ({len(shown)})</h2>{_li(shown, 'shown')}</section>"
        f"<section><h2>Not established ({len(failed) + len(untested)})</h2>"
        f"{_li(failed + untested, 'missing')}</section>"
        "</div>")

    if report.integrity_violations:
        rows = "".join(f"<li>{html.escape(v)}</li>"
                       for v in report.integrity_violations)
        parts.append(f"<section><h2>Record integrity</h2>"
                     f"<ul class='bad'>{rows}</ul></section>")

    analysis = claim.failure_analysis
    if analysis and analysis.diagnoses:
        blocks = []
        for d in analysis.diagnoses:
            blocks.append(
                f"<div class='diag'><div class='mode'>{html.escape(d.mode.name)}"
                f"<span class='conf'> · confidence {d.confidence:.2f}</span></div>"
                f"<div>{html.escape(d.evidence)}</div>"
                + (f"<div class='remedy'>Remedy: {html.escape(d.remedy)}</div>"
                   if d.remedy else "") + "</div>")
        parts.append(f"<section><h2>Why</h2>{''.join(blocks)}</section>")

    if claim.next_experiment:
        cost = (f"${claim.next_experiment.cost_usd:,.2f}"
                if claim.next_experiment.cost_usd else "no cost")
        parts.append(
            f"<section><h2>Next experiment</h2>"
            f"<div>{html.escape(claim.next_experiment.description)}</div>"
            f"<div class='remedy'>Cost: {cost}</div></section>")

    parts.append(
        f"<footer>Pass criterion: {html.escape(claim.pass_criterion)}<br>"
        f"Verdicts are computed by inspectable Python gates, never asserted by a "
        f"model. An untested control is not a passed one.</footer>")
    parts.append("</main></body></html>")
    return "".join(parts)
