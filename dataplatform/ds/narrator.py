"""Optional LLM narration of an analysis report.

The findings are produced by the detectors, not the model — the model only writes
them up. It is given the measured statistics and told not to add claims of its own,
because a narrator that invents a finding is worse than no narrator.
"""

from __future__ import annotations

from ..errors import LLMUnavailable
from ..nlp.llm import LLMClient, available
from .models import AnalysisReport

INSTRUCTIONS = """\
You write the executive summary of a data analysis that has already been done.

You are given: a dataset profile, a list of statistically detected patterns, and a
ranked list of recommended modelling approaches. Turn them into prose.

Rules:
- Every statement must trace to something in the input. Do not add findings, do not
  speculate about causes, and do not soften a caveat that was given to you.
- Lead with what a decision-maker needs: what the data can support, what it cannot,
  and what has to be fixed first.
- Name the specific columns and numbers. "Revenue is right-skewed (skew 2.3)" beats
  "some columns have unusual distributions".
- If the data has a problem that undermines the modelling advice (leakage,
  duplicates, too few rows), say so before the advice, not after.
- Around 200-300 words, in plain paragraphs. No headers, no bullet lists.
"""


def _render(report: AnalysisReport) -> str:
    profile = report.profile
    lines = [
        f"DATASET: {profile.dataset}",
        f"rows={profile.n_rows:,} columns={profile.n_columns} duplicates={profile.n_duplicate_rows:,}",
        f"target={report.target or '(none)'} inferred_task={report.task}",
        f"numeric={profile.numeric_columns}",
        f"categorical={profile.categorical_columns}",
        f"temporal={profile.temporal_columns}",
        "",
        "DETECTED PATTERNS:",
    ]
    for pattern in report.patterns:
        stat = f" stat={pattern.statistic}" if pattern.statistic is not None else ""
        p = f" p={pattern.p_value:.2g}" if pattern.p_value is not None else ""
        lines.append(f"- [{pattern.severity}] {pattern.kind}{stat}{p}: {pattern.description}")
        if pattern.implication:
            lines.append(f"    implication: {pattern.implication}")

    lines.append("")
    lines.append("RECOMMENDED APPROACHES:")
    for rec in report.recommendations:
        lines.append(f"{rec.rank}. {rec.algorithm} [{rec.task}, confidence={rec.confidence}]")
        for reason in rec.rationale:
            lines.append(f"    because: {reason}")
        for caveat in rec.caveats:
            lines.append(f"    caveat: {caveat}")
    return "\n".join(lines)


def narrate(report: AnalysisReport, client: LLMClient | None = None) -> str:
    if not available():
        raise LLMUnavailable("no model credentials; the report itself is unaffected")
    client = client or LLMClient()
    return client.text(instructions=INSTRUCTIONS, prompt=_render(report), effort="medium")


def try_narrate(report: AnalysisReport, client: LLMClient | None = None) -> str:
    """Narrate if possible; return an empty string otherwise. Never raises."""
    try:
        return narrate(report, client)
    except Exception:
        return ""
