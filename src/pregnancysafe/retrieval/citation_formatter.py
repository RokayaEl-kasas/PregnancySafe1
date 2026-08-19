"""Formats retrieved chunks into user-facing citations.

Kept separate from retriever.py on purpose: retrieval decides what counts
as relevant, while this module decides how retrieved sources are displayed.

Every citation points back to the original medical guideline source.
"""

from __future__ import annotations

from pregnancysafe.retrieval.retriever import RetrievalResult


def format_hit_as_citation(
    hit: RetrievalResult,
    index: int,
) -> str:
    """Format a single retrieval hit as an English citation line."""

    return (
        f"[{index}] Source: "
        f"{hit.source_name} — "
        f"{hit.source_url}"
    )


def format_citations(
    hits: list[RetrievalResult],
) -> str:
    """Format citations while removing duplicate sources."""

    if not hits:
        return "No sources were retrieved for this question."

    seen_sources: set[str] = set()
    lines: list[str] = []

    index = 1

    for hit in hits:

        key = (
            f"{hit.source_name}|"
            f"{hit.source_url}"
        )

        if key in seen_sources:
            continue

        seen_sources.add(key)

        lines.append(
            format_hit_as_citation(
                hit,
                index,
            )
        )

        index += 1

    return "\n".join(lines)