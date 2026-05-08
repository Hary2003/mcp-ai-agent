# backend/app/agent/conflict.py
# Only responsibility: detect and resolve conflicts between source results.
# Takes raw_results dict, returns (resolved_results, conflict_notes).

import re
from .mcp_servers import SOURCE_AUTHORITY


def resolve_conflicts(raw_results: dict) -> tuple[dict, list[str]]:
    """
    Detect and resolve conflicts between sources.

    Strategy:
      1. Recency   — prefer the most recently updated source
      2. Authority — when recency ties, use SOURCE_AUTHORITY ranking

    Args:
        raw_results: dict of {source_name: result_text}

    Returns:
        (resolved_results, conflict_notes)
        conflict_notes is a list of human-readable strings describing
        what conflicts were found and how they were resolved.
    """
    conflict_notes = []
    date_pattern = re.compile(r"\((\d{4}-\d{2}-\d{2})\)")

    # Extract the most recent date mentioned in each source's result
    source_dates = {}
    for source, result in raw_results.items():
        dates = date_pattern.findall(result)
        if dates:
            source_dates[source] = max(dates)

    # Status keywords that could conflict across sources
    keywords_to_watch = [
        "open", "closed", "in progress", "done",
        "complete", "fixed", "resolved",
    ]

    for kw in keywords_to_watch:
        conflicting = [
            src for src, result in raw_results.items()
            if kw.lower() in result.lower()
        ]

        if len(conflicting) >= 2:
            # Best source = most recent date, then highest authority
            best = max(
                conflicting,
                key=lambda s: (
                    source_dates.get(s, "0000-00-00"),
                    SOURCE_AUTHORITY.get(s, 0),
                ),
            )
            others = [s for s in conflicting if s != best]
            if others:
                conflict_notes.append(
                    f"Status conflict on '{kw}': found in {conflicting}. "
                    f"Prioritising {best} "
                    f"(authority: {SOURCE_AUTHORITY.get(best, 0)}, "
                    f"date: {source_dates.get(best, 'unknown')})."
                )

    return raw_results, conflict_notes