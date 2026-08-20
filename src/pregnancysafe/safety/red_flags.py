"""Hard-coded red-flag symptom screening.

This module runs before retrieval. If a user's symptoms match a red flag,
the agent immediately returns an emergency-referral response and skips
the normal RAG pipeline.

Matching is deliberately implemented as keyword co-occurrence rather
than an LLM call because red-flag detection must be fast, auditable,
deterministic, and independent of model availability.
"""

from __future__ import annotations

from dataclasses import dataclass

from pregnancysafe.schemas import RedFlag
from pregnancysafe.utils.config_loader import load_red_flags


# Each red flag is matched against multiple keyword groups.
#
# At least one keyword from every group must appear for the red flag
# to trigger. This reduces false positives.
#
# Example:
# "headache" alone should NOT trigger preeclampsia.
# The query must contain both a severe-headache keyword and a
# vision-disturbance keyword.
_KEYWORD_GROUPS: dict[str, list[list[str]]] = {
    "preeclampsia_warning": [
        [
            "severe headache",
            "strong headache",
            "persistent headache",
        ],
        [
            "blurred vision",
            "visual disturbance",
            "vision problems",
            "flashing lights",
        ],
    ],
    "pyelonephritis_warning": [
        [
            "fever",
            "high temperature",
        ],
        [
            "flank pain",
            "back pain",
            "kidney pain",
            "side pain",
        ],
    ],
    "bleeding_warning": [
        [
            "bleeding",
            "vaginal bleeding",
        ],
        [
            "blood loss",
            "heavy bleeding",
        ],
    ],
    "hyperemesis_warning": [
        [
            "severe vomiting",
            "persistent vomiting",
            "repeated vomiting",
        ],
        [
            "unable to drink",
            "dehydration",
            "cannot keep fluids down",
        ],
    ],
}


@dataclass
class RedFlagMatch:
    """Represents a detected red-flag condition."""

    red_flag: RedFlag
    matched_keywords: list[str]


def screen_for_red_flags(user_text: str) -> list[RedFlagMatch]:
    """Scan user text for red-flag keyword co-occurrence.

    The function returns a list of RedFlagMatch objects.

    If the returned list is not empty, the agent should stop normal
    processing and refer the user to appropriate emergency medical care.

    Args:
        user_text: Free-text description of the user's symptoms.

    Returns:
        A list of detected red flags. Returns an empty list when no
        red flag is detected or when the input is empty.
    """

    text = user_text.strip().lower()

    if not text:
        return []

    red_flags_by_id = {
        red_flag.id: red_flag
        for red_flag in load_red_flags()
    }

    matches: list[RedFlagMatch] = []

    for flag_id, keyword_groups in _KEYWORD_GROUPS.items():
        red_flag = red_flags_by_id.get(flag_id)

        # Keep the detector robust if the configuration and code
        # become temporarily out of sync.
        if red_flag is None:
            continue

        matched_keywords: list[str] = []
        all_groups_hit = True

        for group in keyword_groups:
            hit = next(
                (
                    keyword
                    for keyword in group
                    if keyword in text
                ),
                None,
            )

            if hit is None:
                all_groups_hit = False
                break

            matched_keywords.append(hit)

        if all_groups_hit:
            matches.append(
                RedFlagMatch(
                    red_flag=red_flag,
                    matched_keywords=matched_keywords,
                )
            )

    return matches