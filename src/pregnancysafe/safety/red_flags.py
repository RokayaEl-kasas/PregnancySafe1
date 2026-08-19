"""Hard-coded red-flag symptom screening.

This runs BEFORE retrieval. If a user's described symptoms match a red flag,
the agent must short-circuit straight to an emergency-referral message and
skip the normal RAG answer entirely — a well-cited but leisurely RAG
response is the wrong output when the input describes a possible
preeclampsia or hemorrhage emergency.

Matching is deliberately simple (keyword co-occurrence) rather than another
LLM call: red-flag detection needs to be fast, auditable, and not dependent
on a model call that could fail or be slow.
"""

from __future__ import annotations

from dataclasses import dataclass

from pregnancysafe.schemas import RedFlag
from pregnancysafe.utils.config_loader import load_red_flags

# Each red flag from data/medication_safety.json is matched against a set of
# keyword groups; ALL groups must have at least one hit for the flag to fire.
# This keeps false positives low (e.g. "headache" alone shouldn't trigger the
# preeclampsia flag — it needs the visual-disturbance/BP co-occurrence too).
_KEYWORD_GROUPS_AR: dict[str, list[list[str]]] = {
    "preeclampsia_warning": [
        ["صداع شديد", "صداع قوي", "صداع مستمر"],
        ["زغللة", "اضطراب رؤية", "رؤية ضبابية", "وميض ضوء"],
    ],
    "pyelonephritis_warning": [
        ["حمى", "سخونية", "ارتفاع حرارة"],
        ["ألم بالخاصرة", "ألم في الظهر", "وجع الكلى", "خاصرة", "الكلى"],
    ],
    "bleeding_warning": [
        ["نزيف", "نزول دم"],
    ],
    "hyperemesis_warning": [
        ["قيء شديد", "ترجيع مستمر", "قيء متكرر"],
        ["عدم القدرة على شرب", "جفاف", "مش قادرة اشرب"],
    ],
}


@dataclass
class RedFlagMatch:
    red_flag: RedFlag
    matched_keywords: list[str]


def screen_for_red_flags(user_text: str) -> list[RedFlagMatch]:
    """Scan free-text symptom description for red-flag keyword co-occurrence.

    Returns a list (usually empty) of RedFlagMatch. The agent should treat
    ANY non-empty result as "stop, do not answer normally, refer to
    emergency care" — see agent/pregnancy_agent.py.
    """
    text = user_text.strip()
    if not text:
        return []

    red_flags_by_id = {rf.id: rf for rf in load_red_flags()}
    matches: list[RedFlagMatch] = []

    for flag_id, keyword_groups in _KEYWORD_GROUPS_AR.items():
        red_flag = red_flags_by_id.get(flag_id)
        if red_flag is None:
            continue  # data file and code are out of sync; skip rather than crash

        matched_keywords: list[str] = []
        all_groups_hit = True
        for group in keyword_groups:
            hit = next((kw for kw in group if kw in text), None)
            if hit is None:
                all_groups_hit = False
                break
            matched_keywords.append(hit)

        if all_groups_hit:
            matches.append(RedFlagMatch(red_flag=red_flag, matched_keywords=matched_keywords))

    return matches
