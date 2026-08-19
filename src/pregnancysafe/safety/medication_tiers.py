"""Medication safety-tier lookups and trimester-restriction checks.

This is the module the agent calls before ever surfacing a medication
recommendation. It never guesses: if a medication isn't in
data/medication_safety.json, get_medication_by_id returns None and the
caller (agent/pregnancy_agent.py) must fall back to a safe refusal instead
of inventing a tier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pregnancysafe.schemas import Medication, SafetyTier, Trimester
from pregnancysafe.utils.config_loader import load_medication_safety


def get_medications_for_disease(disease_id: str) -> list[Medication]:
    """Return all documented medications for a disease_id (e.g. 'hypertension').
    Returns an empty list — never raises — if the disease isn't in the dataset,
    since an unknown disease should be handled as an out-of-scope refusal
    upstream, not a crash.
    """
    catalog = load_medication_safety()
    block = catalog.get(disease_id)
    return block.medications if block else []


def get_medication_by_id(disease_id: str, medication_id: str) -> Optional[Medication]:
    for med in get_medications_for_disease(disease_id):
        if med.id == medication_id:
            return med
    return None


@dataclass
class TrimesterCheckResult:
    is_restricted: bool
    reason_ar: Optional[str] = None


# Machine-checkable trimester_restriction tags -> (blocked trimesters, human reason)
# Keep this in sync with the trimester_restriction values used in
# data/medication_safety.json.
_RESTRICTION_RULES: dict[str, tuple[set[Trimester], str]] = {
    "avoid_before_week_10": (
        {Trimester.FIRST},
        "يُفضّل تجنبه في الثلث الأول (قبل الأسبوع 10) لخطر تشوهات خلقية.",
    ),
    "contraindicated_week_20_plus": (
        {Trimester.SECOND, Trimester.THIRD},
        "ممنوع من الأسبوع 20 فصاعدًا لخطر قلة السائل الأمنيوسي.",
    ),
    "caution_after_week_36": (
        {Trimester.THIRD},
        "يُفضّل تجنبه قرب الأسبوع 36-37 لخطر نظري على الجنين.",
    ),
    "avoid_first_trimester_and_near_term": (
        {Trimester.FIRST, Trimester.THIRD},
        "تجنّب في الثلث الأول والقرب من الولادة.",
    ),
    "preferred_first_trimester": (
        set(),  # not a restriction — it's the *preferred* window; no block
        "مفضّل في الثلث الأول.",
    ),
    "switch_from_ptu_after_first_trimester": (
        {Trimester.FIRST},
        "يُستخدم بعد الثلث الأول (PTU مفضّل قبل ذلك).",
    ),
}


def check_trimester_restriction(
    medication: Medication, trimester: Trimester
) -> TrimesterCheckResult:
    """Check whether a medication's documented trimester_restriction tag
    blocks it for the given trimester. Medications with no tag are never
    restricted by this check (tier alone still governs green/yellow/orange/red).
    """
    if not medication.trimester_restriction:
        return TrimesterCheckResult(is_restricted=False)

    rule = _RESTRICTION_RULES.get(medication.trimester_restriction)
    if rule is None:
        # Unknown tag: fail safe by flagging it for human review rather than
        # silently treating it as unrestricted.
        return TrimesterCheckResult(
            is_restricted=True,
            reason_ar=f"علامة قيود غير معروفة ({medication.trimester_restriction}) — راجعي طبيبك.",
        )

    blocked_trimesters, reason_ar = rule
    if trimester in blocked_trimesters:
        return TrimesterCheckResult(is_restricted=True, reason_ar=reason_ar)
    return TrimesterCheckResult(is_restricted=False)


def effective_tier(medication: Medication, trimester: Trimester) -> SafetyTier:
    """Tier as it should be *displayed* for this trimester: escalates a
    green/yellow medication to red if a trimester restriction blocks it
    outright (e.g. NSAIDs are 'red' already, but this also protects
    medications whose base tier is green/yellow but have a hard trimester
    block, like the methylprednisolone before-week-10 case)."""
    check = check_trimester_restriction(medication, trimester)
    if check.is_restricted and medication.tier in (SafetyTier.GREEN, SafetyTier.YELLOW):
        return SafetyTier.RED
    return medication.tier
