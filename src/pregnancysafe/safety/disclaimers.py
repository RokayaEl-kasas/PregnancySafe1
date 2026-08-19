"""Mandatory disclaimer text.

Every response the agent produces — regardless of confidence, retrieval
quality, or medication tier — must carry this disclaimer. It is appended in
code (not left to prompt-following) so it cannot be dropped by a model
sampling quirk or a prompt-injection attempt embedded in retrieved text.
"""

from __future__ import annotations

MANDATORY_DISCLAIMER_AR = (
    "⚠️ إخلاء مسؤولية: المعلومات دي للتثقيف الصحي فقط ومبنية على مصادر طبية "
    "رسمية (WHO / NICE / ACOG / FDA / MotherToBaby)، لكنها مش بديل عن استشارة "
    "طبيبك المختص. القرار النهائي بخصوص أي دواء أو علاج لازم يكون مع طبيب "
    "متابع لحالتك."
)

MANDATORY_DISCLAIMER_EN = (
    "⚠️ Disclaimer: This information is for health education only and is "
    "grounded in official medical sources (WHO / NICE / ACOG / FDA / "
    "MotherToBaby), but it is not a substitute for advice from your own "
    "physician. Any decision about medication or treatment should be made "
    "with the doctor managing your care."
)


def attach_disclaimer(answer_text: str, lang: str = "ar") -> str:
    """Append the mandatory disclaimer to an answer. `lang` selects AR or EN;
    unrecognized values default to Arabic since that's the project's primary
    audience language."""
    disclaimer = MANDATORY_DISCLAIMER_EN if lang == "en" else MANDATORY_DISCLAIMER_AR
    return f"{answer_text.rstrip()}\n\n{disclaimer}"
