"""Mandatory medical disclaimer.

Every response produced by the agent must include this disclaimer,
regardless of confidence, retrieval quality, or medication safety tier.

The disclaimer is appended programmatically rather than relying on
LLM prompt instructions. This prevents it from being omitted because
of model behavior or prompt injection in retrieved content.
"""

from __future__ import annotations


MANDATORY_DISCLAIMER = (
    "Disclaimer: This information is for health education only and is "
    "grounded in official medical sources (WHO / NICE / ACOG / FDA / "
    "MotherToBaby). It is not a substitute for advice from a qualified "
    "healthcare professional. Any decision about medication or treatment "
    "should be made with the healthcare professional managing the patient's care."
)


def attach_disclaimer(answer_text: str) -> str:
    """Append the mandatory medical disclaimer to an answer."""
    return f"{answer_text.rstrip()}\n\n{MANDATORY_DISCLAIMER}"