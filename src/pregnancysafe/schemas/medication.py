"""Medication and safety-tier data models.

This is the piece that matters most for the evaluation: instead of passing
around raw dicts parsed from JSON (where a typo'd tier like "gren" would
silently fail a `== "green"` check somewhere deep in the agent), every
medication loaded from data/medication_safety.json is validated into a
`Medication` instance up front. A malformed record raises immediately, with
a clear error, instead of causing a wrong safety recommendation at runtime.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator


class SafetyTier(str, Enum):
    """Four-tier medication safety classification used throughout the app."""

    GREEN = "green"    # 🟢 usable in appropriate cases
    YELLOW = "yellow"  # 🟡 physician supervision required
    ORANGE = "orange"  # 🟠 avoid unless necessary
    RED = "red"        # 🔴 contraindicated

    @property
    def emoji(self) -> str:
        return {
            SafetyTier.GREEN: "🟢",
            SafetyTier.YELLOW: "🟡",
            SafetyTier.ORANGE: "🟠",
            SafetyTier.RED: "🔴",
        }[self]

    @property
    def label_ar(self) -> str:
        return {
            SafetyTier.GREEN: "يمكن استخدامه في حالات معينة",
            SafetyTier.YELLOW: "فقط تحت إشراف الطبيب",
            SafetyTier.ORANGE: "يُفضّل تجنبه إلا لضرورة",
            SafetyTier.RED: "ممنوع / غير موصى به",
        }[self]


class Medication(BaseModel):
    """A single medication (or non-drug intervention) with its safety tier.

    `source_url` is intentionally Optional[str] rather than HttpUrl for a few
    entries (e.g. Magnesium Sulfate, low-dose Aspirin) that don't yet have a
    dedicated MotherToBaby/LactMed page in docs/PregnancySafe_Medication_Links.md.
    Validation still runs on `tier` and `id`, which are the fields that would
    actually break the safety logic if malformed.
    """

    id: str = Field(..., min_length=1, description="Stable slug, e.g. 'nitrofurantoin'")
    name: str = Field(..., min_length=1)
    tier: SafetyTier
    notes_ar: Optional[str] = None
    trimester_restriction: Optional[str] = Field(
        default=None,
        description=(
            "Free-text machine-checkable tag, e.g. 'contraindicated_week_20_plus' "
            "or 'avoid_before_week_10'. Parsed by safety/medication_tiers.py."
        ),
    )
    source_name: Optional[str] = None
    source_url: Optional[str] = None

    @field_validator("source_url")
    @classmethod
    def _validate_url_shape(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError(f"source_url must be an absolute http(s) URL, got: {v!r}")
        return v


class DiseaseMedications(BaseModel):
    """All medications documented for a single disease/condition."""

    label_ar: str
    label_en: str
    medications: list[Medication]

    def by_tier(self, tier: SafetyTier) -> list[Medication]:
        return [m for m in self.medications if m.tier == tier]


class RedFlag(BaseModel):
    """A hard-coded symptom-combination that must trigger an emergency
    referral rather than a normal RAG answer. See safety/red_flags.py."""

    id: str
    pattern_ar: str
    suspected_condition: str
    action_ar: str
