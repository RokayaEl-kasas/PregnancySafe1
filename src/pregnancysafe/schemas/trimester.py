"""Trimester representation shared across the whole codebase.

Keeping this as a single enum (rather than raw ints/strings scattered through
ingestion, safety, and retrieval code) means a typo like "trimster" or "1st"
fails fast at validation time instead of silently breaking a filter query.
"""

from __future__ import annotations

from enum import IntEnum


class Trimester(IntEnum):
    """Pregnancy trimester. IntEnum so `Trimester.SECOND >= Trimester.FIRST` works
    directly for range checks (e.g. "avoid after week 20" logic)."""

    FIRST = 1
    SECOND = 2
    THIRD = 3

    @classmethod
    def from_week(cls, gestational_week: int) -> "Trimester":
        """Map a gestational week (1-42) to a Trimester.

        Boundaries follow the common ACOG/WHO convention:
        weeks 1-13 -> first, 14-27 -> second, 28+ -> third.
        """
        if gestational_week < 1 or gestational_week > 45:
            raise ValueError(f"gestational_week out of plausible range: {gestational_week}")
        if gestational_week <= 13:
            return cls.FIRST
        if gestational_week <= 27:
            return cls.SECOND
        return cls.THIRD

    def label_ar(self) -> str:
        return {
            Trimester.FIRST: "الثلث الأول",
            Trimester.SECOND: "الثلث الثاني",
            Trimester.THIRD: "الثلث الثالث",
        }[self]
