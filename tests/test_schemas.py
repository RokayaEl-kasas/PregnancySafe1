"""Tests for pregnancysafe.schemas.

The whole point of the schemas layer is to fail loudly on bad data instead
of letting a typo'd tier or malformed URL silently reach the agent. These
tests exist to prove that guarantee actually holds.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pregnancysafe.schemas import (
    Chunk,
    Disease,
    DocumentSource,
    Medication,
    SafetyTier,
    SourceStatus,
    Trimester,
)


class TestSafetyTier:
    def test_emoji_and_label_defined_for_every_tier(self):
        for tier in SafetyTier:
            assert tier.emoji
            assert tier.label_ar

    def test_invalid_tier_string_rejected(self):
        with pytest.raises(ValidationError):
            Medication(id="x", name="Test Drug", tier="not_a_real_tier")


class TestMedication:
    def test_valid_medication_parses(self):
        med = Medication(
            id="labetalol",
            name="Labetalol",
            tier="green",
            source_url="https://www.ncbi.nlm.nih.gov/books/NBK582779/",
        )
        assert med.tier == SafetyTier.GREEN

    def test_missing_required_field_rejected(self):
        with pytest.raises(ValidationError):
            Medication(id="", name="Test", tier="green")  # empty id

    def test_malformed_source_url_rejected(self):
        with pytest.raises(ValidationError):
            Medication(id="x", name="Test", tier="green", source_url="not-a-url")

    def test_none_source_url_allowed(self):
        med = Medication(id="magnesium_sulfate", name="Magnesium Sulfate", tier="green")
        assert med.source_url is None


class TestTrimester:
    @pytest.mark.parametrize(
        "week,expected",
        [(1, Trimester.FIRST), (13, Trimester.FIRST), (14, Trimester.SECOND),
         (27, Trimester.SECOND), (28, Trimester.THIRD), (40, Trimester.THIRD)],
    )
    def test_from_week_boundaries(self, week, expected):
        assert Trimester.from_week(week) == expected

    def test_out_of_range_week_rejected(self):
        with pytest.raises(ValueError):
            Trimester.from_week(0)
        with pytest.raises(ValueError):
            Trimester.from_week(100)

    def test_ordering_supports_range_checks(self):
        assert Trimester.THIRD > Trimester.FIRST


class TestDiseaseAndSources:
    def test_ingestible_sources_filters_by_status(self):
        disease = Disease(
            id="hypertension",
            label_ar="ارتفاع الضغط",
            label_en="Hypertension",
            folder="data/raw/hypertension",
            sources=[
                DocumentSource(name="WHO", url="https://who.int/x", status=SourceStatus.VERIFIED_OPEN),
                DocumentSource(name="ACOG", url="https://acog.org/x", status=SourceStatus.RESTRICTED),
                DocumentSource(name="NICE", url="https://nice.org.uk/x", status=SourceStatus.MANUAL_CHECK),
            ],
        )
        ingestible = disease.ingestible_sources()
        assert len(ingestible) == 1
        assert ingestible[0].name == "WHO"


class TestChunk:
    def test_valid_chunk_parses(self):
        chunk = Chunk(
            chunk_id="hypertension_abc123",
            text="  Some guideline text.  ",
            disease_id="hypertension",
            source_name="WHO",
            source_url="https://who.int/x",
            trimester_tags=[2, 3],
        )
        # str_strip_whitespace config should trim the text
        assert chunk.text == "Some guideline text."
