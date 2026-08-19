"""Tests for the ingestion package: chunking and text cleaning.

No real PDFs are needed here — load_pdf_text itself is thin (delegates to
pypdf), so these tests focus on the logic that's actually ours: sentence-
aware chunking, overlap, trimester tagging, and de-hyphenation cleanup.
"""

from __future__ import annotations

from pregnancysafe.ingestion.chunker import chunk_text
from pregnancysafe.ingestion.loader import _clean_text


class TestChunker:
    def test_short_text_produces_single_chunk(self):
        chunks = chunk_text(
            "Preeclampsia is a serious condition.",
            disease_id="hypertension",
            source_name="WHO",
            source_url="https://who.int/x",
        )
        assert len(chunks) == 1
        assert chunks[0].disease_id == "hypertension"

    def test_long_text_splits_into_multiple_chunks(self):
        sentence = "Blood pressure should be monitored regularly during pregnancy. "
        long_text = sentence * 40  # well over default chunk_size
        chunks = chunk_text(
            long_text,
            disease_id="hypertension",
            source_name="WHO",
            source_url="https://who.int/x",
            chunk_size=200,
            chunk_overlap=30,
        )
        assert len(chunks) > 1
        # No chunk should wildly exceed chunk_size + a sentence's worth of slack
        assert all(len(c.text) < 400 for c in chunks)

    def test_empty_text_produces_no_chunks(self):
        assert chunk_text("", disease_id="uti", source_name="ACOG", source_url="https://x") == []

    def test_third_trimester_keyword_tagged(self):
        chunks = chunk_text(
            "Nitrofurantoin should be avoided near week 36 of pregnancy due to theoretical neonatal risk.",
            disease_id="uti",
            source_name="MotherToBaby",
            source_url="https://ncbi.nlm.nih.gov/x",
        )
        assert 3 in chunks[0].trimester_tags

    def test_chunk_ids_are_unique(self):
        text = "First sentence here. Second sentence here. Third sentence here. " * 10
        chunks = chunk_text(
            text, disease_id="anemia", source_name="ACOG", source_url="https://x",
            chunk_size=100, chunk_overlap=20,
        )
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


class TestTextCleaning:
    def test_dehyphenates_wrapped_words(self):
        raw = "The patient should avoid preg-\nnancy complications."
        cleaned = _clean_text(raw)
        assert "pregnancy" in cleaned
        assert "preg-\nnancy" not in cleaned

    def test_collapses_excess_whitespace(self):
        raw = "Too    many     spaces"
        cleaned = _clean_text(raw)
        assert "  " not in cleaned

    def test_collapses_excess_newlines(self):
        raw = "Paragraph one.\n\n\n\n\nParagraph two."
        cleaned = _clean_text(raw)
        assert "\n\n\n" not in cleaned
