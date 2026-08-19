"""Tests for retrieval + citation formatting.

Uses a fake vector store (matching PregnancySafeVectorStore.query's return
shape) so these tests don't require chromadb/sentence-transformers to be
installed or a real index to be built — the score-threshold filtering logic
is what's actually being tested here, not ChromaDB itself.
"""

from __future__ import annotations

from typing import Any, Optional

from pregnancysafe.retrieval.citation_formatter import format_citations
from pregnancysafe.retrieval.retriever import Retriever


class FakeVectorStore:
    """Drop-in replacement for PregnancySafeVectorStore in tests."""

    def __init__(self, canned_hits: list[dict[str, Any]]):
        self._canned_hits = canned_hits

    def query(
        self, query_text: str, *, top_k: Optional[int] = None, disease_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        hits = self._canned_hits
        if disease_id:
            hits = [h for h in hits if h["disease_id"] == disease_id]
        return hits[: top_k or len(hits)]


class TestRetrieverThreshold:
    def test_low_relevance_hits_filtered_out(self):
        # score_threshold in config.yaml is 0.35 -> relevance = 1 - distance
        fake_store = FakeVectorStore(
            [
                {
                    "text": "Highly relevant passage.",
                    "disease_id": "hypertension",
                    "source_name": "WHO",
                    "source_url": "https://who.int/x",
                    "trimester_tags": [],
                    "distance": 0.1,  # relevance 0.9 -> keep
                },
                {
                    "text": "Barely related passage.",
                    "disease_id": "hypertension",
                    "source_name": "WHO",
                    "source_url": "https://who.int/x",
                    "trimester_tags": [],
                    "distance": 0.9,  # relevance 0.1 -> drop
                },
            ]
        )
        retriever = Retriever(vector_store=fake_store)
        results = retriever.retrieve("blood pressure treatment", disease_id="hypertension")
        assert len(results) == 1
        assert results[0].text == "Highly relevant passage."

    def test_no_hits_returns_empty_list(self):
        retriever = Retriever(vector_store=FakeVectorStore([]))
        assert retriever.retrieve("anything") == []

    def test_disease_filter_applied(self):
        fake_store = FakeVectorStore(
            [
                {
                    "text": "UTI passage.",
                    "disease_id": "uti",
                    "source_name": "ACOG",
                    "source_url": "https://acog.org/x",
                    "trimester_tags": [],
                    "distance": 0.05,
                },
                {
                    "text": "Hypertension passage.",
                    "disease_id": "hypertension",
                    "source_name": "WHO",
                    "source_url": "https://who.int/x",
                    "trimester_tags": [],
                    "distance": 0.05,
                },
            ]
        )
        retriever = Retriever(vector_store=fake_store)
        results = retriever.retrieve("infection symptoms", disease_id="uti")
        assert len(results) == 1
        assert results[0].disease_id == "uti"


class TestCitationFormatter:
    def test_empty_hits_returns_placeholder_message(self):
        assert "لا توجد مصادر" in format_citations([])

    def test_duplicate_sources_deduplicated(self):
        from pregnancysafe.retrieval.retriever import RetrievalResult

        hits = [
            RetrievalResult(
                text="chunk 1", disease_id="uti", source_name="ACOG",
                source_url="https://acog.org/x", relevance_score=0.9,
            ),
            RetrievalResult(
                text="chunk 2", disease_id="uti", source_name="ACOG",
                source_url="https://acog.org/x", relevance_score=0.8,
            ),
        ]
        citations = format_citations(hits)
        # Same source cited twice in the underlying chunks -> one citation line
        assert citations.count("ACOG") == 1
