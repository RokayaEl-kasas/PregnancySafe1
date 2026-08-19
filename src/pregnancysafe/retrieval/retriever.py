"""Retrieval layer sitting between the vector store and the agent.

Applies the score_threshold from config.yaml so a low-relevance ChromaDB hit
(e.g. querying about a disease that isn't well covered by the ingested PDFs)
doesn't get passed to the agent as if it were a confident match — that's
what would let the agent produce a fabricated-sounding answer stitched from
irrelevant text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pregnancysafe.indexing.vector_store import PregnancySafeVectorStore
from pregnancysafe.utils.config_loader import load_raw_config
from pregnancysafe.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    text: str
    disease_id: str
    source_name: str
    source_url: str
    trimester_tags: list[int] = field(default_factory=list)
    relevance_score: float = 0.0  # 1.0 = perfect match, 0.0 = no similarity


class Retriever:
    def __init__(self, vector_store: Optional[PregnancySafeVectorStore] = None) -> None:
        self._store = vector_store or PregnancySafeVectorStore()
        self._cfg = load_raw_config()["retrieval"]

    def retrieve(
        self,
        query_text: str,
        *,
        disease_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[RetrievalResult]:
        top_k = top_k or self._cfg["top_k"]
        threshold = self._cfg["score_threshold"]

        raw_hits = self._store.query(query_text, top_k=top_k, disease_id=disease_id)

        results: list[RetrievalResult] = []
        for hit in raw_hits:
            # ChromaDB cosine distance -> similarity score in [0, 1]
            relevance = max(0.0, 1.0 - hit["distance"])
            if relevance < threshold:
                continue
            results.append(
                RetrievalResult(
                    text=hit["text"],
                    disease_id=hit["disease_id"],
                    source_name=hit["source_name"],
                    source_url=hit["source_url"],
                    trimester_tags=hit["trimester_tags"],
                    relevance_score=round(relevance, 4),
                )
            )

        if not results:
            logger.info(
                "No chunks passed score_threshold=%.2f for query: %r", threshold, query_text
            )
        return results
