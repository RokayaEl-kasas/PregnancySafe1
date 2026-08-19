"""ChromaDB-backed vector store for PregnancySafe chunks.

Wraps chromadb + sentence-transformers behind a small typed interface so the
rest of the codebase (retrieval, agent, tests) never touches the ChromaDB
client directly. Chunk metadata (disease_id, source_name, source_url,
trimester_tags) is stored alongside embeddings, which is what lets
retrieval/citation_formatter.py attribute every retrieved passage back
to its original guideline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pregnancysafe.schemas import Chunk
from pregnancysafe.utils.config_loader import load_raw_config
from pregnancysafe.utils.logging_config import get_logger

logger = get_logger(__name__)


class PregnancySafeVectorStore:
    """Thin wrapper around a persistent ChromaDB collection."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "chromadb is required for indexing/retrieval. "
                "Install with: pip install chromadb"
            ) from exc

        cfg = load_raw_config()

        persist_directory = (
            persist_directory
            or cfg["vectorstore"]["persist_directory"]
        )
        collection_name = (
            collection_name
            or cfg["vectorstore"]["collection_name"]
        )
        embedding_model = (
            embedding_model
            or cfg["embeddings"]["model_name"]
        )

        self._batch_size = int(
            cfg["embeddings"].get("batch_size", 8)
        )

        Path(persist_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

        self._client = chromadb.PersistentClient(
            path=persist_directory
        )

        self._embedding_fn = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model,
                device=cfg["embeddings"].get("device", "cpu"),
            )
        )

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata={
                "hnsw:space": cfg["vectorstore"].get(
                    "distance_metric",
                    "cosine",
                )
            },
        )

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Upsert chunks into ChromaDB in small batches.

        Uses chunk_id as the ChromaDB document id, so re-running ingestion
        on the same source is idempotent.
        """
        if not chunks:
            return 0

        total = 0

        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start:start + self._batch_size]

            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[
                    {
                        "disease_id": c.disease_id,
                        "source_name": c.source_name,
                        "source_url": c.source_url,
                        "trimester_tags": ",".join(
                            str(t) for t in c.trimester_tags
                        ),
                    }
                    for c in batch
                ],
            )

            total += len(batch)

            logger.info(
                "Upserted batch: %d-%d / %d chunks",
                start + 1,
                start + len(batch),
                len(chunks),
            )

        logger.info(
            "Upserted %d chunks into collection.",
            total,
        )

        return total

    def query(
        self,
        query_text: str,
        *,
        top_k: Optional[int] = None,
        disease_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Query the collection.

        Returns a list of dicts:
        {text, disease_id, source_name, source_url,
         trimester_tags, distance}
        ordered by relevance (lowest distance first).
        """
        cfg = load_raw_config()["retrieval"]
        top_k = top_k or cfg["top_k"]

        where = (
            {"disease_id": disease_id}
            if disease_id
            else None
        )

        result = self._collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where,
        )

        hits: list[dict[str, Any]] = []

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for doc, meta, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            hits.append(
                {
                    "text": doc,
                    "disease_id": meta.get("disease_id"),
                    "source_name": meta.get("source_name"),
                    "source_url": meta.get("source_url"),
                    "trimester_tags": [
                        int(t)
                        for t in (
                            meta.get("trimester_tags") or ""
                        ).split(",")
                        if t
                    ],
                    "distance": distance,
                }
            )

        return hits

    def count(self) -> int:
        """Return the number of chunks currently stored."""
        return self._collection.count()