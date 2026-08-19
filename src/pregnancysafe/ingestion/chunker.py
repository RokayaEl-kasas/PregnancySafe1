"""Text chunking with disease and trimester tagging.

Splits cleaned document text into overlapping chunks sized per
config/config.yaml (`chunking.chunk_size` / `chunk_overlap`), tags each
chunk with the disease_id it came from, and applies a lightweight keyword
heuristic to tag trimester relevance so the retriever can filter (e.g. "only
show me third-trimester-relevant chunks").
"""

from __future__ import annotations

import hashlib
import re

from pregnancysafe.schemas import Chunk
from pregnancysafe.utils.config_loader import load_raw_config

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?؟۔])\s+")

_TRIMESTER_KEYWORDS = {
    1: [
        "first trimester",
        "week 1",
        "week 2",
        "week 3",
        "الثلث الأول",
        "الأسابيع الأولى",
    ],
    2: [
        "second trimester",
        "الثلث الثاني",
    ],
    3: [
        "third trimester",
        "week 28",
        "week 36",
        "week 37",
        "الثلث الثالث",
        "قرب الولادة",
    ],
}


def _tag_trimesters(chunk_text: str) -> list[int]:
    lower = chunk_text.lower()
    tags = []

    for trimester, keywords in _TRIMESTER_KEYWORDS.items():
        if any(kw.lower() in lower for kw in keywords):
            tags.append(trimester)

    return tags


def _split_into_sentences(text: str) -> list[str]:
    return [
        s.strip()
        for s in _SENTENCE_BOUNDARY_RE.split(text)
        if s.strip()
    ]


def _get_overlap_sentences(
    sentences: list[str],
    overlap_size: int,
) -> list[str]:
    """Return complete sentences from the end of a chunk for overlap.

    The old implementation used the last N characters of a chunk.
    That could cut words in half and produce broken chunk beginnings such as
    's (OTIS)' or 'ported included'.

    This implementation keeps the overlap sentence-aligned instead.
    """

    if overlap_size <= 0 or not sentences:
        return []

    overlap: list[str] = []
    total_length = 0

    for sentence in reversed(sentences):
        sentence_length = len(sentence)

        # Always keep at least one complete sentence if possible.
        if not overlap:
            overlap.insert(0, sentence)
            total_length += sentence_length
            continue

        # Stop before exceeding the configured overlap size.
        if total_length + 1 + sentence_length > overlap_size:
            break

        overlap.insert(0, sentence)
        total_length += 1 + sentence_length

    return overlap


def chunk_text(
    text: str,
    *,
    disease_id: str,
    source_name: str,
    source_url: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Sentence-boundary-aware chunking with sentence-aligned overlap.

    Sentences are accumulated into a chunk until adding the next sentence
    would exceed `chunk_size`.

    When a chunk is closed, the overlap is created from complete sentences
    at the end of the previous chunk rather than from raw characters.
    This prevents chunks from starting in the middle of words.

    Chunk IDs include a deterministic source hash so chunks from different
    source documents cannot collide even when their chunk indices and text
    are identical.
    """

    if chunk_size is None or chunk_overlap is None:
        cfg = load_raw_config()["chunking"]
        chunk_size = (
            chunk_size
            if chunk_size is not None
            else cfg["chunk_size"]
        )
        chunk_overlap = (
            chunk_overlap
            if chunk_overlap is not None
            else cfg["chunk_overlap"]
        )

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    sentences = _split_into_sentences(text)

    if not sentences:
        return []

    chunks: list[Chunk] = []

    # Create a deterministic identifier for the source document.
    # This prevents collisions when multiple documents belong to
    # the same disease and each document starts its chunk index at 0.
    source_hash = hashlib.sha1(
        f"{source_name}|{source_url}".encode("utf-8")
    ).hexdigest()[:10]

    def _flush(buffer_sentences: list[str]) -> None:
        if not buffer_sentences:
            return

        buffer = " ".join(buffer_sentences).strip()

        if not buffer:
            return

        # Chunk index is local to this source document.
        chunk_index = len(chunks)

        # Include the chunk index and text in the content hash.
        # This keeps the ID deterministic while distinguishing
        # repeated/identical text appearing at different positions.
        chunk_hash = hashlib.sha1(
            f"{chunk_index}:{buffer}".encode("utf-8")
        ).hexdigest()[:12]

        chunks.append(
            Chunk(
                chunk_id=(
                    f"{disease_id}_{source_hash}_"
                    f"{chunk_index:04d}_{chunk_hash}"
                ),
                text=buffer,
                disease_id=disease_id,
                source_name=source_name,
                source_url=source_url,
                trimester_tags=_tag_trimesters(buffer),
            )
        )

    current_sentences: list[str] = []

    for sentence in sentences:
        candidate_sentences = current_sentences + [sentence]
        candidate = " ".join(candidate_sentences).strip()

        if (
            len(candidate) > chunk_size
            and current_sentences
        ):
            # Save the current chunk.
            _flush(current_sentences)

            # Create overlap from COMPLETE sentences.
            overlap_sentences = _get_overlap_sentences(
                current_sentences,
                chunk_overlap,
            )

            # Start the next chunk with the clean sentence overlap.
            current_sentences = overlap_sentences + [sentence]

        else:
            current_sentences = candidate_sentences

    # Flush the final chunk.
    _flush(current_sentences)

    return chunks