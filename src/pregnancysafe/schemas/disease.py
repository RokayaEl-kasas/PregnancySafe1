"""Disease/condition and document-source models.

These mirror config/config.yaml's `disease_sources` block and
docs/PregnancySafe_Medical_Sources.md's status flags, so the ingestion
pipeline can filter out `restricted` sources programmatically instead of
someone remembering to skip ACOG Nausea/Vomiting (189) by hand.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class SourceStatus(str, Enum):
    """Mirrors the 🟢/🟡/🔴 flags in docs/PregnancySafe_Medical_Sources.md."""

    VERIFIED_OPEN = "verified_open"   # 🟢 confirmed free full text
    MANUAL_CHECK = "manual_check"     # 🟡 bot-detection/login wall, download by hand
    RESTRICTED = "restricted"         # 🔴 membership required, abstract only


class DocumentSource(BaseModel):
    """A single cited guideline/reference (WHO, NICE, ACOG, Cochrane, etc.)."""

    name: str
    url: str
    pdf_url: Optional[str] = None
    doi: Optional[str] = None
    status: SourceStatus
    note: Optional[str] = None
    fallback_url: Optional[str] = None

    @property
    def is_ingestible(self) -> bool:
        """Only verified_open sources should be auto-ingested; manual_check
        sources need a human to download the PDF first, and restricted
        sources should never be scraped (see docs/PregnancySafe_Medical_Sources.md)."""
        return self.status == SourceStatus.VERIFIED_OPEN


class Disease(BaseModel):
    """A pregnancy-related condition tracked by the knowledge base."""

    id: str = Field(..., description="Slug matching the data/raw/<id>/ folder name")
    label_ar: str
    label_en: str
    folder: str
    sources: list[DocumentSource] = Field(default_factory=list)

    def ingestible_sources(self) -> list[DocumentSource]:
        return [s for s in self.sources if s.is_ingestible]


class Chunk(BaseModel):
    """A single retrieval-ready chunk produced by ingestion/chunker.py."""

    chunk_id: str
    text: str
    disease_id: str
    source_name: str
    source_url: str
    trimester_tags: list[int] = Field(
        default_factory=list,
        description="Trimester(s) (1/2/3) this chunk is relevant to, if determinable. Empty = general.",
    )

    model_config = ConfigDict(str_strip_whitespace=True)
