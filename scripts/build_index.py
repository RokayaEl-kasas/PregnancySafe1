#!/usr/bin/env python3
"""CLI: read data/processed/*.json chunks -> embed -> upsert into ChromaDB.

Usage:
    python scripts/build_index.py                 # index everything in data/processed/
    python scripts/build_index.py --disease uti    # index a single disease's chunks

Run scripts/run_ingestion.py first — this script only reads already-chunked
JSON, it doesn't touch PDFs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pregnancysafe.indexing.vector_store import PregnancySafeVectorStore
from pregnancysafe.schemas import Chunk
from pregnancysafe.utils.config_loader import REPO_ROOT
from pregnancysafe.utils.logging_config import get_logger

logger = get_logger(__name__)


def run(disease_filter: str | None = None) -> None:
    processed_dir = REPO_ROOT / "data" / "processed"
    files = (
        [processed_dir / f"{disease_filter}.json"]
        if disease_filter
        else sorted(processed_dir.glob("*.json"))
    )

    store = PregnancySafeVectorStore()
    total = 0

    for path in files:
        if not path.exists():
            logger.warning("No processed chunks found at %s — run run_ingestion.py first.", path)
            continue

        with open(path, encoding="utf-8") as f:
            raw_chunks = json.load(f)

        chunks = [Chunk(**c) for c in raw_chunks]
        added = store.add_chunks(chunks)
        total += added
        logger.info("Indexed %d chunks from %s", added, path.name)

    logger.info("Indexing complete. Total chunks in collection: %d", store.count())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/update the PregnancySafe vector index.")
    parser.add_argument("--disease", default=None, help="Limit to a single disease_id (e.g. 'uti').")
    args = parser.parse_args()
    run(disease_filter=args.disease)


if __name__ == "__main__":
    main()
