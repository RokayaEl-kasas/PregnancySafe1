#!/usr/bin/env python3
"""CLI: load PDFs+HTML (recursively) -> clean -> chunk -> save to data/processed/.

Usage:
    python scripts/run_ingestion.py
    python scripts/run_ingestion.py --disease uti
    python scripts/run_ingestion.py --manifest data/raw/download_sources.log

By default this looks for data/raw/download_sources.log (drop a batch
downloader's log there — see ingestion/manifest.py) to attribute each
ingested file to its real source URL. Without a manifest, files fall back to
the disease's first verified_open source in config.yaml.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from pregnancysafe.ingestion.chunker import chunk_text
from pregnancysafe.ingestion.loader import load_disease_documents
from pregnancysafe.ingestion.manifest import load_download_manifest
from pregnancysafe.utils.config_loader import REPO_ROOT, load_diseases
from pregnancysafe.utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "raw" / "download_sources.log"


def run(disease_filter: Optional[str] = None, manifest_path: Optional[Path] = None) -> None:
    diseases = load_diseases()
    raw_data_root = REPO_ROOT / "data" / "raw"
    processed_dir = REPO_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_path or DEFAULT_MANIFEST_PATH
    manifest = load_download_manifest(manifest_path)
    if manifest:
        logger.info("Loaded %d citation mappings from %s", len(manifest), manifest_path)
    else:
        logger.info(
            "No download manifest found at %s — citations will fall back to each "
            "disease's default source in config.yaml.",
            manifest_path,
        )

    # Ingest every folder that actually exists on disk under data/raw/, not
    # just the ones pre-listed in config.yaml — a folder with no
    # disease_sources entry (e.g. a newly added disease) still gets
    # ingested, it just falls back to manifest-only citations.
    disk_disease_ids = {p.name for p in raw_data_root.iterdir() if p.is_dir()} if raw_data_root.exists() else set()
    target_ids = {disease_filter} if disease_filter else (set(diseases.keys()) | disk_disease_ids)

    total_chunks = 0
    for disease_id in sorted(target_ids):
        disease = diseases.get(disease_id)
        if disease is None:
            # Folder exists on disk but isn't declared in config.yaml yet —
            # ingest it anyway using folder name as the id; citations will
            # rely entirely on the manifest for this one.
            from pregnancysafe.schemas import Disease

            disease = Disease(id=disease_id, label_ar=disease_id, label_en=disease_id, folder=f"data/raw/{disease_id}")
            logger.warning(
                "%s has no config.yaml disease_sources entry — ingesting anyway using the manifest only.",
                disease_id,
            )

        documents = load_disease_documents(disease, raw_data_root, manifest=manifest, repo_root=REPO_ROOT)
        if not documents:
            continue

        disease_chunks = []
        for doc in documents:
            chunks = chunk_text(
                doc.text,
                disease_id=disease_id,
                source_name=doc.source_name,
                source_url=doc.source_url,
            )
            disease_chunks.extend(chunks)
            logger.info("  %s -> %d chunks (%s)", doc.path.name, len(chunks), doc.source_name)

        out_path = processed_dir / f"{disease_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in disease_chunks], f, ensure_ascii=False, indent=2)

        logger.info("Wrote %d chunks for %s -> %s", len(disease_chunks), disease_id, out_path)
        total_chunks += len(disease_chunks)

    logger.info("Ingestion complete. Total chunks: %d", total_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PregnancySafe document ingestion.")
    parser.add_argument("--disease", default=None, help="Limit to a single disease_id (e.g. 'uti').")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to a download log for citation mapping (default: data/raw/download_sources.log).",
    )
    args = parser.parse_args()
    run(
        disease_filter=args.disease,
        manifest_path=Path(args.manifest) if args.manifest else None,
    )


if __name__ == "__main__":
    main()
