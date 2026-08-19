#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pregnancysafe.retrieval.retriever import Retriever


def main() -> None:
    retriever = Retriever()

    question = (
        "A pregnant woman has chronic hypertension with a blood pressure "
        "of 145/92 mmHg. What antihypertensive treatment should be considered?"
    )

    print("\n" + "=" * 80)
    print("RETRIEVAL TEST")
    print("=" * 80)

    print(f"\nQuestion:\n{question}")

    print("\nDisease filter: hypertension")

    results = retriever.retrieve(
        question,
        disease_id="hypertension",
        top_k=10,
    )

    print(f"\nRetrieved results: {len(results)}")

    if not results:
        print("\nNO RESULTS PASSED THE SCORE THRESHOLD.")
        print("Check retrieval.score_threshold in config.yaml.")
        return

    for i, result in enumerate(results, start=1):
        print("\n" + "-" * 80)
        print(f"RESULT #{i}")
        print("-" * 80)

        print(f"Relevance score : {result.relevance_score}")
        print(f"Disease         : {result.disease_id}")
        print(f"Source          : {result.source_name}")
        print(f"Trimester tags  : {result.trimester_tags}")
        print(f"URL             : {result.source_url}")

        print("\nText:")
        print(result.text[:1200])


if __name__ == "__main__":
    main()