#!/usr/bin/env python3
"""Runs evaluation/eval_cases.json against PregnancyAgent and reports pass/fail
per case, plus a summary by category (red_flag / safe_refusal / medication_tier).

Only exercises the parts of the agent that don't require a built vector
index (red-flag screening, scope checking, medication-tier lookup) — cases
that need real retrieval (category 'normal_query_no_red_flag') will still
run but their 'answer produced' check is best-effort until scripts/build_index.py
has been run against real ingested PDFs.

Usage:
    python evaluation/evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path

from pregnancysafe.agent import PregnancyAgent
from pregnancysafe.utils.config_loader import REPO_ROOT
from pregnancysafe.utils.logging_config import get_logger

logger = get_logger(__name__)


def _check_case(agent: PregnancyAgent, case: dict) -> tuple[bool, str]:
    try:
        response = agent.ask(
            case["query"],
            disease_id=case.get("disease_id"),
            gestational_week=case.get("gestational_week"),
            medication_id=case.get("medication_id"),
        )
    except Exception as exc:  # noqa: BLE001 - an eval case must never crash the run
        return False, f"raised exception: {exc}"

    expected = case["expected"]
    mismatches = []

    if "is_red_flag" in expected and response.is_red_flag != expected["is_red_flag"]:
        mismatches.append(f"is_red_flag: expected {expected['is_red_flag']}, got {response.is_red_flag}")

    if "is_out_of_scope" in expected and response.is_out_of_scope != expected["is_out_of_scope"]:
        mismatches.append(
            f"is_out_of_scope: expected {expected['is_out_of_scope']}, got {response.is_out_of_scope}"
        )

    if "medication_tier" in expected:
        actual_tier = response.medication_tier.value if response.medication_tier else None
        if actual_tier != expected["medication_tier"]:
            mismatches.append(
                f"medication_tier: expected {expected['medication_tier']}, got {actual_tier}"
            )

    # The disclaimer must be present on every single response, no exceptions.
    from pregnancysafe.safety.disclaimers import MANDATORY_DISCLAIMER_AR

    if MANDATORY_DISCLAIMER_AR not in response.answer_text:
        mismatches.append("disclaimer missing from answer_text")

    if mismatches:
        return False, "; ".join(mismatches)
    return True, "ok"


def run(cases_path: Path | None = None) -> dict:
    cases_path = cases_path or (REPO_ROOT / "evaluation" / "eval_cases.json")
    with open(cases_path, encoding="utf-8") as f:
        data = json.load(f)

    agent = PregnancyAgent()  # retriever constructed lazily only if a case needs it

    results = []
    for case in data["cases"]:
        # Cases needing real retrieval are skipped gracefully if no index
        # exists yet, rather than failing the whole eval run.
        try:
            passed, detail = _check_case(agent, case)
        except ImportError as exc:
            passed, detail = False, f"skipped — missing dependency: {exc}"
        results.append({"id": case["id"], "category": case["category"], "passed": passed, "detail": detail})

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])

    print(f"\n{'=' * 60}")
    print(f"PregnancySafe Evaluation — {passed_count}/{total} cases passed")
    print(f"{'=' * 60}")
    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"{status} [{r['category']:>10}] {r['id']}: {r['detail']}")

    by_category: dict[str, list[bool]] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r["passed"])
    print(f"\n{'-' * 60}")
    print("By category:")
    for category, outcomes in by_category.items():
        print(f"  {category}: {sum(outcomes)}/{len(outcomes)}")

    return {"total": total, "passed": passed_count, "results": results}


if __name__ == "__main__":
    run()
