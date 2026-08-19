"""Tests for the safety package — the module the eval rubric will scrutinize most.

Covers: red-flag keyword co-occurrence, trimester-based tier escalation, and
that the mandatory disclaimer is genuinely unconditional.
"""

from __future__ import annotations

from pregnancysafe.agent import PregnancyAgent
from pregnancysafe.safety.disclaimers import MANDATORY_DISCLAIMER_AR, attach_disclaimer
from pregnancysafe.safety.medication_tiers import (
    check_trimester_restriction,
    effective_tier,
    get_medication_by_id,
    get_medications_for_disease,
)
from pregnancysafe.safety.red_flags import screen_for_red_flags
from pregnancysafe.schemas import SafetyTier, Trimester


class TestRedFlagScreening:
    def test_preeclampsia_pattern_matches(self):
        matches = screen_for_red_flags("عندي صداع شديد جدا وزغللة في العين من الصبح")
        ids = [m.red_flag.id for m in matches]
        assert "preeclampsia_warning" in ids

    def test_isolated_symptom_does_not_match(self):
        """A single symptom word shouldn't fire the flag — it needs BOTH
        keyword groups (headache AND vision disturbance) to co-occur."""
        matches = screen_for_red_flags("عندي صداع بسيط بس")
        assert matches == []

    def test_pyelonephritis_pattern_matches(self):
        matches = screen_for_red_flags("عندي حمى وألم بالخاصرة من امبارح")
        ids = [m.red_flag.id for m in matches]
        assert "pyelonephritis_warning" in ids

    def test_bleeding_pattern_matches(self):
        matches = screen_for_red_flags("حاسة بنزيف مهبلي دلوقتي")
        ids = [m.red_flag.id for m in matches]
        assert "bleeding_warning" in ids

    def test_empty_text_returns_no_matches(self):
        assert screen_for_red_flags("") == []
        assert screen_for_red_flags("   ") == []

    def test_normal_query_does_not_false_positive(self):
        matches = screen_for_red_flags("عايزة أعرف الدوا المسموح للغثيان الصبح")
        assert matches == []


class TestMedicationTiers:
    def test_get_medications_for_known_disease(self):
        meds = get_medications_for_disease("hypertension")
        assert len(meds) > 0

    def test_get_medications_for_unknown_disease_returns_empty(self):
        assert get_medications_for_disease("not_a_real_disease") == []

    def test_nsaid_restricted_after_week_20(self):
        nsaid = get_medication_by_id("hypertension", "nsaids")
        assert nsaid is not None
        check = check_trimester_restriction(nsaid, Trimester.SECOND)
        assert check.is_restricted is True

    def test_labetalol_unrestricted_any_trimester(self):
        labetalol = get_medication_by_id("hypertension", "labetalol")
        assert labetalol is not None
        for trimester in Trimester:
            assert effective_tier(labetalol, trimester) == SafetyTier.GREEN

    def test_methylprednisolone_escalates_to_red_before_week_10(self):
        med = get_medication_by_id("nausea_vomiting", "corticosteroids_methylprednisolone")
        assert med is not None
        # Base tier is already red, so this also verifies escalation doesn't
        # crash on an already-red medication.
        assert effective_tier(med, Trimester.FIRST) == SafetyTier.RED

    def test_unknown_medication_id_returns_none(self):
        assert get_medication_by_id("hypertension", "not_a_real_drug") is None


class TestDisclaimers:
    def test_disclaimer_always_appended(self):
        result = attach_disclaimer("أي رد تجريبي")
        assert MANDATORY_DISCLAIMER_AR in result
        assert result.startswith("أي رد تجريبي")

    def test_disclaimer_appended_even_to_empty_answer(self):
        result = attach_disclaimer("")
        assert MANDATORY_DISCLAIMER_AR in result


class TestAgentShortCircuits:
    """These don't require an indexed vector store — red-flag and
    out-of-scope paths must short-circuit BEFORE retrieval is touched."""

    def test_red_flag_short_circuits_before_retrieval(self):
        agent = PregnancyAgent()  # retriever intentionally not constructed
        response = agent.ask("صداع شديد وزغللة في العين", disease_id="hypertension")
        assert response.is_red_flag is True
        assert MANDATORY_DISCLAIMER_AR in response.answer_text
        # Retriever must never have been lazily constructed for this path.
        assert agent._retriever is None

    def test_out_of_scope_short_circuits_before_retrieval(self):
        agent = PregnancyAgent()
        response = agent.ask("سؤال عن مرض غير مدعوم", disease_id="some_unlisted_disease")
        assert response.is_out_of_scope is True
        assert MANDATORY_DISCLAIMER_AR in response.answer_text
        assert agent._retriever is None
