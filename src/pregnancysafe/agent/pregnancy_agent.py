"""PregnancyAgent — the single coordinator the application talks to.

Pipeline:
  1. Red-flag screening
  2. Scope check
  3. Retrieval
  4. Evidence filtering
  5. Medication detection + safety lookup
  6. Grounded answer composition
  7. Mandatory English disclaimer

This module deliberately does NOT call an LLM by default.
An optional answer_composer can be injected later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from pregnancysafe.retrieval.citation_formatter import format_citations
from pregnancysafe.retrieval.retriever import (
    RetrievalResult,
    Retriever,
)
from pregnancysafe.safety.medication_tiers import (
    effective_tier,
    get_medication_by_id,
    get_medications_for_disease,
)
from pregnancysafe.safety.red_flags import (
    RedFlagMatch,
    screen_for_red_flags,
)
from pregnancysafe.schemas import (
    Medication,
    SafetyTier,
    Trimester,
)
from pregnancysafe.utils.config_loader import load_diseases
from pregnancysafe.utils.logging_config import get_logger


logger = get_logger(__name__)


AnswerComposer = Callable[
    [str, list[RetrievalResult]],
    str,
]


# Minimum relevance accepted by the agent.
MIN_AGENT_RELEVANCE_SCORE = 0.50


# ---------------------------------------------------------------------------
# English disclaimer
# ---------------------------------------------------------------------------

ENGLISH_DISCLAIMER = (
    "\n\nDisclaimer: This information is for educational purposes only "
    "and is based on retrieved medical guidance. It is not a substitute "
    "for evaluation or treatment by a qualified healthcare professional. "
    "Clinical decisions should be made by the patient's treating clinician."
)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Normalize whitespace in retrieved evidence."""

    return " ".join(text.split()).strip()


def _normalize_for_comparison(text: str) -> str:
    """Normalize text for duplicate detection."""

    return _clean_text(text).lower()


def _is_duplicate_text(
    text: str,
    existing_texts: list[str],
) -> bool:
    """Detect exact or near-duplicate retrieved passages."""

    normalized = _normalize_for_comparison(text)

    if not normalized:
        return True

    for existing in existing_texts:

        existing_normalized = _normalize_for_comparison(
            existing
        )

        if normalized == existing_normalized:
            return True

        if (
            len(normalized) > 120
            and len(existing_normalized) > 120
        ):

            if (
                normalized in existing_normalized
                or existing_normalized in normalized
            ):
                return True

    return False


def _truncate(text: str, max_length: int = 500) -> str:
    """Keep evidence readable without cutting words in half."""

    text = _clean_text(text)

    if len(text) <= max_length:
        return text

    shortened = text[:max_length]

    if " " in shortened:
        shortened = shortened.rsplit(" ", 1)[0]

    return shortened + "..."


def _source_priority(source_name: str) -> int:
    """Prefer major guideline sources when relevance is similar."""

    source = source_name.lower()

    if "nice" in source:
        return 4

    if "who" in source:
        return 3

    if "acog" in source:
        return 3

    if "fda" in source:
        return 2

    if "mothertobaby" in source:
        return 2

    return 1


# ---------------------------------------------------------------------------
# Evidence selection
# ---------------------------------------------------------------------------

def _select_unique_hits(
    hits: list[RetrievalResult],
    max_hits: int = 8,
) -> list[RetrievalResult]:
    """Select strong, diverse, non-duplicate evidence."""

    ordered = sorted(
        hits,
        key=lambda hit: (
            _source_priority(hit.source_name),
            hit.relevance_score,
        ),
        reverse=True,
    )

    selected: list[RetrievalResult] = []
    used_texts: list[str] = []

    for hit in ordered:

        text = _clean_text(hit.text)

        if not text:
            continue

        if _is_duplicate_text(
            text,
            used_texts,
        ):
            continue

        selected.append(hit)
        used_texts.append(text)

        if len(selected) >= max_hits:
            break

    return selected


# ---------------------------------------------------------------------------
# Topic detection
# ---------------------------------------------------------------------------

def _contains_any(
    text: str,
    keywords: list[str],
) -> bool:
    """Return True if any keyword appears in text."""

    return any(
        keyword in text
        for keyword in keywords
    )


def _classify_evidence(
    hit: RetrievalResult,
) -> str:
    """Lightweight evidence classification.

    This is intentionally rule-based. It does not invent medical facts.
    """

    text = _clean_text(hit.text).lower()

    diagnosis_keywords = [
        "diagnos",
        "diagnostic",
        "blood pressure of 140/90",
        "140/90 mmhg",
        "first episode of hypertension",
        "assessment",
        "identify hypertension",
        "measure blood pressure",
    ]

    management_keywords = [
        "management",
        "manage",
        "treatment",
        "treat",
        "antihypertensive",
        "severe hypertension",
        "160/110",
        "160 mmhg",
        "110 mmhg",
        "start antihypertensive",
        "continue antihypertensive",
    ]

    monitoring_keywords = [
        "monitoring",
        "monitor",
        "follow-up",
        "follow up",
        "blood pressure monitoring",
        "measure blood pressure",
        "response to treatment",
    ]

    if _contains_any(
        text,
        management_keywords,
    ):
        return "management"

    if _contains_any(
        text,
        diagnosis_keywords,
    ):
        return "diagnosis"

    if _contains_any(
        text,
        monitoring_keywords,
    ):
        return "monitoring"

    return "general"


# ---------------------------------------------------------------------------
# Answer composer
# ---------------------------------------------------------------------------

def _default_answer_composer(
    query: str,
    hits: list[RetrievalResult],
) -> str:
    """Compose a concise English answer grounded only in retrieved evidence.

    No clinical recommendation is generated unless it is supported by
    retrieved evidence.
    """

    if not hits:
        return (
            "I cannot provide a reliable answer from the approved "
            "medical sources currently available in PregnancySafe."
        )

    selected_hits = _select_unique_hits(
        hits,
        max_hits=8,
    )

    if not selected_hits:
        return (
            "The retrieved sources did not contain sufficiently clear "
            "evidence to answer this question reliably."
        )

    diagnosis: list[RetrievalResult] = []
    management: list[RetrievalResult] = []
    monitoring: list[RetrievalResult] = []
    general: list[RetrievalResult] = []

    for hit in selected_hits:

        category = _classify_evidence(hit)

        if category == "diagnosis":
            diagnosis.append(hit)

        elif category == "management":
            management.append(hit)

        elif category == "monitoring":
            monitoring.append(hit)

        else:
            general.append(hit)

    sections: list[str] = []

    # -----------------------------------------------------------------------
    # Diagnosis
    # -----------------------------------------------------------------------

    if diagnosis:

        lines = []

        for hit in diagnosis[:3]:

            text = _truncate(
                hit.text,
                500,
            )

            lines.append(
                f"- {text}"
            )

        if lines:

            sections.append(
                "### Diagnosis and assessment\n\n"
                + "\n\n".join(lines)
            )

    # -----------------------------------------------------------------------
    # Management
    # -----------------------------------------------------------------------

    if management:

        lines = []

        for hit in management[:3]:

            text = _truncate(
                hit.text,
                500,
            )

            lines.append(
                f"- {text}"
            )

        if lines:

            sections.append(
                "### Management\n\n"
                + "\n\n".join(lines)
            )

    # -----------------------------------------------------------------------
    # Monitoring
    # -----------------------------------------------------------------------

    if monitoring:

        lines = []

        for hit in monitoring[:3]:

            text = _truncate(
                hit.text,
                500,
            )

            lines.append(
                f"- {text}"
            )

        if lines:

            sections.append(
                "### Monitoring and follow-up\n\n"
                + "\n\n".join(lines)
            )

    # -----------------------------------------------------------------------
    # General evidence
    # -----------------------------------------------------------------------

    if general and len(sections) < 2:

        lines = []

        for hit in general[:2]:

            text = _truncate(
                hit.text,
                500,
            )

            lines.append(
                f"- {text}"
            )

        if lines:

            sections.append(
                "### Relevant guideline evidence\n\n"
                + "\n\n".join(lines)
            )

    if not sections:

        return (
            "The retrieved sources did not contain sufficiently clear "
            "evidence to answer this question reliably."
        )

    return (
        "Based on the retrieved medical guidelines:\n\n"
        + "\n\n".join(sections)
    )


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    answer_text: str
    is_red_flag: bool = False
    is_out_of_scope: bool = False
    red_flag_matches: list[RedFlagMatch] = field(
        default_factory=list
    )
    retrieved: list[RetrievalResult] = field(
        default_factory=list
    )
    medication: Optional[Medication] = None
    medication_tier: Optional[SafetyTier] = None
    citations: str = ""


# ---------------------------------------------------------------------------
# Medication detection
# ---------------------------------------------------------------------------

def _detect_medication(
    query_text: str,
    disease_id: Optional[str] = None,
) -> tuple[
    Optional[Medication],
    Optional[str],
]:
    """Detect a documented medication.

    Only medications present in the validated medication catalog
    are considered.
    """

    query_lower = query_text.lower()

    if disease_id:

        medications = get_medications_for_disease(
            disease_id
        )

    else:

        medications = []

        for known_disease in load_diseases().keys():

            medications.extend(
                get_medications_for_disease(
                    known_disease
                )
            )

    medications = sorted(
        medications,
        key=lambda medication: len(
            medication.name
        ),
        reverse=True,
    )

    for medication in medications:

        names_to_check = [
            medication.name.lower(),
            medication.id.lower().replace(
                "_",
                " ",
            ),
        ]

        for name in names_to_check:

            if name and name in query_lower:

                return (
                    medication,
                    medication.id,
                )

    return None, None


# ---------------------------------------------------------------------------
# Main Agent
# ---------------------------------------------------------------------------

class PregnancyAgent:
    """Main PregnancySafe coordinator."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        answer_composer: Optional[AnswerComposer] = None,
    ) -> None:

        self._retriever = retriever

        self._answer_composer = (
            answer_composer
            or _default_answer_composer
        )

        self._known_diseases = set(
            load_diseases().keys()
        )

    def _get_retriever(self) -> Retriever:
        """Create the retriever lazily."""

        if self._retriever is None:
            self._retriever = Retriever()

        return self._retriever

    def ask(
        self,
        query_text: str,
        *,
        disease_id: Optional[str] = None,
        gestational_week: Optional[int] = None,
        medication_id: Optional[str] = None,
    ) -> AgentResponse:

        # ================================================================
        # 1. RED-FLAG SCREENING
        # ================================================================

        red_flag_matches = screen_for_red_flags(
            query_text
        )

        if red_flag_matches:

            logger.warning(
                "Red flag(s) triggered: %s",
                [
                    match.red_flag.id
                    for match in red_flag_matches
                ],
            )

            lines = [
                (
                    f"- {match.red_flag.action_ar} "
                    f"(suspected: "
                    f"{match.red_flag.suspected_condition})"
                )
                for match in red_flag_matches
            ]

            answer = (
                "The symptoms described may indicate a "
                "potential medical emergency:\n\n"
                + "\n".join(lines)
                + "\n\n"
                "Please contact your healthcare professional "
                "or seek emergency medical care immediately. "
                "PregnancySafe is not a substitute for emergency care."
                + ENGLISH_DISCLAIMER
            )

            return AgentResponse(
                answer_text=answer,
                is_red_flag=True,
                red_flag_matches=red_flag_matches,
            )

        # ================================================================
        # 2. SCOPE CHECK
        # ================================================================

        if (
            disease_id is not None
            and disease_id not in self._known_diseases
        ):

            answer = (
                "This question is outside the current scope "
                "of PregnancySafe.\n\n"
                "Available disease areas: "
                + ", ".join(
                    sorted(self._known_diseases)
                )
                + "\n\n"
                "Please consult a qualified healthcare professional "
                "for conditions outside the available evidence base."
                + ENGLISH_DISCLAIMER
            )

            return AgentResponse(
                answer_text=answer,
                is_out_of_scope=True,
            )

        # ================================================================
        # 3. RETRIEVAL
        # ================================================================

        try:

            raw_hits = self._get_retriever().retrieve(
                query_text,
                disease_id=disease_id,
            )

            hits = [
                hit
                for hit in raw_hits
                if hit.relevance_score
                >= MIN_AGENT_RELEVANCE_SCORE
            ]

            if raw_hits and not hits:

                logger.info(
                    "Retrieved %d weak chunks; none passed "
                    "agent threshold %.2f.",
                    len(raw_hits),
                    MIN_AGENT_RELEVANCE_SCORE,
                )

        except ImportError as exc:

            logger.warning(
                "Retrieval unavailable: %s",
                exc,
            )

            hits = []

        # ================================================================
        # 4. DETERMINE DISEASE
        # ================================================================

        detected_disease_id = disease_id

        if (
            detected_disease_id is None
            and hits
        ):

            strongest_hit = max(
                hits,
                key=lambda hit: hit.relevance_score,
            )

            detected_disease_id = (
                strongest_hit.disease_id
            )

        # ================================================================
        # 5. MEDICATION SAFETY LOOKUP
        # ================================================================

        medication: Optional[Medication] = None
        tier: Optional[SafetyTier] = None

        detected_medication_id = medication_id

        if not detected_medication_id:

            detected_medication, detected_id = (
                _detect_medication(
                    query_text,
                    disease_id=detected_disease_id,
                )
            )

            if detected_medication is not None:

                medication = detected_medication
                detected_medication_id = detected_id

                logger.info(
                    "Medication detected: %s (%s)",
                    medication.name,
                    medication.id,
                )

        if detected_medication_id:

            if detected_disease_id:

                medication = get_medication_by_id(
                    detected_disease_id,
                    detected_medication_id,
                )

            else:

                for known_disease in self._known_diseases:

                    medication = get_medication_by_id(
                        known_disease,
                        detected_medication_id,
                    )

                    if medication is not None:

                        detected_disease_id = (
                            known_disease
                        )

                        break

        # ================================================================
        # 6. TRIMESTER-SPECIFIC MEDICATION TIER
        # ================================================================

        if medication:

            trimester = (
                Trimester.from_week(
                    gestational_week
                )
                if gestational_week is not None
                else Trimester.SECOND
            )

            tier = effective_tier(
                medication,
                trimester,
            )

            logger.info(
                "Medication safety resolved: %s -> %s",
                medication.id,
                tier,
            )

        # ================================================================
        # 7. GROUNDED ANSWER
        # ================================================================

        if not hits:

            answer = (
                "I cannot provide a reliable answer from the "
                "approved medical sources currently available "
                "in PregnancySafe.\n\n"
                "The question may be outside the available "
                "evidence base, or there may not be sufficient "
                "retrieved evidence to answer it accurately."
            )

        else:

            answer = self._answer_composer(
                query_text,
                hits,
            )

        # ================================================================
        # 8. MEDICATION INFORMATION
        # ================================================================

        if medication and tier:

            medication_section = (
                "### Medication safety\n\n"
                f"{medication.name}: "
                f"{tier.emoji} {tier.label_ar}\n"
                f"{medication.notes_ar or ''}"
            )

            answer = (
                medication_section
                + "\n\n"
                + answer
            )

        # ================================================================
        # 9. ENGLISH DISCLAIMER
        # ================================================================

        answer = (
            answer
            + ENGLISH_DISCLAIMER
        )

        # ================================================================
        # 10. RESPONSE
        # ================================================================

        return AgentResponse(
            answer_text=answer,
            retrieved=hits,
            medication=medication,
            medication_tier=tier,
            citations=format_citations(hits),
        )