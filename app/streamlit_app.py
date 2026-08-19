"""PregnancySafe — Streamlit demo UI.

Input: gestational week + free-text clinical question + optional
disease/medication selection.

Output:
- Red-flag alerts
- Medication safety tier
- Grounded RAG answer
- Citations
- Mandatory disclaimer

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parents[1] / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import streamlit as st
from dotenv import load_dotenv

from pregnancysafe.agent import PregnancyAgent
from pregnancysafe.safety.medication_tiers import (
    get_medications_for_disease,
)
from pregnancysafe.schemas import Trimester
from pregnancysafe.utils.config_loader import load_diseases


load_dotenv()


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PregnancySafe",
    page_icon="🤰",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource
def get_agent() -> PregnancyAgent:
    """Create and cache the PregnancySafe agent."""

    answer_composer = None

    if os.environ.get("GROQ_API_KEY"):
        from pregnancysafe.llm.groq_composer import (
            create_groq_composer,
        )

        answer_composer = create_groq_composer()

    return PregnancyAgent(
        answer_composer=answer_composer
    )


@st.cache_resource
def get_disease_options() -> dict[str, str]:
    """Return English disease labels mapped to disease IDs."""

    diseases = load_diseases()

    return {
        disease.label_en: disease_id
        for disease_id, disease in diseases.items()
    }


# ---------------------------------------------------------------------------
# English trimester labels
# ---------------------------------------------------------------------------

TRIMESTER_LABELS = {
    Trimester.FIRST: "First trimester",
    Trimester.SECOND: "Second trimester",
    Trimester.THIRD: "Third trimester",
}


def get_trimester_label(trimester: Trimester) -> str:
    """Return an English label for the trimester."""

    return TRIMESTER_LABELS.get(
        trimester,
        "Unknown trimester",
    )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:

    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------

    st.title("🤰 PregnancySafe")

    st.caption(
        "A pregnancy medical guidance system for symptom and medication "
        "safety, grounded in official medical sources including "
        "WHO, NICE, ACOG, FDA, and MotherToBaby."
    )

    # -----------------------------------------------------------------------
    # Disease options
    # -----------------------------------------------------------------------

    disease_options = get_disease_options()

    # -----------------------------------------------------------------------
    # Input form
    # -----------------------------------------------------------------------

    with st.form("query_form"):

        col1, col2 = st.columns(2)

        with col1:

            gestational_week = st.number_input(
                "Gestational age (weeks)",
                min_value=1,
                max_value=42,
                value=20,
                step=1,
            )

        with col2:

            disease_label = st.selectbox(
                "Condition / Topic",
                options=list(disease_options.keys()),
            )

        # -------------------------------------------------------------------
        # Clinical question
        # -------------------------------------------------------------------

        symptoms = st.text_area(
            "Clinical question / Symptoms",
            placeholder=(
                "Example: A pregnant woman has high blood pressure. "
                "According to the NICE guideline, how should "
                "hypertension in pregnancy be diagnosed and managed?"
            ),
            height=150,
        )

        # -------------------------------------------------------------------
        # Medication
        # -------------------------------------------------------------------

        disease_id = disease_options[disease_label]

        med_options = {
            medication.name: medication.id
            for medication in get_medications_for_disease(
                disease_id
            )
        }

        med_label = st.selectbox(
            "Medication to check (optional)",
            options=["-- None --"] + list(med_options.keys()),
        )

        submitted = st.form_submit_button(
            "Check Safety"
        )

    # -----------------------------------------------------------------------
    # Wait for submission
    # -----------------------------------------------------------------------

    if not submitted:
        return

    # -----------------------------------------------------------------------
    # Trimester
    # -----------------------------------------------------------------------

    trimester = Trimester.from_week(
        int(gestational_week)
    )

    trimester_label = get_trimester_label(
        trimester
    )

    st.info(
        f"Current trimester: **{trimester_label}** "
        f"(week {gestational_week})"
    )

    # -----------------------------------------------------------------------
    # Build query
    # -----------------------------------------------------------------------

    medication_id = (
        med_options.get(med_label)
        if med_label != "-- None --"
        else None
    )

    query_text = symptoms.strip()

    if not query_text:

        query_text = (
            f"Provide medical guidance about "
            f"{disease_label} during pregnancy."
        )

    # -----------------------------------------------------------------------
    # Run agent
    # -----------------------------------------------------------------------

    agent = get_agent()

    with st.spinner(
        "Retrieving medical evidence and generating the response..."
    ):

        response = agent.ask(
            query_text,
            disease_id=disease_id,
            gestational_week=int(gestational_week),
            medication_id=medication_id,
        )

    # -----------------------------------------------------------------------
    # Red flags
    # -----------------------------------------------------------------------

    if response.is_red_flag:

        st.error(
            response.answer_text
        )

        return

    # -----------------------------------------------------------------------
    # Out of scope
    # -----------------------------------------------------------------------

    if response.is_out_of_scope:

        st.warning(
            response.answer_text
        )

        return

    # -----------------------------------------------------------------------
    # Medication safety
    # -----------------------------------------------------------------------

    if (
        response.medication
        and response.medication_tier
    ):

        tier = response.medication_tier

        tier_display = {
            "green": st.success,
            "yellow": st.warning,
            "orange": st.warning,
            "red": st.error,
        }.get(
            tier.value,
            st.info,
        )

        tier_display(
            f"{tier.emoji} "
            f"{response.medication.name}: "
            f"{tier.label_en}"
        )

    # -----------------------------------------------------------------------
    # Answer
    # -----------------------------------------------------------------------

    st.markdown("### Answer")

    st.write(
        response.answer_text
    )

    # -----------------------------------------------------------------------
    # Sources
    # -----------------------------------------------------------------------

    if response.citations:

        with st.expander(
            "📚 Sources and Citations"
        ):

            st.text(
                response.citations
            )

    # -----------------------------------------------------------------------
    # System note
    # -----------------------------------------------------------------------

    st.caption(
        "Note: If no evidence is retrieved, the relevant PDF sources "
        "may need to be downloaded and indexed first "
        "(scripts/run_ingestion.py followed by scripts/build_index.py)."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()