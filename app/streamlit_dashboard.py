"""PregnancyCare — orange-themed RAG dashboard built on the existing PregnancyAgent.

Visually reproduces the "PregnancyCare" UI (Pregnancy Week / Condition / Medication
selectors, Answer + Medication safety block, Recommendation checklist, Risk Level
gauge, Evidence/Sources tabs, right-hand Evidence Assessment panel, and the
Pregnancy Diseases / Warning Signs sidebar pages), but stays wired to the real
`PregnancyAgent.ask()` backend instead of static demo data.

This file is intentionally isolated from the project's data/ directory and the
core RAG modules. It consumes the public AgentResponse fields already exposed by
PregnancyAgent.ask() (answer_text, is_red_flag, is_out_of_scope, retrieved,
medication, medication_tier, citations).

Run:
    streamlit run app/streamlit_dashboard.py
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

# ---------------------------------------------------------------------------
# Project path / imports
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv
from pregnancysafe.agent import PregnancyAgent
from pregnancysafe.safety.medication_tiers import get_medications_for_disease
from pregnancysafe.schemas import Trimester
from pregnancysafe.utils.config_loader import load_diseases

load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------------------
# Red-flag / warning-sign data source.
# Confirmed source: pregnancysafe.utils.config_loader.load_red_flags() returns
# the list of RedFlag objects parsed from data/medication_safety.json — the
# same objects safety/red_flags.py matches keywords against. We only read
# display fields defensively since the exact RedFlag attribute names beyond
# `.id` weren't shared.
# ---------------------------------------------------------------------------
def _load_red_flags() -> list[Any] | None:
    try:
        from pregnancysafe.utils.config_loader import load_red_flags
    except Exception:
        return None
    try:
        flags = load_red_flags()
    except Exception:
        return None
    return list(flags) if flags else None


# ---------------------------------------------------------------------------
# Theme — orange / cream ("PregnancyCare")
# ---------------------------------------------------------------------------
CREAM_BG = "#FBF8F5"
WHITE = "#FFFFFF"
ORANGE = "#E67E33"
ORANGE_DARK = "#C7631F"
BROWN = "#7A3E12"
BROWN_DARK = "#5C2E0D"
BORDER = "#EDE0D3"
TEXT = "#2B2320"
MUTED = "#8A7A6C"
GREEN = "#2F9E5C"
AMBER = "#B8860B"
RED = "#C0392B"
PALE_ORANGE = "#FDF1E6"

st.set_page_config(
    page_title="PregnancyCare",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {CREAM_BG};
            color: {TEXT};
        }}

        [data-testid="stSidebar"] {{
            background: {WHITE};
            border-right: 1px solid {BORDER};
        }}

        .brand-title {{
            font-size: 1.25rem;
            font-weight: 800;
            color: {TEXT};
        }}

        .brand-sub {{
            color: {MUTED};
            font-size: .78rem;
            text-transform: uppercase;
            letter-spacing: .06em;
            margin: 4px 0 14px;
        }}

        .page-title {{
            color: {TEXT};
            font-size: 1.9rem;
            font-weight: 800;
            margin-bottom: 2px;
        }}

        .page-sub {{
            color: {MUTED};
            font-size: .92rem;
            margin-bottom: 16px;
        }}

        .field-badge {{
            display: inline-block;
            background: {ORANGE};
            color: {WHITE};
            font-weight: 700;
            font-size: .82rem;
            padding: 5px 12px;
            border-radius: 6px;
            margin-bottom: 6px;
        }}

        .section-card {{
            background: {WHITE};
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 18px 20px;
        }}

        .answer-box {{
            background: {WHITE};
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 20px 22px;
            line-height: 1.75;
        }}

        .med-safety-line {{
            background: {PALE_ORANGE};
            border-radius: 10px;
            padding: 10px 14px;
            margin-bottom: 12px;
            font-size: .93rem;
        }}

        .rec-item, .diag-item {{
            display: flex;
            align-items: flex-start;
            gap: 8px;
            margin-bottom: 8px;
            font-size: .92rem;
        }}

        .disclaimer {{
            border-left: 4px solid {ORANGE};
            background: {PALE_ORANGE};
            padding: 12px 14px;
            border-radius: 10px;
            color: {MUTED};
            font-size: .78rem;
        }}

        .evidence-item {{
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 14px;
            margin-bottom: 12px;
            background: #FFFDFB;
        }}

        .relevance-badge {{
            display: inline-block;
            background: {PALE_ORANGE};
            color: {ORANGE_DARK};
            font-weight: 700;
            font-size: .74rem;
            padding: 3px 9px;
            border-radius: 999px;
        }}

        .kicker {{
            font-size: .72rem;
            text-transform: uppercase;
            letter-spacing: .07em;
            color: {ORANGE_DARK};
            font-weight: 800;
            margin-bottom: 4px;
        }}

        .key-info-line {{
            font-size: .88rem;
            margin-bottom: 6px;
        }}

        .risk-label-low {{ color: {GREEN}; font-weight: 800; }}
        .risk-label-mid {{ color: {AMBER}; font-weight: 800; }}
        .risk-label-high {{ color: {RED}; font-weight: 800; }}
        .risk-label-none {{ color: {MUTED}; font-weight: 800; }}

        button[kind="primary"] {{
            background: {BROWN} !important;
            border-color: {BROWN} !important;
        }}
        button[kind="primary"]:hover {{
            background: {BROWN_DARK} !important;
            border-color: {BROWN_DARK} !important;
        }}

        div[data-testid="stMetric"] {{
            background: {WHITE};
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 10px 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource
def get_agent() -> PregnancyAgent:
    composer = None
    if os.environ.get("GROQ_API_KEY"):
        try:
            from pregnancysafe.llm.groq_composer import create_groq_composer
            composer = create_groq_composer()
        except Exception:
            composer = None
    return PregnancyAgent(answer_composer=composer)


@st.cache_resource
def get_disease_options() -> dict[str, str]:
    diseases = load_diseases()
    return {disease.label_en: disease_id for disease_id, disease in diseases.items()}


TRIMESTER_LABELS = {
    Trimester.FIRST: "First trimester",
    Trimester.SECOND: "Second trimester",
    Trimester.THIRD: "Third trimester",
}


def trimester_label(value: Trimester) -> str:
    return TRIMESTER_LABELS.get(value, "Unknown trimester")


def tier_dot(tier: Any) -> str:
    return tier.emoji if tier else "⚪"


def tier_risk_score(tier: Any, is_red_flag: bool) -> float | None:
    if is_red_flag:
        return 0.95
    if tier is None:
        return None
    raw = getattr(tier, "value", "")
    return {"green": 0.15, "yellow": 0.40, "orange": 0.62, "red": 0.88}.get(raw)


def source_domain(url: str) -> str:
    if not url:
        return "Source"
    text = url.replace("https://", "").replace("http://", "")
    return text.split("/", 1)[0]


# ---------------------------------------------------------------------------
# Risk Level gauge (SVG semicircle, matches the "Risk Level" card)
# ---------------------------------------------------------------------------
def render_risk_gauge(score: float | None) -> None:
    cx, cy, r = 100, 95, 78
    if score is None:
        angle_deg = 90.0
        label, css = "Not assessed", "risk-label-none"
    else:
        score = max(0.0, min(1.0, score))
        angle_deg = 180.0 - (score * 180.0)
        if score < 0.35:
            label, css = "Low Risk", "risk-label-low"
        elif score < 0.7:
            label, css = "Moderate Risk", "risk-label-mid"
        else:
            label, css = "High Risk", "risk-label-high"

    rad = math.radians(angle_deg)
    nx = cx + (r - 8) * math.cos(rad)
    ny = cy - (r - 8) * math.sin(rad)

    svg = f"""
    <div style="text-align:center;">
    <svg viewBox="0 0 200 115" width="240" height="140">
      <path d="M 22 95 A 78 78 0 0 1 61 27" stroke="{GREEN}" stroke-width="14" fill="none" stroke-linecap="round"/>
      <path d="M 61 27 A 78 78 0 0 1 100 17" stroke="#D8C93A" stroke-width="14" fill="none" stroke-linecap="round"/>
      <path d="M 100 17 A 78 78 0 0 1 139 27" stroke="{ORANGE}" stroke-width="14" fill="none" stroke-linecap="round"/>
      <path d="M 139 27 A 78 78 0 0 1 178 95" stroke="{RED}" stroke-width="14" fill="none" stroke-linecap="round"/>
      <line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="{TEXT}" stroke-width="3" stroke-linecap="round"/>
      <circle cx="{cx}" cy="{cy}" r="6" fill="{TEXT}"/>
    </svg>
    <div class="{css}">{label}</div>
    </div>
    """
    st.markdown(svg, unsafe_allow_html=True)


def render_evidence(response: Any, max_items: int = 5) -> None:
    hits = getattr(response, "retrieved", []) or []
    if not hits:
        st.info("No evidence chunks passed the current retrieval threshold.")
        return

    for idx, hit in enumerate(hits[:max_items], start=1):
        citation_text = getattr(hit, "citation", None) or getattr(hit, "formatted_citation", None) or hit.text
        st.markdown(
            f"""
            <div class="evidence-item">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <strong>{idx}. {hit.source_name}</strong>
                <span class="relevance-badge">Relevance: {hit.relevance_score:.2f}</span>
              </div>
              <div style="color:{MUTED}; font-size:.78rem; margin:2px 0 6px;">
                {source_domain(getattr(hit, "source_url", ""))} • disease: {hit.disease_id}
              </div>
              <div style="font-size:.88rem;">{citation_text.strip()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def run_query(query: str, disease_id: str, week: int, medication_id: str | None) -> Any:
    agent = get_agent()
    start = time.perf_counter()
    response = agent.ask(
        query,
        disease_id=disease_id,
        gestational_week=week,
        medication_id=medication_id,
    )
    elapsed = time.perf_counter() - start
    st.session_state.last_response = response
    st.session_state.last_elapsed = elapsed
    st.session_state.profile_week = week
    st.session_state.profile_condition = disease_id
    return response


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
NAV_ITEMS = [
    ("🏛️", "Ask Assistant"),
    ("💊", "Medication Safety"),
    ("📋", "Pregnancy Diseases"),
    ("⚠️", "Warning Signs"),
    ("📚", "Evidence Library"),
]


def sidebar() -> str:
    with st.sidebar:
        st.markdown('<div class="brand-title">🤰 PregnancyCare</div>', unsafe_allow_html=True)
        st.markdown('<div class="brand-sub">RAG Assistant</div>', unsafe_allow_html=True)

        labels = [f"{icon}  {name}" for icon, name in NAV_ITEMS]
        choice = st.radio("Navigation", labels, label_visibility="collapsed")
        page = choice.split("  ", 1)[1]

        st.markdown("---")
        st.markdown("**Pregnancy Profile**")
        week = st.session_state.get("profile_week", 28)
        disease_options = get_disease_options()
        condition_id = st.session_state.get("profile_condition")
        condition_label = next(
            (label for label, did in disease_options.items() if did == condition_id), "—"
        )
        st.markdown(f"**Week:** {week}")
        st.markdown(f"**Condition:** {condition_label}")

        st.markdown(
            """
            <div class="disclaimer">
            This assistant is for informational purposes only and is not a substitute
            for professional medical advice.
            </div>
            """,
            unsafe_allow_html=True,
        )
    return page


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_ask() -> None:
    st.markdown('<div class="page-title">✨ Ask PregnancyCare Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Get evidence-based answers for your pregnancy health questions.</div>',
        unsafe_allow_html=True,
    )

    disease_options = get_disease_options()
    labels = list(disease_options.keys())
    default_week = int(st.session_state.get("profile_week", 28))

    with st.form("ask_form", border=True):
        row1 = st.columns([1, 1.6, 1.4])
        with row1[0]:
            st.markdown('<div class="field-badge">Pregnancy Week</div>', unsafe_allow_html=True)
            week = st.selectbox("Pregnancy week", list(range(1, 43)),
                                 index=list(range(1, 43)).index(default_week) if default_week in range(1, 43) else 27,
                                 label_visibility="collapsed")
        with row1[1]:
            st.markdown('<div class="field-badge">Condition</div>', unsafe_allow_html=True)
            disease_label = st.selectbox("Condition", labels, label_visibility="collapsed")
        with row1[2]:
            st.markdown('<div class="field-badge">Medication (Optional)</div>', unsafe_allow_html=True)
            disease_id = disease_options[disease_label]
            meds = get_medications_for_disease(disease_id)
            med_names = {m.name: m.id for m in meds}
            med_label = st.selectbox("Medication (optional)", ["None"] + list(med_names), label_visibility="collapsed")

        st.markdown('<div class="field-badge">Your Question</div>', unsafe_allow_html=True)
        query = st.text_area(
            "Your question",
            height=90,
            placeholder="Is it safe to take ibuprofen for headache during 28th week of pregnancy?",
            label_visibility="collapsed",
            key="ask_query_text",
        )

        submit_col, clear_col = st.columns([1, 1])
        with submit_col:
            submitted = st.form_submit_button("🔎 Ask Assistant", type="primary", use_container_width=True)
        with clear_col:
            cleared = st.form_submit_button("🗑️ Clear", use_container_width=True)

    if cleared:
        st.session_state.ask_query_text = ""
        st.session_state.last_response = None
        st.session_state.last_elapsed = None
        st.rerun()

    if submitted:
        text = query.strip() or f"Provide pregnancy guidance about {disease_label}."
        med_id = None if med_label == "None" else med_names.get(med_label)
        with st.spinner("Retrieving evidence and generating the grounded response..."):
            response = run_query(text, disease_id, int(week), med_id)
    else:
        response = st.session_state.get("last_response")

    left, right = st.columns([2.1, 1])

    with right:
        st.markdown('<div class="kicker">Evidence Assessment</div>', unsafe_allow_html=True)
        hits = getattr(response, "retrieved", []) or [] if response else []
        if not hits:
            st.markdown("**No Evidence**")
            st.progress(0.0)
            st.caption("Overall Confidence: 0.00 / 1.00")
        else:
            avg_rel = sum(h.relevance_score for h in hits) / len(hits)
            st.progress(min(1.0, avg_rel))
            st.caption(f"Overall Confidence: {avg_rel:.2f} / 1.00")

        st.markdown('<div class="kicker" style="margin-top:14px;">Top Sources</div>', unsafe_allow_html=True)
        if hits:
            for name in list(dict.fromkeys(h.source_name for h in hits))[:5]:
                st.write(f"• {name}")
        else:
            st.caption("None yet.")

        st.markdown('<div class="kicker" style="margin-top:14px;">Key Information</div>', unsafe_allow_html=True)
        if response is None:
            st.caption("Ask a question to see flags here.")
        else:
            medication = getattr(response, "medication", None)
            tier = getattr(response, "medication_tier", None)
            if medication and tier:
                st.markdown(f'<div class="key-info-line">{tier_dot(tier)} {medication.name}: {tier.label_ar}</div>', unsafe_allow_html=True)
            weekval = st.session_state.get("profile_week", int(week))
            st.markdown(f'<div class="key-info-line">📅 Trimester: {trimester_label(Trimester.from_week(int(weekval)))}</div>', unsafe_allow_html=True)

    with left:
        if response is None:
            a, b, c = st.columns(3)
            a.metric("Evidence-first", "RAG")
            b.metric("Retrieval", "ChromaDB")
            c.metric("Safety", "Built-in")
            st.info("Enter a question above to start the live RAG flow.")
            render_risk_gauge_card(None)
            return

        if response.is_red_flag:
            st.error("🚨 Red-flag screening triggered — seek prompt professional medical evaluation.")
        elif response.is_out_of_scope:
            st.warning("This question is outside the current validated evidence scope.")

        st.markdown("### ✅ Answer")
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)

        medication = getattr(response, "medication", None)
        tier = getattr(response, "medication_tier", None)
        if medication and tier:
            note = medication.notes_ar or ""
            st.markdown("**Medication safety**")
            st.markdown(
                f'<div class="med-safety-line">{medication.name}: {tier_dot(tier)} {note}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(response.answer_text)
        st.markdown("</div>", unsafe_allow_html=True)

        rec_col, risk_col = st.columns(2)
        with rec_col:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("**Recommendation**")
            for item in recommendation_items(tier, response):
                st.markdown(f'<div class="rec-item">✅ {item}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with risk_col:
            render_risk_gauge_card(tier_risk_score(tier, bool(response.is_red_flag)))

        tabs = st.tabs(["Evidence", "Sources"])
        with tabs[0]:
            render_evidence(response, max_items=10)
        with tabs[1]:
            if response.citations:
                st.code(response.citations, language="text")
            else:
                st.info("No citations were returned.")


def render_risk_gauge_card(score: float | None) -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("**Risk Level**")
    render_risk_gauge(score)
    st.markdown("</div>", unsafe_allow_html=True)


def recommendation_items(tier: Any, response: Any) -> list[str]:
    if response is None or tier is None:
        return [
            "Consult your healthcare provider for personalized advice",
            "Keep track of your symptoms",
            "Follow up at your next antenatal visit",
        ]
    raw = getattr(tier, "value", "")
    if raw == "green":
        return [
            "Generally considered safe for this trimester",
            "Still confirm with your healthcare provider",
            "Use the lowest effective dose",
        ]
    if raw in {"yellow", "orange"}:
        return [
            "Use only if the benefits outweigh the risks",
            "Consult a specialist before use",
            "Monitor closely for side effects",
        ]
    if raw == "red":
        return [
            "Avoid unless specifically prescribed by a specialist",
            "Discuss safer alternatives with your doctor",
            "Seek prompt consultation if already taken",
        ]
    return [
        "Consult your healthcare provider for personalized advice",
        "Keep track of your symptoms",
        "Follow up at your next antenatal visit",
    ]


def page_medication() -> None:
    st.markdown('<div class="page-title">Medication Safety</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Explore the validated medication catalog exposed by the project safety layer.</div>', unsafe_allow_html=True)

    diseases = get_disease_options()
    disease_label = st.selectbox("Condition area", list(diseases))
    disease_id = diseases[disease_label]
    meds = get_medications_for_disease(disease_id)

    if not meds:
        st.info("No structured medication entries are currently available for this disease area.")
        return

    search = st.text_input("Search medication", placeholder="Type a medication name...").strip().lower()
    filtered = [m for m in meds if not search or search in m.name.lower() or search in m.id.lower()]

    for med in filtered:
        with st.container(border=True):
            a, b, c = st.columns([1.4, 1, 2.2])
            a.markdown(f"### {med.name}")
            a.caption(med.id)
            b.write(f"**Tier:** {med.tier.emoji} {med.tier.label_ar}")
            c.write(med.notes_ar or "No note supplied.")


def page_diseases() -> None:
    st.markdown('<div class="page-title">Pregnancy Diseases</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Conditions currently covered by the knowledge base.</div>', unsafe_allow_html=True)

    diseases = load_diseases()
    search = st.text_input("Search a condition", placeholder="e.g. Hypertension").strip().lower()

    status_emoji = {"verified_open": "🟢", "manual_check": "🟡", "restricted": "🔴"}

    for disease_id, disease in diseases.items():
        label = disease.label_en
        if search and search not in label.lower() and search not in disease_id.lower():
            continue
        with st.container(border=True):
            st.markdown(f"### {label}  ·  {disease.label_ar}")
            st.caption(disease_id)

            meds = get_medications_for_disease(disease_id)
            if meds:
                st.caption(f"{len(meds)} tracked medication(s): " + ", ".join(m.name for m in meds[:6]))

            if disease.sources:
                st.markdown("**Sources**")
                for src in disease.sources:
                    emoji = status_emoji.get(getattr(src.status, "value", src.status), "⚪")
                    line = f"{emoji} [{src.name}]({src.url})"
                    if src.note:
                        line += f" — {src.note}"
                    st.markdown(line)


def page_warnings() -> None:
    st.markdown('<div class="page-title">Warning Signs</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Red-flag symptoms that require prompt medical evaluation.</div>', unsafe_allow_html=True)

    red_flags = _load_red_flags()
    if not red_flags:
        st.warning(
            "`load_red_flags()` returned nothing (or couldn't be imported). "
            "Check that `data/medication_safety.json` has a red-flag section "
            "and that `pregnancysafe.utils.config_loader.load_red_flags` is reachable."
        )
        return

    for flag in red_flags:
        with st.container(border=True):
            st.markdown(f"⚠️ **{flag.suspected_condition}**")
            st.caption(flag.id)
            st.write(f"**Pattern:** {flag.pattern_ar}")
            st.markdown(f"**Action:** {flag.action_ar}")


def page_evidence() -> None:
    st.markdown('<div class="page-title">Evidence Library</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Inspect the evidence returned by your most recent RAG query.</div>', unsafe_allow_html=True)
    response = st.session_state.get("last_response")
    if response is None:
        st.info("Run a query first; the returned evidence will appear here.")
        return
    render_evidence(response, max_items=20)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    inject_css()
    if "last_response" not in st.session_state:
        st.session_state.last_response = None

    page = sidebar()
    if page == "Ask Assistant":
        page_ask()
    elif page == "Medication Safety":
        page_medication()
    elif page == "Pregnancy Diseases":
        page_diseases()
    elif page == "Warning Signs":
        page_warnings()
    else:
        page_evidence()


if __name__ == "__main__":
    main()