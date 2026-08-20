# PregnancySafe Blue Dashboard

This dashboard is an isolated UI layer for the existing PregnancySafe RAG project.

## What it changes
- Adds a new Streamlit dashboard file: `app/streamlit_dashboard.py`
- Uses the existing `PregnancyAgent.ask()` API.
- Does **not** modify `data/`, the vector store, ingestion, retrieval, safety, or agent code.
- Uses a blue/white visual theme.
- Shows answer, medication tier, retrieved evidence, citations, session history, and live retrieval analytics.

## Run
From the repository root:

```bash
streamlit run app/streamlit_dashboard.py
```

The normal project requirements already include Streamlit.

## Integration note
The dashboard expects the current backend response fields:
- `answer_text`
- `is_red_flag`
- `is_out_of_scope`
- `retrieved`
- `medication`
- `medication_tier`
- `citations`

Those fields are provided by the existing `PregnancyAgent.ask()` implementation.
