"""Groq-backed answer composer — optional drop-in for the default template
composer in agent/pregnancy_agent.py.

This module is intentionally isolated from the safety pipeline: it only
generates the *prose* of an answer from already-retrieved, already-filtered
citations (retrieval/retriever.py has already applied score_threshold).
Red-flag screening, scope checking, medication-tier lookup, and the
mandatory disclaimer all happen in agent/pregnancy_agent.py regardless of
which composer is plugged in — swapping composers can change *how the
answer reads*, never *whether the safety checks ran*.

Usage:
    from pregnancysafe.llm.groq_composer import create_groq_composer
    from pregnancysafe.agent import PregnancyAgent

    agent = PregnancyAgent(answer_composer=create_groq_composer())

Setup:
    1. Get a free key at https://console.groq.com/keys
    2. Copy .env.example to .env and set GROQ_API_KEY=<your key>
       (never paste a real key into a chat/prompt — .env stays local and is
       already excluded via .gitignore)
    3. pip install groq   (or: pip install -e ".[llm]")
"""

from __future__ import annotations

import os
from typing import Optional

from pregnancysafe.retrieval.retriever import RetrievalResult
from pregnancysafe.utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = "llama-3.3-70b-versatile"

_SYSTEM_PROMPT_AR = (
    "إنتِ مساعد معلومات طبية متخصص في صحة الحمل. "
    "لازم تجاوبي فقط بناءً على المقتطفات المسترجعة من المصادر الرسمية اللي هتوصلك في الرسالة، "
    "من غير ما تضيفي أي معلومة أو جرعة أو توصية علاجية مش موجودة فيها حرفيًا. "
    "لو المقتطفات مش كافية أو مش متعلقة بالسؤال، قولي بوضوح إن المعلومات المتاحة غير كافية "
    "بدل ما تخمّني أو تكملي من معرفتك العامة. "
    "جاوبي بالعربية المصرية، بأسلوب واضح ومباشر ومحترم، من غير مقدمات طويلة."
)


def _build_context_block(hits: list[RetrievalResult]) -> str:
    if not hits:
        return "(لا توجد مقتطفات مسترجعة لهذا السؤال)"
    lines = [f"[{i}] المصدر: {h.source_name}\n{h.text}" for i, h in enumerate(hits, start=1)]
    return "\n\n".join(lines)


def create_groq_composer(
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 600,
):
    """Build and return a Groq-backed composer callable.

    The `groq` import and the API-key check both happen lazily — at
    composer *call* time, not at `create_groq_composer()` call time — so
    building the agent doesn't fail just because a key isn't configured yet;
    it only fails if that composer actually gets invoked without one.
    """

    def _composer(query_text: str, hits: list[RetrievalResult]) -> str:
        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Copy .env.example to .env and add your key "
                "from https://console.groq.com/keys — see llm/groq_composer.py docstring."
            )

        try:
            from groq import Groq
        except ImportError as exc:
            raise ImportError(
                "groq package is required for the Groq composer. Install with: pip install groq"
            ) from exc

        client = Groq(api_key=resolved_key)
        context_block = _build_context_block(hits)

        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT_AR},
                    {
                        "role": "user",
                        "content": f"السؤال: {query_text}\n\nالمقتطفات المسترجعة:\n{context_block}",
                    },
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001
            # Fail safe rather than fail loud: surfacing a raw API/network
            # error to a pregnant user asking about medication safety is
            # worse than a plain, honest template answer.
            logger.error("Groq API call failed (%s) — falling back to template composer.", exc)
            from pregnancysafe.agent.pregnancy_agent import _default_answer_composer

            return _default_answer_composer(query_text, hits)

    return _composer
