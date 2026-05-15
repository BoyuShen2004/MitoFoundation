"""LangChain ``ChatOpenAI`` factory for scrape **routing** only (`llm_strategy.plan_scrape_route`).

Report markdown is heuristic-only; see ``llm_report.py``.
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI


def build_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "ollama")
    base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
    model_name = os.getenv("OPENAI_MODEL", "qwen3.5:9b")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    llm_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
    kwargs: dict = {
        "model": model_name,
        "temperature": temperature,
        "api_key": api_key,
        "base_url": base_url,
        "max_retries": max(0, llm_retries),
    }
    raw_timeout = os.getenv("LLM_TIMEOUT", "1200")
    try:
        kwargs["timeout"] = max(30.0, float(str(raw_timeout).strip()))
    except ValueError:
        kwargs["timeout"] = 1200.0
    return ChatOpenAI(**kwargs)
