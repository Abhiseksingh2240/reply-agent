"""LLM agent: fusion / final recommendation."""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.llm.openrouter_client import OpenRouterClient
from app.llm import prompts
from app.models import ArbitrationOut


def run_agent(client: OpenRouterClient, settings: Settings, payload: dict[str, Any]) -> ArbitrationOut:
    msgs = prompts.arbitration_prompt(payload)
    data = client.chat_json(msgs, model=settings.openrouter_model_strong, max_tokens=400)
    return ArbitrationOut.model_validate(
        {
            "final_fraud_probability": float(data.get("final_fraud_probability", 0.0)),
            "final_economic_priority": float(data.get("final_economic_priority", 0.0)),
            "recommend_fraud": bool(data.get("recommend_fraud", False)),
            "explanation": str(data.get("explanation", "")),
        }
    )
