"""LLM agent: phishing / manipulation context."""

from __future__ import annotations

from typing import Any

from app.llm.openrouter_client import OpenRouterClient
from app.llm import prompts
from app.models import SocialEngineeringOut


def run_agent(client: OpenRouterClient, payload: dict[str, Any]) -> SocialEngineeringOut:
    msgs = prompts.social_engineering_prompt(payload)
    data = client.chat_json(msgs)
    return SocialEngineeringOut.model_validate(
        {
            "manipulation_likelihood": float(data.get("manipulation_likelihood", 0.0)),
            "confidence": float(data.get("confidence", 0.0)),
            "reasons": list(data.get("reasons") or []),
        }
    )
