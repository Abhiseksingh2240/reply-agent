"""LLM agent: mobility plausibility."""

from __future__ import annotations

from typing import Any

from app.llm.openrouter_client import OpenRouterClient
from app.llm import prompts
from app.models import MobilityPresenceOut


def run_agent(client: OpenRouterClient, payload: dict[str, Any]) -> MobilityPresenceOut:
    msgs = prompts.mobility_prompt(payload)
    data = client.chat_json(msgs)
    return MobilityPresenceOut.model_validate(
        {
            "presence_plausible": bool(data.get("presence_plausible", True)),
            "confidence": float(data.get("confidence", 0.0)),
            "reasons": list(data.get("reasons") or []),
        }
    )
