"""LLM agent: audio metadata / transcript."""

from __future__ import annotations

from typing import Any

from app.llm.openrouter_client import OpenRouterClient
from app.llm import prompts
from app.models import AudioSignalOut


def run_agent(client: OpenRouterClient, payload: dict[str, Any]) -> AudioSignalOut:
    msgs = prompts.audio_prompt(payload)
    data = client.chat_json(msgs)
    return AudioSignalOut.model_validate(
        {
            "scam_signal_score": float(data.get("scam_signal_score", 0.0)),
            "confidence": float(data.get("confidence", 0.0)),
            "reasons": list(data.get("reasons") or []),
        }
    )
