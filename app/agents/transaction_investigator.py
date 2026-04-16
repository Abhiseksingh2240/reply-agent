"""LLM agent: transaction behavior."""

from __future__ import annotations

from typing import Any

from app.llm.openrouter_client import OpenRouterClient
from app.llm import prompts
from app.models import TransactionInvestigatorOut


def run_agent(client: OpenRouterClient, payload: dict[str, Any]) -> TransactionInvestigatorOut:
    msgs = prompts.transaction_investigator_prompt(payload)
    data = client.chat_json(msgs)
    return TransactionInvestigatorOut.model_validate(
        {
            "suspicious": bool(data.get("suspicious", False)),
            "confidence": float(data.get("confidence", 0.0)),
            "reasons": list(data.get("reasons") or []),
        }
    )
