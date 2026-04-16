"""Langfuse session helpers aligned with Langfuse v3."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

import ulid

from app.config import Settings, is_langfuse_configured


def _normalize_team_name(name: str) -> str:
    s = (name or "reply-agent").strip().replace(" ", "-")
    return s or "reply-agent"


def new_session_id(settings: Settings) -> str:
    """Challenge format: TEAM_NAME-ULID."""
    return f"{_normalize_team_name(settings.team_name)}-{ulid.new().str}"


class LangfuseRun:
    """Thin wrapper for flush + compatibility with existing pipeline hooks.

    In Langfuse v3, traces are primarily created via `@observe()` and
    `langfuse.langchain.CallbackHandler`, which we now use in OpenRouterClient.
    """

    def __init__(self, settings: Settings, session_id: str, dataset: str):
        self.settings = settings
        self.session_id = session_id
        self.dataset = dataset
        self._client = None

        os.environ["LANGFUSE_MEDIA_UPLOAD_ENABLED"] = (
            "true" if settings.langfuse_media_upload_enabled else "false"
        )

        if is_langfuse_configured(settings):
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host.rstrip("/"),
            )

    def start_trace(self) -> None:
        # Trace creation handled by @observe() + CallbackHandler during LLM calls.
        return

    def flush(self) -> None:
        if self._client:
            self._client.flush()

    @contextmanager
    def span(self, name: str, metadata: dict[str, Any] | None = None) -> Iterator[None]:
        _ = (name, metadata)
        yield

    def log_llm(
        self,
        name: str,
        model: str,
        input_text: str,
        output_obj: dict[str, Any],
        usage: dict[str, Any] | None = None,
    ) -> None:
        _ = (name, model, input_text, output_obj, usage)
        return

    def log_event(self, name: str, metadata: dict[str, Any]) -> None:
        _ = (name, metadata)
        return
