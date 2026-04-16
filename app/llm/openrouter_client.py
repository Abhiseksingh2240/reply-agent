"""OpenRouter client with LangChain+Langfuse callback tracking."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from app.config import Settings

try:
    from langfuse import observe
except Exception:  # pragma: no cover
    def observe(*_args: Any, **_kwargs: Any):
        def _decorator(fn: Any) -> Any:
            return fn
        return _decorator


class OpenRouterClient:
    def __init__(self, settings: Settings, session_id: str | None = None):
        self.settings = settings
        self.session_id = session_id
        base = settings.openrouter_base_url.rstrip("/")
        self._url = f"{base}/chat/completions"
        self._model_cache: dict[str, Any] = {}

    def _build_langchain_model(self, model: str):
        from langchain_openai import ChatOpenAI

        if model in self._model_cache:
            return self._model_cache[model]
        llm = ChatOpenAI(
            api_key=self.settings.openrouter_api_key,
            base_url=self.settings.openrouter_base_url,
            model=model,
            temperature=self.settings.openrouter_temperature,
            max_retries=self.settings.openrouter_max_retries,
            timeout=self.settings.openrouter_timeout_s,
        )
        self._model_cache[model] = llm
        return llm

    @staticmethod
    def _supports_langchain_message(messages: list[dict[str, Any]]) -> bool:
        for m in messages:
            if not isinstance(m, dict):
                return False
            content = m.get("content")
            if not isinstance(content, str):
                # e.g. input_audio content list -> keep HTTP path for now.
                return False
        return True

    @staticmethod
    def _to_langchain_messages(messages: list[dict[str, Any]]):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        out = []
        for m in messages:
            role = str(m.get("role") or "user").lower()
            content = str(m.get("content") or "")
            if role == "system":
                out.append(SystemMessage(content=content))
            elif role == "assistant":
                out.append(AIMessage(content=content))
            else:
                out.append(HumanMessage(content=content))
        return out

    @observe()
    def _invoke_langchain(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int,
        json_mode: bool,
        session_id: str | None,
    ) -> tuple[str, dict[str, Any]]:
        llm = self._build_langchain_model(model)
        lc_messages = self._to_langchain_messages(messages)

        cfg: dict[str, Any] = {}
        sid = session_id or self.session_id
        if sid:
            cfg["metadata"] = {"langfuse_session_id": sid}

        try:
            from langfuse.langchain import CallbackHandler

            cfg["callbacks"] = [CallbackHandler()]
        except Exception:
            pass

        # response_format json_object is not uniformly exposed in all wrappers;
        # we enforce JSON in prompts and parse strictly in chat_json.
        if max_tokens > 0:
            llm = llm.bind(max_tokens=max_tokens)

        response = llm.invoke(lc_messages, config=cfg)
        content = response.content
        if isinstance(content, list):
            text = "\n".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in content)
        else:
            text = str(content or "")

        usage = None
        meta = getattr(response, "response_metadata", None)
        if isinstance(meta, dict):
            usage = meta.get("token_usage") or meta.get("usage")
        return text, {"usage": usage}

    def _invoke_httpx(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int,
        json_mode: bool,
    ) -> tuple[str, dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "model": model,
            "temperature": self.settings.openrouter_temperature,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        last_err: Exception | None = None
        for attempt in range(self.settings.openrouter_max_retries):
            try:
                with httpx.Client(timeout=self.settings.openrouter_timeout_s) as client:
                    resp = client.post(self._url, headers=headers, json=body)
                    resp.raise_for_status()
                    data = resp.json()
                content = data["choices"][0]["message"].get("content") or ""
                return str(content), data
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"OpenRouter request failed after retries: {last_err}")

    def complete_chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        max_tokens: int = 500,
        json_mode: bool = False,
        session_id: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        model = model or self.settings.openrouter_model_cheap

        if self._supports_langchain_message(messages):
            try:
                return self._invoke_langchain(
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    session_id=session_id,
                )
            except Exception:
                # Fallback to plain HTTP if wrapper cannot parse/provider mismatch.
                pass

        return self._invoke_httpx(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )

    def chat_json(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int = 500,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        content, data = self.complete_chat(
            messages,  # type: ignore[arg-type]
            model=model,
            max_tokens=max_tokens,
            json_mode=True,
            session_id=session_id,
        )
        parsed = json.loads(content)
        usage = data.get("usage") if isinstance(data, dict) else None
        if isinstance(parsed, dict):
            parsed["_usage"] = usage
            return parsed
        return {"value": parsed, "_usage": usage}
