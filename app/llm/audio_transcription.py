"""Optional OpenRouter multimodal transcription (input_audio)."""

from __future__ import annotations

import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from app.config import Settings
from app.llm.openrouter_client import OpenRouterClient


def _suffix_to_format(suffix: str) -> str:
    s = (suffix or "").lower().lstrip(".")
    mapping = {
        "wav": "wav",
        "mp3": "mp3",
        "flac": "flac",
        "aac": "aac",
        "ogg": "ogg",
        "m4a": "aac",
        "opus": "ogg",
    }
    return mapping.get(s, "wav")


def transcription_cache_key(path: Path) -> str:
    try:
        st = path.stat()
        raw = f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}".encode("utf-8", errors="replace")
    except OSError:
        raw = str(path).encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()


def load_transcription_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.items()}
    except (json.JSONDecodeError, OSError):
        return {}


def save_transcription_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def transcribe_audio_file(
    client: OpenRouterClient,
    path: Path,
    settings: Settings,
) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    try:
        raw = p.read_bytes()
    except OSError:
        return ""
    cap = max(8_192, int(settings.audio_transcription_max_bytes))
    raw = raw[:cap]
    if not raw:
        return ""
    fmt = _suffix_to_format(p.suffix)
    b64 = base64.b64encode(raw).decode("ascii")
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Transcribe all intelligible speech. Same language as the audio. "
                        "Plain text only, max 400 words. If no speech, reply exactly: [silence]"
                    ),
                },
                {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}},
            ],
        }
    ]
    try:
        text, _data = client.complete_chat(
            messages,
            model=settings.openrouter_model_transcription,
            max_tokens=600,
            json_mode=False,
        )
        return (text or "").strip()[:8000]
    except Exception:
        return ""


def transcribe_paths_parallel(
    client: OpenRouterClient,
    paths: list[Path],
    settings: Settings,
    disk_cache: dict[str, str],
    disk_cache_path: Path,
    max_workers: int,
) -> tuple[dict[str, str], int]:
    """Path string -> transcript, plus count of OpenRouter transcription calls (excl. cache hits)."""
    out: dict[str, str] = {}
    pending: list[tuple[str, Path]] = []
    for p in paths:
        key = transcription_cache_key(p)
        if key in disk_cache:
            out[str(p)] = disk_cache[key]
        else:
            pending.append((key, p))

    if not pending:
        return out, 0

    workers = max(1, min(max_workers, len(pending)))

    def _one(item: tuple[str, Path]) -> tuple[str, str, str]:
        ck, pth = item
        txt = transcribe_audio_file(client, pth, settings)
        return ck, str(pth), txt

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, item): item for item in pending}
        for fut in as_completed(futs):
            try:
                ck, pstr, txt = fut.result()
                disk_cache[ck] = txt
                out[pstr] = txt
            except Exception:
                pass

    save_transcription_cache(disk_cache_path, disk_cache)
    return out, len(pending)
