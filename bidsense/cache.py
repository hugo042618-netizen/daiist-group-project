"""
A small disk cache for Gemini calls: one JSON file per call, named by the hash of what
defines the call (model, prompt version, text). Never stores the input text itself.
"""

import hashlib
import json
from pathlib import Path

import config


def key(*parts) -> str:
    return hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:32]


def _path(kind: str, cache_key: str) -> Path:
    return config.CACHE_DIR / kind / f"{cache_key}.json"


def get(kind: str, cache_key: str) -> dict | None:
    path = _path(kind, cache_key)
    if not config.USE_CACHE or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def put(kind: str, cache_key: str, value: dict) -> None:
    if not config.USE_CACHE:
        return
    path = _path(kind, cache_key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass  # a read-only disk must not break a request
