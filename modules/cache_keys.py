"""Deterministic fingerprints for safe reuse of session-local results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def content_digest(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
        ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_fingerprint(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    if path.is_file():
        return content_digest(path.read_bytes())
    records = [
        (str(file.relative_to(path)), file.stat().st_size, file.stat().st_mtime_ns)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    ]
    return content_digest(records)


def stage_cache_key(
    stage: str,
    input_value: Any,
    *,
    versions: dict[str, Any],
    configuration: dict[str, Any] | None = None,
) -> str:
    """Hash the input and every implementation detail that can change a result."""
    return content_digest(
        {
            "stage": stage,
            "input_digest": content_digest(input_value),
            "versions": versions,
            "configuration": configuration or {},
        }
    )
