"""Small, dependency-free progress reporting helpers for long-running commands."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def emit(event: str, **fields: Any) -> None:
    """Write one machine-readable progress event and flush it immediately."""
    print(
        "[scdfc] " + json.dumps({"event": event, **fields}, ensure_ascii=False),
        flush=True,
    )


def append_jsonl(path: str | Path, event: str, **fields: Any) -> dict[str, Any]:
    """Append a JSONL event, then mirror it to stdout for live monitoring."""
    record = {"event": event, **fields}
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    emit(event, **fields)
    return record
