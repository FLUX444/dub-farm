from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, BaseModel):
        payload = data.model_dump()
    elif isinstance(data, list) and data and isinstance(data[0], BaseModel):
        payload = [item.model_dump() for item in data]
    else:
        payload = data
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path, model: type[T] | None = None) -> Any:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if model is None:
        return raw
    if isinstance(raw, list):
        return [model.model_validate(item) for item in raw]
    return model.model_validate(raw)
