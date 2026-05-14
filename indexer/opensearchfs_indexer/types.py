from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FileRecord:
    slug: str
    content: str
    content_hash: str
    updated_at: str


@dataclass(frozen=True)
class ExistingFileState:
    content_hash: str | None
    embedding_model_id: str | None


@dataclass(frozen=True)
class TextChunk:
    chunk_id: int
    text: str


PathTreePolicy = dict[str, dict[str, Any]]
