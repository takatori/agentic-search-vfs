from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


DEFAULT_INDEX_PREFIX = "opensearchfs"
FILES_INDEX = "opensearchfs-chunks"
META_INDEX = "opensearchfs-meta"
PATH_TREE_DOC_ID = "__path_tree__"
EMBEDDING_DIMS = 768
DEFAULT_DATA_ROOT = "./example_data"
DEFAULT_PATH_TREE = "./example_data/path_tree.json"
DEFAULT_EMBEDDING_PROVIDER = "ruri"
DEFAULT_RURI_INDEX_MODEL = "cl-nagoya/ruri-v3-310m"
DEFAULT_NOMIC_INDEX_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_BULK_BATCH_SIZE = 64
DEFAULT_EMBEDDING_CHUNK_MAX_CHARS = 3000
DEFAULT_EMBEDDING_CHUNK_OVERLAP_CHARS = 300
READ_PROFILES = ("PUBLIC", "BILLING", "INTERNAL")
DEFAULT_PROFILE_PASSWORDS = {
    "PUBLIC": "SearchFsReadA2026!",
    "BILLING": "SearchFsReadB2026!",
    "INTERNAL": "SearchFsReadC2026!",
}
REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_index_prefix(value: str | None) -> str:
    prefix = (value or DEFAULT_INDEX_PREFIX).strip() or DEFAULT_INDEX_PREFIX
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", prefix):
        raise ValueError(
            f'Invalid index prefix "{prefix}". Use lowercase letters, numbers, "_" or "-", starting with a letter or number.'
        )
    return prefix


def set_index_prefix(prefix: str) -> None:
    global FILES_INDEX, META_INDEX
    resolved = resolve_index_prefix(prefix)
    FILES_INDEX = f"{resolved}-chunks"
    META_INDEX = f"{resolved}-meta"


def load_mapping(file_name: str) -> dict[str, Any]:
    mapping_path = REPO_ROOT / "src" / "opensearch-adapter" / file_name
    parsed = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"Invalid {file_name}: expected object.")
    return parsed


def resolve_embedding_provider_name(value: str | None = None) -> str:
    raw = (
        value
        or os.environ.get("INDEX_EMBEDDING_PROVIDER")
        or os.environ.get("EMBEDDING_PROVIDER")
        or DEFAULT_EMBEDDING_PROVIDER
    )
    provider = raw.strip().lower()
    if provider not in {"ruri", "nomic"}:
        raise ValueError(
            f'Invalid embedding provider "{raw}". Expected one of: ruri, nomic.'
        )
    return provider


def resolve_ruri_index_model_name() -> str:
    return (
        os.environ.get("RURI_INDEX_MODEL")
        or os.environ.get("RURI_MODEL")
        or DEFAULT_RURI_INDEX_MODEL
    )


def resolve_nomic_index_model_name() -> str:
    return (
        os.environ.get("NOMIC_INDEX_MODEL")
        or os.environ.get("NOMIC_MODEL")
        or DEFAULT_NOMIC_INDEX_MODEL
    )


def parse_positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def parse_non_negative_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def resolve_embedding_chunk_max_chars() -> int:
    return parse_positive_int_env(
        "EMBEDDING_CHUNK_MAX_CHARS", DEFAULT_EMBEDDING_CHUNK_MAX_CHARS
    )


def resolve_embedding_chunk_overlap_chars() -> int:
    max_chars = resolve_embedding_chunk_max_chars()
    overlap = parse_non_negative_int_env(
        "EMBEDDING_CHUNK_OVERLAP_CHARS", DEFAULT_EMBEDDING_CHUNK_OVERLAP_CHARS
    )
    if overlap >= max_chars:
        raise ValueError(
            "EMBEDDING_CHUNK_OVERLAP_CHARS must be smaller than EMBEDDING_CHUNK_MAX_CHARS."
        )
    return overlap


def resolve_provider_max_seq_length(provider: str) -> str:
    if provider == "ruri":
        return os.environ.get("RURI_MAX_SEQ_LENGTH", "")
    return os.environ.get("NOMIC_MAX_SEQ_LENGTH", "")


def resolve_document_embedding_model_id(provider: str | None = None) -> str:
    resolved = resolve_embedding_provider_name(provider)
    chunk_max_chars = resolve_embedding_chunk_max_chars()
    chunk_overlap_chars = resolve_embedding_chunk_overlap_chars()
    max_seq_length = resolve_provider_max_seq_length(resolved)
    chunk_config = (
        f"chunk_chars={chunk_max_chars}:overlap={chunk_overlap_chars}:"
        f"max_seq={max_seq_length or 'model-default'}:context=slug-title"
    )
    if resolved == "ruri":
        return (
            f"ruri:{resolve_ruri_index_model_name()}:"
            f"dims={EMBEDDING_DIMS}:prefix=ruri-v3-ja:{chunk_config}"
        )
    return (
        f"nomic:{resolve_nomic_index_model_name()}:"
        f"dims={EMBEDDING_DIMS}:prefix=search_document:"
        f"pool=mean-layernorm-normalize:{chunk_config}"
    )
