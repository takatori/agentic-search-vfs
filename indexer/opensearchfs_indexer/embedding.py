from __future__ import annotations

import os
import re
from typing import Any

from . import config
from .types import FileRecord, TextChunk


class DocumentEmbedder:
    model_id: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


def create_document_embedder() -> DocumentEmbedder:
    provider = config.resolve_embedding_provider_name()
    if provider == "ruri":
        return RuriDocumentEmbedder()
    return NomicDocumentEmbedder()


class RuriDocumentEmbedder(DocumentEmbedder):
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        model_name = config.resolve_ruri_index_model_name()
        model_kwargs: dict[str, Any] = {}
        device = os.environ.get("RURI_INDEX_DEVICE")
        if device:
            model_kwargs["device"] = device
        self._batch_size = int(os.environ.get("RURI_BATCH_SIZE", "8"))
        max_seq_length = os.environ.get("RURI_MAX_SEQ_LENGTH")
        self.model_id = config.resolve_document_embedding_model_id("ruri")
        self._model = SentenceTransformer(model_name, **model_kwargs)
        if max_seq_length:
            self._model.max_seq_length = int(max_seq_length)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(
            [f"検索文書: {text}" for text in texts],
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        rows = embeddings.astype(float).tolist()
        for row in rows:
            if not isinstance(row, list) or len(row) != config.EMBEDDING_DIMS:
                raise ValueError(
                    f"Expected {config.EMBEDDING_DIMS}-dimensional Ruri embeddings."
                )
        return rows


class NomicDocumentEmbedder(DocumentEmbedder):
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        model_name = config.resolve_nomic_index_model_name()
        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        device = os.environ.get("NOMIC_INDEX_DEVICE")
        if device:
            model_kwargs["device"] = device
        self._batch_size = int(os.environ.get("NOMIC_BATCH_SIZE", "8"))
        max_seq_length = os.environ.get("NOMIC_MAX_SEQ_LENGTH")
        self.model_id = config.resolve_document_embedding_model_id("nomic")
        self._model = SentenceTransformer(model_name, **model_kwargs)
        if max_seq_length:
            self._model.max_seq_length = int(max_seq_length)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import torch.nn.functional as F

        embeddings = self._model.encode(
            [f"search_document: {text}" for text in texts],
            batch_size=self._batch_size,
            normalize_embeddings=False,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        embeddings = F.layer_norm(
            embeddings,
            normalized_shape=(embeddings.shape[1],),
        )
        embeddings = embeddings[:, : config.EMBEDDING_DIMS]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        rows = embeddings.detach().cpu().float().tolist()
        for row in rows:
            if not isinstance(row, list) or len(row) != config.EMBEDDING_DIMS:
                raise ValueError(
                    f"Expected {config.EMBEDDING_DIMS}-dimensional Nomic embeddings."
                )
        return rows


def split_text_into_chunks(content: str) -> list[TextChunk]:
    max_chars = config.resolve_embedding_chunk_max_chars()
    overlap_chars = config.resolve_embedding_chunk_overlap_chars()
    text = content.strip()
    if not text:
        return [TextChunk(chunk_id=0, text="")]
    chunks: list[TextChunk] = []
    start = 0
    while start < len(text):
        hard_end = min(start + max_chars, len(text))
        end = hard_end
        if hard_end < len(text):
            min_break = start + max(max_chars // 2, 1)
            break_at = max(
                text.rfind("\n\n", min_break, hard_end),
                text.rfind("\n", min_break, hard_end),
                text.rfind(" ", min_break, hard_end),
            )
            if break_at > start:
                end = break_at
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(TextChunk(chunk_id=len(chunks), text=chunk_text))
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return chunks or [TextChunk(chunk_id=0, text=text[:max_chars])]


def extract_document_title(content: str, slug: str) -> str:
    front_matter_match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    if front_matter_match:
        title_match = re.search(
            r'^title:\s*["\']?(.*?)["\']?\s*$',
            front_matter_match.group(1),
            re.MULTILINE,
        )
        if title_match:
            return title_match.group(1).strip()
    heading_match = re.search(r"^#\s+(.+?)\s*$", content, re.MULTILINE)
    if heading_match:
        return heading_match.group(1).strip()
    return slug.rsplit("/", 1)[-1]


def build_chunk_embedding_text(record: FileRecord, chunk: TextChunk) -> str:
    return (
        f"Document path: /{record.slug}.mdx\n"
        f"Title: {extract_document_title(record.content, record.slug)}\n\n"
        f"{chunk.text}"
    )
