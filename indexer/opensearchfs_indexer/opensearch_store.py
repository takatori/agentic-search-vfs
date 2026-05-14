from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from typing import Any

from opensearchpy import OpenSearch

from . import config
from .embedding import (
    build_chunk_embedding_text,
    create_document_embedder,
    split_text_into_chunks,
)
from .types import ExistingFileState, FileRecord, PathTreePolicy


def create_client() -> OpenSearch:
    url = os.environ.get("OPENSEARCH_URL")
    username = os.environ.get("OPENSEARCH_USERNAME_SYSTEM")
    password = os.environ.get("OPENSEARCH_PASSWORD_SYSTEM")
    if not url:
        raise ValueError("Missing OPENSEARCH_URL")
    if not username or not password:
        raise ValueError(
            "Missing OPENSEARCH_USERNAME_SYSTEM or OPENSEARCH_PASSWORD_SYSTEM."
        )
    return OpenSearch(
        hosts=[url],
        http_auth=(username, password),
        verify_certs=False,
        ssl_show_warn=False,
    )


def recreate_index(client: OpenSearch, index: str, definition: dict[str, Any]) -> None:
    if client.indices.exists(index=index):
        client.indices.delete(index=index)
        print(f'Deleted index "{index}".')
    client.indices.create(index=index, body=definition)
    print(f'Created index "{index}".')


def ensure_index(client: OpenSearch, index: str, definition: dict[str, Any]) -> None:
    if not client.indices.exists(index=index):
        client.indices.create(index=index, body=definition)
        print(f'Created index "{index}".')
        return
    mappings = definition.get("mappings")
    if isinstance(mappings, dict):
        client.indices.put_mapping(index=index, body=mappings)


def fetch_existing_file_states(client: OpenSearch) -> dict[str, ExistingFileState]:
    if not client.indices.exists(index=config.FILES_INDEX):
        return {}

    body: dict[str, Any] = {
        "size": 1000,
        "_source": ["slug", "content_hash", "embedding_model_id"],
        "sort": [{"slug": {"order": "asc"}}],
        "query": {"match_all": {}},
    }
    out: dict[str, ExistingFileState] = {}
    while True:
        res = client.search(index=config.FILES_INDEX, body=body)
        hits = res.get("hits", {}).get("hits", [])
        if not hits:
            break
        for hit in hits:
            source = hit.get("_source", {})
            slug = source.get("slug")
            if isinstance(slug, str) and slug:
                content_hash = source.get("content_hash")
                embedding_model_id = source.get("embedding_model_id")
                out[slug] = ExistingFileState(
                    content_hash=content_hash if isinstance(content_hash, str) else None,
                    embedding_model_id=embedding_model_id
                    if isinstance(embedding_model_id, str)
                    else None,
                )
        body["search_after"] = hits[-1].get("sort")
    return out


def bulk_sync_files(
    client: OpenSearch,
    records: list[FileRecord],
    existing_states: dict[str, ExistingFileState],
) -> tuple[int, int, int]:
    by_slug = {record.slug: record for record in records}
    embedding_model_id = config.resolve_document_embedding_model_id()
    changed = [
        record
        for record in records
        if (state := existing_states.get(record.slug)) is None
        or state.content_hash != record.content_hash
        or state.embedding_model_id != embedding_model_id
    ]
    deleted = sorted(set(existing_states) - set(by_slug))
    bulk_batch_size = int(
        os.environ.get(
            "OPENSEARCH_BULK_BATCH_SIZE", str(config.DEFAULT_BULK_BATCH_SIZE)
        )
    )
    if bulk_batch_size <= 0:
        raise ValueError("OPENSEARCH_BULK_BATCH_SIZE must be positive.")

    if changed:
        embedder = create_document_embedder()
        if embedder.model_id != embedding_model_id:
            raise ValueError(
                f"Embedding model id mismatch: expected {embedding_model_id}, got {embedder.model_id}."
            )
        print(f"Using document embedding model: {embedding_model_id}", flush=True)
        for start in range(0, len(changed), bulk_batch_size):
            batch = changed[start : start + bulk_batch_size]
            chunked_records = [
                (record, split_text_into_chunks(record.content)) for record in batch
            ]
            chunk_texts = [
                build_chunk_embedding_text(record, chunk)
                for record, chunks in chunked_records
                for chunk in chunks
            ]
            vectors = embedder.embed_documents(chunk_texts)
            vector_index = 0
            operations: list[dict[str, Any]] = []
            for record, chunks in chunked_records:
                indexed_chunks: list[dict[str, Any]] = []
                for chunk in chunks:
                    vector = vectors[vector_index]
                    vector_index += 1
                    indexed_chunks.append(
                        {
                            "chunk_id": chunk.chunk_id,
                            "text": chunk.text,
                            "embedding": vector,
                        }
                    )
                operations.append(
                    {"index": {"_index": config.FILES_INDEX, "_id": record.slug}}
                )
                operations.append(
                    {
                        "slug": record.slug,
                        "content": record.content,
                        "content_hash": record.content_hash,
                        "embedding_model_id": embedding_model_id,
                        "chunk_count": len(indexed_chunks),
                        "chunks": indexed_chunks,
                        "updated_at": record.updated_at,
                    }
                )
            if vector_index != len(vectors):
                raise ValueError("Chunk embedding count did not match indexed chunks.")
            bulk_write(client, operations)
            print(
                f"Indexed {min(start + len(batch), len(changed))}/{len(changed)} changed files.",
                flush=True,
            )

    for start in range(0, len(deleted), bulk_batch_size):
        batch = deleted[start : start + bulk_batch_size]
        operations = [
            {"delete": {"_index": config.FILES_INDEX, "_id": slug}}
            for slug in batch
        ]
        bulk_write(client, operations)

    if changed or deleted:
        client.indices.refresh(index=config.FILES_INDEX)

    unchanged = len(records) - len(changed)
    return len(changed), unchanged, len(deleted)


def bulk_write(client: OpenSearch, operations: list[dict[str, Any]]) -> None:
    if not operations:
        return
    response = client.bulk(body=operations, refresh=False)
    if response.get("errors"):
        first_error = next(
            (
                item
                for item in response.get("items", [])
                if any(value.get("error") for value in item.values())
            ),
            response,
        )
        raise RuntimeError(f"Bulk sync reported errors: {first_error}")


def index_path_tree_document(client: OpenSearch, policy: PathTreePolicy) -> None:
    now = datetime.now(timezone.utc).isoformat()
    payload = base64.b64encode(
        json.dumps(policy, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    client.index(
        index=config.META_INDEX,
        id=config.PATH_TREE_DOC_ID,
        refresh=True,
        body={
            "doc_type": config.PATH_TREE_DOC_ID,
            "tree_version": now,
            "payload": payload,
            "created_at": now,
            "updated_at": now,
        },
    )
    print(f'Indexed path tree doc "{config.PATH_TREE_DOC_ID}" in "{config.META_INDEX}".')
