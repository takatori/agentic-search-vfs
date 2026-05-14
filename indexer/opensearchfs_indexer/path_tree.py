from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import FileRecord, PathTreePolicy


def normalize_slug(slug: str) -> str:
    normalized = re.sub(r"/+", "/", slug.replace("\\", "/"))
    normalized = re.sub(r"^\./", "", normalized)
    normalized = re.sub(r"/+$", "", normalized)
    if not normalized:
        raise ValueError("Invalid slug: empty string.")
    return normalized[1:] if normalized.startswith("/") else normalized


def path_to_slug(file_path: str) -> str:
    normalized = "/" + normalize_slug(file_path)
    if not normalized.endswith(".mdx"):
        raise ValueError(f'Path must end with .mdx: "{file_path}"')
    return normalized[1 : -len(".mdx")]


def parse_json_with_trailing_commas(raw: str) -> Any:
    return json.loads(re.sub(r",\s*([}\]])", r"\1", raw))


def load_path_tree_policy(path: Path) -> PathTreePolicy:
    parsed = parse_json_with_trailing_commas(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("Path tree policy must be an object.")

    out: PathTreePolicy = {}
    for raw_slug, value in parsed.items():
        slug = normalize_slug(str(raw_slug))
        if not isinstance(value, dict):
            raise ValueError(f'Invalid path tree entry for "{slug}": expected object.')
        is_public = value.get("isPublic")
        groups = value.get("groups")
        if not isinstance(is_public, bool):
            raise ValueError(
                f'Invalid path tree entry for "{slug}": "isPublic" must be boolean.'
            )
        if not isinstance(groups, list) or not all(
            isinstance(group, str) for group in groups
        ):
            raise ValueError(
                f'Invalid path tree entry for "{slug}": "groups" must be string[].'
            )
        clean_groups = sorted({group.strip() for group in groups if group.strip()})
        out[slug] = {"isPublic": is_public, "groups": clean_groups}
    return out


def collect_files(data_root: Path, policy: PathTreePolicy) -> list[FileRecord]:
    files = sorted(path for path in data_root.rglob("*.mdx") if path.is_file())
    if not files:
        raise ValueError(f"No ingestible files found under {data_root}.")

    records: list[FileRecord] = []
    missing_policy: list[str] = []
    for file_path in files:
        rel = file_path.relative_to(data_root).as_posix()
        slug = path_to_slug(rel)
        if slug not in policy:
            missing_policy.append(slug)
            continue
        content = file_path.read_text(encoding="utf-8")
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        updated_at = datetime.fromtimestamp(
            file_path.stat().st_mtime, tz=timezone.utc
        ).isoformat()
        records.append(
            FileRecord(
                slug=slug,
                content=content,
                content_hash=content_hash,
                updated_at=updated_at,
            )
        )

    if missing_policy:
        joined = ", ".join(sorted(missing_policy))
        raise ValueError(f"Missing path_tree.json ACL entries for slugs: {joined}")
    return records


def effective_policy_for_records(
    policy: PathTreePolicy, records: list[FileRecord]
) -> PathTreePolicy:
    record_slugs = {record.slug for record in records}
    extra_policy_slugs = sorted(set(policy) - record_slugs)
    if extra_policy_slugs:
        print(
            "Ignoring path_tree.json entries without source files: "
            + ", ".join(extra_policy_slugs)
        )
    return {slug: policy[slug] for slug in sorted(record_slugs)}
