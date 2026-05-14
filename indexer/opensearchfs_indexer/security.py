from __future__ import annotations

import json
import os
from typing import Any

from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError

from . import config
from .types import PathTreePolicy


def compile_access_plan(policy: PathTreePolicy) -> tuple[list[str], dict[str, list[str]]]:
    public_slugs: set[str] = set()
    group_slugs: dict[str, set[str]] = {}
    for slug, entry in policy.items():
        if entry["isPublic"]:
            public_slugs.add(slug)
        for group in entry["groups"]:
            group_slugs.setdefault(str(group), set()).add(slug)
    merged_group_slugs = {
        group: sorted(public_slugs | slugs) for group, slugs in group_slugs.items()
    }
    return sorted(public_slugs), merged_group_slugs


def build_read_only_index_permissions(
    slugs: list[str] | None,
) -> list[dict[str, Any]]:
    files_permission: dict[str, Any] = {
        "index_patterns": [config.FILES_INDEX],
        "allowed_actions": ["read"],
    }
    if slugs is not None:
        files_permission["dls"] = json.dumps(
            {"terms": {"slug": slugs}}, ensure_ascii=False
        )
    return [
        files_permission,
        {"index_patterns": [config.META_INDEX], "allowed_actions": ["read"]},
    ]


def get_existing_role(client: OpenSearch, role_name: str) -> dict[str, Any]:
    try:
        response = client.transport.perform_request(
            "GET", f"/_plugins/_security/api/roles/{role_name}"
        )
    except NotFoundError:
        return {}
    if isinstance(response, dict):
        role = response.get(role_name, response)
        if isinstance(role, dict):
            return role
    return {}


def permission_targets_current_indices(permission: dict[str, Any]) -> bool:
    patterns = permission.get("index_patterns")
    if not isinstance(patterns, list):
        return False
    return config.FILES_INDEX in patterns or config.META_INDEX in patterns


def merge_role_index_permissions(
    existing_role: dict[str, Any],
    current_permissions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing_permissions = existing_role.get("index_permissions")
    if not isinstance(existing_permissions, list):
        return current_permissions
    preserved = [
        permission
        for permission in existing_permissions
        if isinstance(permission, dict)
        and not permission_targets_current_indices(permission)
    ]
    return preserved + current_permissions


def setup_read_only_security_profiles(
    client: OpenSearch, policy: PathTreePolicy
) -> None:
    public_slugs, group_slugs = compile_access_plan(policy)
    all_slugs = sorted(policy)
    for profile in config.READ_PROFILES:
        role_name = f"opensearchfs_{profile.lower()}"
        username = os.environ.get(f"OPENSEARCH_USERNAME_{profile}") or profile.lower()
        password = (
            os.environ.get(f"OPENSEARCH_PASSWORD_{profile}")
            or config.DEFAULT_PROFILE_PASSWORDS[profile]
        )
        slugs = (
            public_slugs
            if profile == "PUBLIC"
            else group_slugs.get(profile.lower(), public_slugs)
        )
        dls_slugs = None if slugs == all_slugs else slugs
        existing_role = get_existing_role(client, role_name)
        current_index_permissions = build_read_only_index_permissions(dls_slugs)
        client.transport.perform_request(
            "PUT",
            f"/_plugins/_security/api/roles/{role_name}",
            body={
                "cluster_permissions": existing_role.get("cluster_permissions", []),
                "index_permissions": merge_role_index_permissions(
                    existing_role, current_index_permissions
                ),
                "tenant_permissions": existing_role.get("tenant_permissions", []),
            },
        )
        client.transport.perform_request(
            "PUT",
            f"/_plugins/_security/api/internalusers/{username}",
            body={"password": password, "backend_roles": [role_name], "attributes": {}},
        )
        client.transport.perform_request(
            "PUT",
            f"/_plugins/_security/api/rolesmapping/{role_name}",
            body={"users": [username], "backend_roles": [role_name], "hosts": []},
        )
        print(f'Configured OpenSearch DLS profile {profile} as user "{username}".')
