from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from . import config
from .opensearch_store import (
    bulk_sync_files,
    create_client,
    ensure_index,
    fetch_existing_file_states,
    index_path_tree_document,
    recreate_index,
)
from .path_tree import (
    collect_files,
    effective_policy_for_records,
    load_path_tree_policy,
)
from .security import setup_read_only_security_profiles
from .types import ExistingFileState


def run(args: argparse.Namespace) -> None:
    load_dotenv()
    config.set_index_prefix(args.index_prefix)
    data_root = Path(args.data_root).resolve()
    path_tree_path = Path(args.path_tree).resolve()
    policy = load_path_tree_policy(path_tree_path)
    records = collect_files(data_root, policy)
    effective_policy = effective_policy_for_records(policy, records)

    client = create_client()
    try:
        if args.recreate:
            recreate_index(
                client, config.META_INDEX, config.load_mapping("meta-mapping.json")
            )
            recreate_index(
                client, config.FILES_INDEX, config.load_mapping("mappings.json")
            )
            existing_states: dict[str, ExistingFileState] = {}
        else:
            ensure_index(
                client, config.META_INDEX, config.load_mapping("meta-mapping.json")
            )
            ensure_index(
                client, config.FILES_INDEX, config.load_mapping("mappings.json")
            )
            existing_states = fetch_existing_file_states(client)

        changed, unchanged, deleted = bulk_sync_files(client, records, existing_states)
        index_path_tree_document(client, effective_policy)
        setup_read_only_security_profiles(client, effective_policy)
        print(
            "Bootstrap complete: "
            f"{len(records)} files, {changed} changed, {unchanged} unchanged, {deleted} deleted."
        )
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync documentation into OpenSearchFs.")
    parser.add_argument("--sync", action="store_true", help="Sync changed docs.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate OpenSearchFs indices before syncing.",
    )
    parser.add_argument("--data-root", default=config.DEFAULT_DATA_ROOT)
    parser.add_argument("--path-tree", default=config.DEFAULT_PATH_TREE)
    parser.add_argument(
        "--index-prefix",
        default=os.environ.get(
            "OPENSEARCHFS_INDEX_PREFIX", config.DEFAULT_INDEX_PREFIX
        ),
        help="OpenSearch index prefix. Files/meta indices are '<prefix>-chunks' and '<prefix>-meta'.",
    )
    args = parser.parse_args()
    if not args.sync and not args.recreate:
        args.sync = True
    run(args)
