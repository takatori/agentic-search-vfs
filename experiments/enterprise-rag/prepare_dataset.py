from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset


DEFAULT_OUTPUT_ROOT = Path("data_eval/enterprise_rag")
DEFAULT_SPLIT = "test"
DATASET_NAME = "onyx-dot-app/EnterpriseRAG-Bench"


def safe_path_component(value: str, fallback: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    normalized = re.sub(r"\s+", "-", normalized)
    normalized = re.sub(r"[^\w.-]+", "_", normalized, flags=re.UNICODE)
    normalized = normalized.strip("._-").lower()
    if not normalized:
        normalized = fallback
    return normalized[:80]


def yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def make_document_content(
    split: str,
    source_type: str,
    title: str,
    doc_id: str,
    content: str,
) -> str:
    return "\n".join(
        [
            "---",
            'dataset: "enterprise-rag-bench"',
            f"split: {yaml_string(split)}",
            f"source_type: {yaml_string(source_type)}",
            f"title: {yaml_string(title)}",
            f"doc_id: {yaml_string(doc_id)}",
            "---",
            "",
            f"# {title}",
            "",
            content.strip(),
            "",
        ]
    )


def stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def question_type(row: dict[str, Any]) -> str:
    return str(row.get("question_type") or "unknown").strip() or "unknown"


def source_type(row: dict[str, Any]) -> str:
    return str(row.get("source_type") or "unknown").strip() or "unknown"


def expected_doc_ids(row: dict[str, Any]) -> list[str]:
    return string_list(row.get("expected_doc_ids"))


def is_answerable_question(row: dict[str, Any]) -> bool:
    return len(expected_doc_ids(row)) > 0


def stratified_sample(
    rows: list[dict[str, Any]],
    n: int,
    key_name: str,
    seed: int,
) -> list[dict[str, Any]]:
    if n <= 0 or n >= len(rows):
        return sorted(rows, key=lambda row: str(row.get("question_id") or ""))

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key_name) or "unknown")].append(row)

    quotas = allocate_proportional_quotas(
        {key: len(value) for key, value in groups.items()},
        n,
    )
    sampled: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = list(groups[key])
        rng = random.Random(stable_seed(seed, key))
        rng.shuffle(group)
        sampled.extend(group[: quotas.get(key, 0)])

    return sorted(sampled, key=lambda row: str(row.get("question_id") or ""))


def allocate_proportional_quotas(counts: dict[str, int], n: int) -> dict[str, int]:
    positive_counts = {key: value for key, value in counts.items() if value > 0}
    total = sum(positive_counts.values())
    if n <= 0 or total == 0:
        return {key: 0 for key in counts}
    target = min(n, total)
    raw = {
        key: (value * target / total)
        for key, value in positive_counts.items()
    }
    quotas = {
        key: min(positive_counts[key], int(value))
        for key, value in raw.items()
    }

    # Give every represented stratum at least one slot when possible.
    remaining = target - sum(quotas.values())
    if target >= len(positive_counts):
        for key in sorted(positive_counts):
            if remaining <= 0:
                break
            if quotas[key] == 0:
                quotas[key] = 1
                remaining -= 1

    while remaining > 0:
        candidates = [
            key
            for key in positive_counts
            if quotas[key] < positive_counts[key]
        ]
        if not candidates:
            break
        key = max(
            candidates,
            key=lambda item: (raw[item] - quotas[item], positive_counts[item], item),
        )
        quotas[key] += 1
        remaining -= 1

    return {key: quotas.get(key, 0) for key in counts}


def selected_questions(
    split: str,
    limit_questions: int,
    sample_questions: int,
    seed: int,
    answerable_only: bool,
) -> list[dict[str, Any]]:
    dataset = load_dataset(DATASET_NAME, "questions", split=split)
    rows = [
        row
        for row in dataset
        if isinstance(row, dict) and (not answerable_only or is_answerable_question(row))
    ]
    if sample_questions > 0:
        return stratified_sample(
            rows,
            sample_questions,
            key_name="question_type",
            seed=seed,
        )

    selected_rows: list[dict[str, Any]] = []
    for row in dataset:
        if isinstance(row, dict) and (not answerable_only or is_answerable_question(row)):
            selected_rows.append(row)
        if limit_questions > 0 and len(selected_rows) >= limit_questions:
            break
    return selected_rows


def iter_documents(split: str, scan_limit: int = 0) -> Iterable[dict[str, Any]]:
    dataset = load_dataset(DATASET_NAME, "documents", split=split, streaming=True)
    scanned = 0
    for row in dataset:
        if isinstance(row, dict):
            yield row
            scanned += 1
            if scan_limit > 0 and scanned >= scan_limit:
                break


def is_valid_document(row: dict[str, Any]) -> bool:
    doc_id = str(row.get("doc_id") or "").strip()
    content = str(row.get("content") or "").strip()
    return bool(doc_id and content)


def doc_id_for(row: dict[str, Any]) -> str:
    return str(row.get("doc_id") or "").strip()


def rel_path_for_document(split: str, row: dict[str, Any]) -> str:
    doc_id = str(row.get("doc_id") or "").strip()
    source_type = str(row.get("source_type") or "unknown").strip() or "unknown"
    title = str(row.get("title") or "untitled").strip() or "untitled"
    source_component = safe_path_component(source_type, "unknown")
    title_component = safe_path_component(title, "untitled")
    doc_component = safe_path_component(doc_id, "doc")
    return (
        Path("enterprise-rag") / split / source_component / title_component / f"{doc_component}.mdx"
    ).as_posix()


def reservoir_sample_optional_doc_ids(
    split: str,
    required_doc_ids: set[str],
    target: int,
    seed: int,
    document_scan_limit: int,
) -> set[str]:
    if target <= 0:
        return set()
    reservoir: list[str] = []
    seen = 0
    rng = random.Random(stable_seed(seed, "docs"))

    for row in iter_documents(split, document_scan_limit):
        if not is_valid_document(row):
            continue
        doc_id = doc_id_for(row)
        if doc_id in required_doc_ids:
            continue
        seen += 1
        if len(reservoir) < target:
            reservoir.append(doc_id)
            continue
        replace_at = rng.randrange(seen)
        if replace_at < target:
            reservoir[replace_at] = doc_id

    return set(reservoir)


def first_n_optional_doc_ids(
    split: str,
    required_doc_ids: set[str],
    limit_documents: int,
    document_scan_limit: int,
) -> set[str]:
    selected: set[str] = set()
    if limit_documents <= 0:
        return selected
    for row in iter_documents(split, document_scan_limit):
        if not is_valid_document(row):
            continue
        doc_id = doc_id_for(row)
        if doc_id in required_doc_ids:
            continue
        selected.add(doc_id)
        if len(selected) >= limit_documents:
            break
    return selected


def select_optional_doc_ids(
    split: str,
    required_doc_ids: set[str],
    limit_documents: int,
    sample_documents: int,
    seed: int,
    document_scan_limit: int,
) -> set[str] | None:
    if sample_documents <= 0:
        if limit_documents <= 0:
            return None
        return first_n_optional_doc_ids(
            split,
            required_doc_ids,
            limit_documents,
            document_scan_limit,
        )

    optional_target = max(sample_documents - len(required_doc_ids), 0)
    if optional_target == 0:
        return set()

    return reservoir_sample_optional_doc_ids(
        split,
        required_doc_ids,
        optional_target,
        seed,
        document_scan_limit,
    )


def write_selected_documents(
    split: str,
    docs_root: Path,
    selected_doc_ids: set[str] | None,
) -> tuple[dict[str, str], dict[str, dict[str, Any]], set[str]]:
    path_tree: dict[str, dict[str, Any]] = {}
    doc_paths: dict[str, str] = {}
    remaining = set(selected_doc_ids) if selected_doc_ids is not None else set()

    for row in iter_documents(split):
        if not is_valid_document(row):
            continue
        doc_id = doc_id_for(row)
        if selected_doc_ids is not None and doc_id not in remaining:
            continue
        rel_path = rel_path_for_document(split, row)
        output_path = docs_root / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            make_document_content(
                split,
                source_type(row),
                str(row.get("title") or "untitled").strip() or "untitled",
                doc_id,
                str(row.get("content") or "").strip(),
            ),
            encoding="utf-8",
        )
        doc_paths[doc_id] = f"/{rel_path}"
        path_tree[rel_path[: -len(".mdx")]] = {"isPublic": True, "groups": []}
        if selected_doc_ids is not None:
            remaining.remove(doc_id)
        if selected_doc_ids is not None and not remaining:
            break

    return doc_paths, path_tree, remaining


def build_dataset(
    split: str,
    limit_questions: int,
    limit_documents: int,
    sample_questions: int,
    qa_sample_questions: int,
    sample_documents: int,
    seed: int,
    answerable_only: bool,
    document_scan_limit: int,
    docs_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    questions = selected_questions(
        split,
        limit_questions,
        sample_questions,
        seed,
        answerable_only,
    )
    required_doc_ids = {
        doc_id
        for row in questions
        for doc_id in expected_doc_ids(row)
    }
    optional_doc_ids = select_optional_doc_ids(
        split,
        required_doc_ids,
        limit_documents,
        sample_documents,
        seed,
        document_scan_limit,
    )
    selected_doc_ids = (
        None if optional_doc_ids is None else required_doc_ids | optional_doc_ids
    )

    doc_paths, path_tree, missing_selected = write_selected_documents(
        split,
        docs_root,
        selected_doc_ids,
    )
    missing_required = required_doc_ids & missing_selected

    if missing_required:
        print(
            "Warning: missing expected documents: "
            + ", ".join(sorted(missing_required))
        )

    cases: list[dict[str, Any]] = []
    for row in questions:
        question_id = str(row.get("question_id") or "").strip()
        question = str(row.get("question") or "").strip()
        if not question_id or not question:
            continue
        expected_ids = expected_doc_ids(row)
        positive_paths = [
            doc_paths[doc_id] for doc_id in expected_ids if doc_id in doc_paths
        ]
        cases.append(
            {
                "id": question_id,
                "question_id": question_id,
                "dataset": "enterprise-rag-bench",
                "questionType": str(row.get("question_type") or "").strip(),
                "sourceTypes": string_list(row.get("source_types")),
                "question": question,
                "goldAnswer": str(row.get("gold_answer") or "").strip(),
                "answerFacts": string_list(row.get("answer_facts")),
                "expectedDocIds": expected_ids,
                "positivePaths": positive_paths,
            }
        )

    qa_cases = (
        stratified_sample(cases, qa_sample_questions, "questionType", seed + 101)
        if qa_sample_questions > 0
        else []
    )
    return cases, qa_cases, path_tree


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Onyx EnterpriseRAG-Bench as OpenSearchFs evaluation docs."
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--limit-questions",
        type=int,
        default=0,
        help="Limit selected questions. 0 means all questions in the split.",
    )
    parser.add_argument(
        "--limit-documents",
        type=int,
        default=0,
        help="Limit non-ground-truth documents. Ground-truth documents are always included. 0 means all documents.",
    )
    parser.add_argument(
        "--sample-questions",
        type=int,
        default=0,
        help="Stratified sample size for retrieval questions. 0 keeps legacy --limit-questions behavior.",
    )
    parser.add_argument(
        "--qa-sample-questions",
        type=int,
        default=0,
        help="Stratified subset size for Agent QA cases, selected from retrieval cases.",
    )
    parser.add_argument(
        "--sample-documents",
        type=int,
        default=0,
        help="Total sampled corpus size including ground-truth docs. 0 keeps legacy --limit-documents behavior.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--document-scan-limit",
        type=int,
        default=0,
        help="Limit document rows scanned for optional document sampling. 0 scans the full split.",
    )
    parser.add_argument(
        "--answerable-only",
        action="store_true",
        help="Exclude questions without expected_doc_ids.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete the output root before writing generated files.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    docs_root = output_root / "docs"
    if output_root.exists() and not args.keep_existing:
        shutil.rmtree(output_root)
    docs_root.mkdir(parents=True, exist_ok=True)

    cases, qa_cases, path_tree = build_dataset(
        split=str(args.split),
        limit_questions=int(args.limit_questions),
        limit_documents=int(args.limit_documents),
        sample_questions=int(args.sample_questions),
        qa_sample_questions=int(args.qa_sample_questions),
        sample_documents=int(args.sample_documents),
        seed=int(args.seed),
        answerable_only=bool(args.answerable_only),
        document_scan_limit=int(args.document_scan_limit),
        docs_root=docs_root,
    )

    path_tree_path = output_root / "path_tree.json"
    path_tree_path.write_text(
        json.dumps(path_tree, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    cases_path = output_root / "eval_cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as fp:
        for case in cases:
            fp.write(json.dumps(case, ensure_ascii=False) + "\n")

    qa_cases_path = output_root / "eval_cases_qa50.jsonl"
    if qa_cases:
        with qa_cases_path.open("w", encoding="utf-8") as fp:
            for case in qa_cases:
                fp.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(
        f"Prepared {len(cases)} EnterpriseRAG-Bench questions and {len(path_tree)} documents under {docs_root}."
    )
    print(f"Wrote {path_tree_path} and {cases_path}.")
    if qa_cases:
        print(f"Wrote {qa_cases_path}.")


if __name__ == "__main__":
    main()
