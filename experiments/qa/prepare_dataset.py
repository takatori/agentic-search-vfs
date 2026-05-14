from __future__ import annotations

import argparse
import json
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset


DEFAULT_OUTPUT_ROOT = Path("data_eval/qa")
DEFAULT_SPLIT = "validation"
DATASET_ORDER = ("jaquad", "jsquad")


def safe_path_component(value: str, fallback: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    normalized = re.sub(r"\s+", "-", normalized)
    normalized = re.sub(r'[\\/:*?"<>|#%{}^~`\[\]]+', "_", normalized)
    normalized = normalized.strip("._-")
    if not normalized:
        normalized = fallback
    return normalized[:80]


def normalize_answer_text(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).lower()


def iter_dataset_rows(dataset_name: str, split: str) -> Iterable[dict[str, Any]]:
    if dataset_name == "jaquad":
        dataset = load_dataset("SkelterLabsInc/JaQuAD", split=split)
    elif dataset_name == "jsquad":
        try:
            dataset = load_dataset("sbintuitions/JSQuAD", split=split)
        except Exception:
            dataset = load_dataset("shunk031/JGLUE", "JSQuAD", split=split)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    for row in dataset:
        if isinstance(row, dict):
            yield row


def extract_answers(value: Any) -> list[str]:
    if isinstance(value, dict):
        texts = value.get("text")
        if isinstance(texts, list):
            return [str(text).strip() for text in texts if str(text).strip()]
        if isinstance(texts, str) and texts.strip():
            return [texts.strip()]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                out.append(item["text"].strip())
            elif isinstance(item, str):
                out.append(item.strip())
        return [item for item in out if item]
    return []


def yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def make_document_content(dataset: str, title: str, context_id: str, context: str) -> str:
    return "\n".join(
        [
            "---",
            f"dataset: {yaml_string(dataset)}",
            f"title: {yaml_string(title)}",
            f"context_id: {yaml_string(context_id)}",
            "---",
            "",
            f"# {title}",
            "",
            context.strip(),
            "",
        ]
    )


def should_keep_case(
    row: dict[str, Any],
    context: str,
    answers: list[str],
    min_context_chars: int,
) -> bool:
    if row.get("is_impossible") is True:
        return False
    if len(context) < min_context_chars:
        return False
    normalized_context = normalize_answer_text(context)
    return any(normalize_answer_text(answer) in normalized_context for answer in answers)


def build_dataset(
    dataset_name: str,
    split: str,
    limit: int,
    min_context_chars: int,
    docs_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    path_tree: dict[str, dict[str, Any]] = {}

    for row in iter_dataset_rows(dataset_name, split):
        context = str(row.get("context") or "").strip()
        question = str(row.get("question") or "").strip()
        title = str(row.get("title") or "untitled").strip() or "untitled"
        row_id = str(row.get("id") or row.get("q_id") or f"{dataset_name}-{len(cases)}")
        answers = extract_answers(row.get("answers"))

        if not question or not answers:
            continue
        if not should_keep_case(row, context, answers, min_context_chars):
            continue

        title_component = safe_path_component(title, "untitled")
        id_component = safe_path_component(row_id, f"context-{len(cases):04d}")
        rel_path = Path("qa") / dataset_name / title_component / f"{id_component}.mdx"
        output_path = docs_root / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            make_document_content(dataset_name, title, row_id, context),
            encoding="utf-8",
        )

        slug = rel_path.as_posix()[: -len(".mdx")]
        expected_path = f"/{rel_path.as_posix()}"
        path_tree[slug] = {"isPublic": True, "groups": []}
        cases.append(
            {
                "id": row_id,
                "dataset": dataset_name,
                "question": question,
                "answers": answers,
                "expectedPath": expected_path,
                "title": title,
            }
        )
        if len(cases) >= limit:
            break

    if len(cases) < limit:
        print(
            f"Warning: requested {limit} {dataset_name} cases, selected {len(cases)} after filtering."
        )
    return cases, path_tree


def parse_dataset_names(value: str) -> list[str]:
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names) - set(DATASET_ORDER))
    if unknown:
        raise ValueError(f"Unsupported dataset names: {', '.join(unknown)}")
    return names or list(DATASET_ORDER)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare JaQuAD/JSQuAD as OpenSearchFs QA evaluation docs."
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--datasets", default=",".join(DATASET_ORDER))
    parser.add_argument("--limit-per-dataset", type=int, default=50)
    parser.add_argument("--min-context-chars", type=int, default=120)
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

    all_cases: list[dict[str, Any]] = []
    all_path_tree: dict[str, dict[str, Any]] = {}
    for dataset_name in parse_dataset_names(args.datasets):
        cases, path_tree = build_dataset(
            dataset_name,
            args.split,
            args.limit_per_dataset,
            args.min_context_chars,
            docs_root,
        )
        all_cases.extend(cases)
        all_path_tree.update(path_tree)

    path_tree_path = output_root / "path_tree.json"
    path_tree_path.write_text(
        json.dumps(all_path_tree, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    cases_path = output_root / "eval_cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as fp:
        for case in all_cases:
            fp.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(
        f"Prepared {len(all_cases)} QA cases, {len(all_path_tree)} documents under {docs_root}."
    )
    print(f"Wrote {path_tree_path} and {cases_path}.")


if __name__ == "__main__":
    main()
