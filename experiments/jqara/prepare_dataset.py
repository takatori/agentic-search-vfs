from __future__ import annotations

import argparse
import json
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any

from datasets import load_dataset


DEFAULT_OUTPUT_ROOT = Path("data_eval/jqara")
DEFAULT_SPLIT = "test"
DATASET_NAME = "hotchpotch/JQaRA"


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


def extract_answers(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def make_document_content(
    split: str,
    title: str,
    passage_row_id: str,
    text: str,
) -> str:
    return "\n".join(
        [
            "---",
            'dataset: "jqara"',
            f"split: {yaml_string(split)}",
            f"title: {yaml_string(title)}",
            f"passage_row_id: {yaml_string(passage_row_id)}",
            "---",
            "",
            f"# {title}",
            "",
            text.strip(),
            "",
        ]
    )


def select_qids(dataset: Any, limit_questions: int) -> set[str] | None:
    if limit_questions <= 0:
        return None
    qids: list[str] = []
    seen: set[str] = set()
    for row in dataset:
        qid = str(row.get("q_id") or "").strip()
        if not qid or qid in seen:
            continue
        seen.add(qid)
        qids.append(qid)
        if len(qids) >= limit_questions:
            break
    return set(qids)


def build_dataset(
    split: str,
    limit_questions: int,
    docs_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    dataset = load_dataset(DATASET_NAME, split=split)
    selected_qids = select_qids(dataset, limit_questions)

    path_tree: dict[str, dict[str, Any]] = {}
    passage_paths: dict[str, str] = {}
    cases: dict[str, dict[str, Any]] = {}

    for row in dataset:
        qid = str(row.get("q_id") or "").strip()
        if not qid:
            continue
        if selected_qids is not None and qid not in selected_qids:
            continue

        passage_row_id = str(row.get("passage_row_id") or "").strip()
        question = str(row.get("question") or "").strip()
        title = str(row.get("title") or "untitled").strip() or "untitled"
        text = str(row.get("text") or "").strip()
        answers = extract_answers(row.get("answers"))
        if not passage_row_id or not question or not text or not answers:
            continue

        rel_path = passage_paths.get(passage_row_id)
        if rel_path is None:
            title_component = safe_path_component(title, "untitled")
            passage_component = safe_path_component(passage_row_id, "passage")
            path = Path("jqara") / split / title_component / f"{passage_component}.mdx"
            output_path = docs_root / path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                make_document_content(split, title, passage_row_id, text),
                encoding="utf-8",
            )
            rel_path = path.as_posix()
            passage_paths[passage_row_id] = rel_path
            slug = rel_path[: -len(".mdx")]
            path_tree[slug] = {"isPublic": True, "groups": []}

        case = cases.setdefault(
            qid,
            {
                "id": qid,
                "q_id": qid,
                "dataset": "jqara",
                "question": question,
                "answers": answers,
                "positivePaths": [],
            },
        )
        if int(row.get("label") or 0) == 1:
            expected_path = f"/{rel_path}"
            if expected_path not in case["positivePaths"]:
                case["positivePaths"].append(expected_path)

    out_cases = [
        case for case in cases.values() if len(case.get("positivePaths", [])) > 0
    ]
    out_cases.sort(key=lambda item: item["q_id"])
    return out_cases, path_tree


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare hotchpotch/JQaRA as OpenSearchFs evaluation docs."
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--limit-questions",
        type=int,
        default=0,
        help="Limit selected q_id values. 0 means all questions in the split.",
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

    cases, path_tree = build_dataset(
        split=str(args.split),
        limit_questions=int(args.limit_questions),
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

    print(
        f"Prepared {len(cases)} JQaRA questions and {len(path_tree)} passages under {docs_root}."
    )
    print(f"Wrote {path_tree_path} and {cases_path}.")


if __name__ == "__main__":
    main()
