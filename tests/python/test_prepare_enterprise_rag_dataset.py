from __future__ import annotations

import tempfile
import unittest
import importlib.util
from pathlib import Path
from unittest.mock import patch

PREPARE_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "enterprise-rag"
    / "prepare_dataset.py"
)
spec = importlib.util.spec_from_file_location(
    "enterprise_rag_prepare_dataset",
    PREPARE_SCRIPT,
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load {PREPARE_SCRIPT}")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


class EnterpriseRagPrepareTest(unittest.TestCase):
    def test_stratified_sample_is_deterministic(self) -> None:
        rows = [
            {"question_id": f"b{i}", "question_type": "basic"} for i in range(10)
        ] + [
            {"question_id": f"s{i}", "question_type": "semantic"} for i in range(10)
        ]

        first = prep.stratified_sample(rows, 6, "question_type", 42)
        second = prep.stratified_sample(rows, 6, "question_type", 42)

        self.assertEqual(first, second)
        self.assertEqual(
            {"basic": 3, "semantic": 3},
            self._counts_by_type(first),
        )

    def test_build_dataset_includes_ground_truth_and_qa_subset(self) -> None:
        questions = [
            self._question("q1", "basic", ["d1"]),
            self._question("q2", "basic", ["d2"]),
            self._question("q3", "semantic", ["d3"]),
            self._question("q4", "semantic", ["d4"]),
        ]
        docs = [
            self._doc("d1", "github"),
            self._doc("d2", "github"),
            self._doc("d3", "slack"),
            self._doc("d4", "slack"),
            self._doc("d5", "confluence"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(prep, "selected_questions", return_value=questions):
                with patch.object(
                    prep,
                    "iter_documents",
                    lambda _split, _scan_limit=0: iter(docs),
                ):
                    cases, qa_cases, path_tree = prep.build_dataset(
                        split="test",
                        limit_questions=0,
                        limit_documents=0,
                        sample_questions=4,
                        qa_sample_questions=2,
                        sample_documents=5,
                        seed=42,
                        answerable_only=True,
                        document_scan_limit=0,
                        docs_root=Path(tmpdir),
                    )

        self.assertEqual(4, len(cases))
        self.assertEqual(2, len(qa_cases))
        self.assertTrue(set(case["id"] for case in qa_cases).issubset(
            set(case["id"] for case in cases),
        ))
        for case in cases:
            self.assertGreater(len(case["expectedDocIds"]), 0)
            self.assertGreater(len(case["positivePaths"]), 0)
            for path in case["positivePaths"]:
                self.assertIn(path.lstrip("/")[: -len(".mdx")], path_tree)

    def _counts_by_type(self, rows: list[dict[str, str]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["question_type"]] = counts.get(row["question_type"], 0) + 1
        return counts

    def _question(
        self,
        question_id: str,
        question_type: str,
        expected_doc_ids: list[str],
    ) -> dict[str, object]:
        return {
            "question_id": question_id,
            "question_type": question_type,
            "source_types": ["github"],
            "question": f"What about {question_id}?",
            "gold_answer": "answer",
            "answer_facts": ["answer"],
            "expected_doc_ids": expected_doc_ids,
        }

    def _doc(self, doc_id: str, source_type: str) -> dict[str, str]:
        return {
            "doc_id": doc_id,
            "source_type": source_type,
            "title": f"Document {doc_id}",
            "content": f"Content for {doc_id}",
        }


if __name__ == "__main__":
    unittest.main()
