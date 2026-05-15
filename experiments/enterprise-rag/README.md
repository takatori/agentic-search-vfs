# EnterpriseRAG-Bench Eval

This experiment prepares Onyx EnterpriseRAG-Bench as an enterprise-like
filesystem with source type and document title hierarchy, then evaluates
retrieval and Ollama agent behavior over `docs_bash`.

## Standard Subset Commands

Prepare the dataset and run retrieval:

```bash
npm run prepare:enterprise-rag-dataset -- --split test --limit-questions 10 --limit-documents 1000
npm run bootstrap:enterprise-rag
npm run eval:enterprise-rag-retrieval -- --limit 10
```

Run the Ollama agent comparison:

```bash
npm run eval:ollama-enterprise-rag -- --limit 5
```

## Fixed 20k Nomic Subset

For the blog-sized Nomic subset experiment, use the fixed 20k corpus / 100
retrieval questions / 50 Agent QA questions setup:

```bash
npm run prepare:enterprise-rag-dataset:20k
npm run bootstrap:enterprise-rag:20k
npm run eval:enterprise-rag-retrieval:20k
npm run eval:ollama-enterprise-rag:20k
```

This is a deterministic subset experiment (`seed=42`), not a full
EnterpriseRAG-Bench score.

## Data Layout

EnterpriseRAG-Bench docs live under this VFS path:

```text
/enterprise-rag/<split>/<source-type>/<title>/<doc-id>.mdx
```

Ground-truth documents referenced by selected questions are always included even
when `--limit-documents` is set. The generated source data is written under
`data_eval/enterprise_rag/` or `data_eval/enterprise_rag_20k/`.

## Reports

The retrieval report compares `lexical` and `semantic_search`. The agent report
compares `grep-only` against `grep+semantic_search`.

Default standard-subset reports are written to:

- `reports/enterprise-rag-retrieval-eval.jsonl`
- `reports/enterprise-rag-retrieval-eval.md`
- `reports/ollama-enterprise-rag-agent-eval.jsonl`
- `reports/ollama-enterprise-rag-agent-eval.md`

The fixed 20k scripts write to the explicit report paths defined in
`package.json`.

## Index Prefix

Standard scripts use isolated `opensearchfs-enterprise-rag-*` indices. The fixed
20k scripts use `opensearchfs-enterprise-rag-nomic-20k-*` indices. These do not
overwrite the normal docs, QA, or JQaRA indices.
