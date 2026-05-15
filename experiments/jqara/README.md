# JQaRA Eval

This experiment prepares JQaRA passages as an AgenticSearchVfs tree, evaluates
full-passage retrieval, and compares an Ollama agent with and without the
explicit `semantic_search` command.

## Commands

Prepare the dataset and run retrieval:

```bash
npm run prepare:jqara-dataset -- --split test --limit-questions 20
npm run bootstrap:jqara
npm run eval:jqara-retrieval
```

Run the Ollama agent comparison:

```bash
npm run eval:ollama-jqara -- --limit 10
```

## Data Layout

JQaRA docs live under this VFS path:

```text
/jqara/<split>/<title>/<passage-id>.mdx
```

The generated source data is written under `data_eval/jqara/`.

## Reports

The retrieval report compares `lexical` and `semantic_search` with nDCG@10,
MRR@10, and Recall@1/5/10. The agent report compares `grep-only` against
`grep+semantic_search`.

Default reports are written to:

- `reports/jqara-retrieval-eval.jsonl`
- `reports/jqara-retrieval-eval.md`
- `reports/ollama-jqara-agent-eval.jsonl`
- `reports/ollama-jqara-agent-eval.md`

## Index Prefix

JQaRA bootstrap/eval scripts use isolated `opensearchfs-jqara-*` indices, so
normal docs and JQaRA docs can coexist.
