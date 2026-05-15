# Ollama QA Eval

This experiment prepares a small JaQuAD/JSQuAD filesystem and runs QA cases
with local Ollama tool calling over `docs_bash`.

## Commands

```bash
npm run prepare:qa-dataset -- --limit-per-dataset 5
npm run bootstrap:qa
npm run eval:ollama-qa -- --limit 1
npm run eval:ollama-qa -- --dataset jsquad --limit 1 --output reports/ollama-qa-eval-jsquad.jsonl
```

Use `OLLAMA_MODEL=gemma4:e4b` to override the default local model.

## Data Layout

The generated docs live under this VFS path:

```text
/qa/<dataset>/<title>/<context-id>.mdx
```

The generated source data is written under `data_eval/qa/`.

## Reports

Default reports are written to:

- `reports/ollama-qa-eval.jsonl`
- `reports/ollama-qa-eval.md`

## Index Prefix

QA bootstrap/eval scripts use isolated `opensearchfs-qa-*` indices, so they do
not replace the default docs index.
