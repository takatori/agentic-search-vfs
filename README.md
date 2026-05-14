# OpenSearchFs: Virtual Filesystem over OpenSearch

This repository is an OpenSearch rewrite of
[`iamleonie/elasticsearch-fs`](https://github.com/iamleonie/elasticsearch-fs).
The structure and `just-bash` integration intentionally stay close to the
original project, which accompanies Leonie Monigatti's article
[Implementing a virtual filesystem over Elasticsearch](https://leoniemonigatti.com/blog/virtual-filesystem-elasticsearch.html).

`grep` is kept as a normal lexical command: OpenSearch performs coarse search,
then `OpenSearchFs.readFile` is used for line-level matching. The practical
agent experiment exposes a separate
`semantic_search "<query>" [path]` command through `docs_bash`. This keeps
`grep` as the lexical baseline while giving the agent an explicit semantic
retrieval primitive.

## Requirements

- Node.js 18+
- Python 3.12+ managed by `uv` for indexing
- Docker for local OpenSearch
- npm

## Setup

```bash
cp .env.example .env
uv sync
npm install
docker compose up -d
npm run bootstrap
```

`npm run bootstrap` reads `./example_data` and
`./example_data/path_tree.json`, then syncs:

- `opensearchfs-chunks`
- `opensearchfs-meta`

`npm run bootstrap` delegates indexing to
`PYTHONPATH=indexer uv run python -m opensearchfs_indexer --sync`.
The Python indexer upserts changed files by content hash and embedding model id,
deletes removed files, indexes nested chunk embeddings inside each file document,
and configures OpenSearch Security plugin roles/users for `PUBLIC`, `BILLING`,
and `INTERNAL`. Runtime commands read through OpenSearch; `./example_data` is
bootstrap input only. The first bootstrap run downloads the selected Python
embedding model from Hugging Face.

Use `npm run bootstrap:recreate` to drop and recreate the OpenSearch indices.
Set `OPENSEARCHFS_INDEX_PREFIX=<prefix>` or pass `--index-prefix` to the
indexer to use isolated index pairs named `<prefix>-chunks` and `<prefix>-meta`.
The default prefix is `opensearchfs`.

## Quickstart

```bash
npm run quickstart -- PUBLIC
```

The quickstart registers `grep` and `semantic_search` as custom commands in
`just-bash`. Built-in `ls` and `cat` operate through `OpenSearchFs`.

## Claude Agent SDK Demo

Run this from an authenticated Claude Code CLI environment:

```bash
npm run agent-demo -- BILLING "請求関連のドキュメントを要約して"
```

The agent receives one MCP tool, `docs_bash`, backed by the same
`OpenSearchFs` runtime. It can run read-only filesystem-style commands such as
`ls`, `cat`, `grep`, `semantic_search`, `pwd`, `cd`, `head`, `tail`, and `find`.
Agent prompts should treat `semantic_search` as an optional retrieval strategy:
use it when exact keywords are uncertain, natural-language matching is useful,
or grep returns weak candidates. They do not need to start with semantic search.

Ruri uses retrieval-specific prefixes: indexed documents are embedded by the
Python indexer as `検索文書: ...`, and semantic queries are embedded by the
TypeScript runtime as `検索クエリ: ...`.

## Ollama QA Eval

Prepare a small JaQuAD/JSQuAD filesystem and run QA cases with local Ollama tool
calling:

```bash
npm run prepare:qa-dataset -- --limit-per-dataset 5
npm run bootstrap:qa
npm run eval:ollama-qa -- --limit 1
npm run eval:ollama-qa -- --dataset jsquad --limit 1 --output reports/ollama-qa-eval-jsquad.jsonl
```

The generated docs live under `/qa/<dataset>/<title>/<context-id>.mdx` inside
OpenSearchFs. Reports are written to `reports/ollama-qa-eval.jsonl` and
`reports/ollama-qa-eval.md`. Use `OLLAMA_MODEL=gemma4:e4b` to override the
default local model. QA bootstrap/eval scripts use the isolated
`opensearchfs-qa-*` indices, so they do not replace the default docs index.

## JQaRA Eval

Prepare JQaRA passages as an OpenSearchFs tree and evaluate full-passage
retrieval:

```bash
npm run prepare:jqara-dataset -- --split test --limit-questions 20
npm run bootstrap:jqara
npm run eval:jqara-retrieval
```

Run the Ollama agent comparison with and without the explicit semantic command:

```bash
npm run eval:ollama-jqara -- --limit 10
```

JQaRA docs live under `/jqara/<split>/<title>/<passage-id>.mdx`. The retrieval
report compares `lexical` and `semantic_search` with nDCG@10, MRR@10, and
Recall@1/5/10. The agent report compares `grep-only` against
`grep+semantic_search`. JQaRA bootstrap/eval scripts use the isolated
`opensearchfs-jqara-*` indices, so normal docs and JQaRA docs can coexist.

## EnterpriseRAG-Bench Eval

Prepare Onyx EnterpriseRAG-Bench as an enterprise-like filesystem with source
type and document title hierarchy:

```bash
npm run prepare:enterprise-rag-dataset -- --split test --limit-questions 10 --limit-documents 1000
npm run bootstrap:enterprise-rag
npm run eval:enterprise-rag-retrieval -- --limit 10
```

Run the Ollama agent comparison with the same `docs_bash` surface used by the
JQaRA experiment:

```bash
npm run eval:ollama-enterprise-rag -- --limit 5
```

EnterpriseRAG-Bench docs live under
`/enterprise-rag/<split>/<source-type>/<title>/<doc-id>.mdx`. Ground-truth
documents referenced by selected questions are always included even when
`--limit-documents` is set. The retrieval report compares `lexical` and
`semantic_search`; the agent report compares `grep-only` against
`grep+semantic_search`. These scripts use isolated
`opensearchfs-enterprise-rag-*` indices, so they do not overwrite the normal
docs, QA, or JQaRA indices.

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

## Profiles

The local defaults are:

```env
OPENSEARCH_URL=https://localhost:19200
OPENSEARCH_USERNAME_SYSTEM=admin
OPENSEARCH_PASSWORD_SYSTEM=SearchFsRoot2026!

OPENSEARCH_USERNAME_PUBLIC=public
OPENSEARCH_PASSWORD_PUBLIC=SearchFsReadA2026!
OPENSEARCH_USERNAME_BILLING=billing
OPENSEARCH_PASSWORD_BILLING=SearchFsReadB2026!
OPENSEARCH_USERNAME_INTERNAL=internal
OPENSEARCH_PASSWORD_INTERNAL=SearchFsReadC2026!

RURI_INDEX_MODEL=cl-nagoya/ruri-v3-310m
RURI_INDEX_DEVICE=cpu
RURI_BATCH_SIZE=8

RURI_QUERY_MODEL=onnx-community/ruri-v3-310m-ONNX
RURI_QUERY_TOKENIZER_MODEL=cl-nagoya/ruri-v3-310m
RURI_QUERY_DTYPE=fp32
```

The default embedding provider is `ruri`. For English datasets such as
EnterpriseRAG-Bench, use Nomic with the same provider on both indexing and query:

```bash
EMBEDDING_PROVIDER=nomic npm run bootstrap:enterprise-rag
EMBEDDING_PROVIDER=nomic npm run eval:enterprise-rag-retrieval
```

Provider selection:

```env
EMBEDDING_PROVIDER=ruri # ruri | nomic
# Optional side-specific overrides:
INDEX_EMBEDDING_PROVIDER=ruri
QUERY_EMBEDDING_PROVIDER=ruri
EMBEDDING_CHUNK_MAX_CHARS=3000
EMBEDDING_CHUNK_OVERLAP_CHARS=300

NOMIC_INDEX_MODEL=nomic-ai/nomic-embed-text-v1.5
NOMIC_INDEX_DEVICE=cpu
NOMIC_BATCH_SIZE=8
NOMIC_MAX_SEQ_LENGTH=1024

NOMIC_QUERY_MODEL=nomic-ai/nomic-embed-text-v1.5
NOMIC_QUERY_TOKENIZER_MODEL=nomic-ai/nomic-embed-text-v1.5
NOMIC_QUERY_DTYPE=fp32
```

Both built-in providers currently emit 768-dimensional vectors. Each OpenSearch
file document stores the full `content` plus a nested `chunks[]` array containing
`chunk_id`, chunk `text`, and a `knn_vector` embedding. `semantic_search` searches
the nested chunk vectors and returns the parent `.mdx` path with the matched
chunk as the snippet, so the agent can still `cat <path>` to read the full file.
Changing provider, model, max sequence length, or chunk settings causes changed
or previously indexed documents to be re-embedded because `embedding_model_id` is
stored with each indexed document.

`SYSTEM` is used for bootstrap/admin work. Runtime profiles should use the
read-only users created by bootstrap so that OpenSearch document-level security
is part of the enforcement path.

## Tests

```bash
npm run typecheck
npm test
```

E2E tests require a bootstrapped OpenSearch instance and `.env` profile
credentials. Unit tests do not require OpenSearch. `semantic_search` E2E checks
are skipped by default; set `OPENSEARCHFS_RUN_RURI_E2E=1` to include them. The
first semantic run may download the configured ONNX query model.

## Notes

- Document indexing lives in `indexer/opensearchfs_indexer/` and uses Python
  `sentence-transformers`.
- Query-time embedding integration lives in `src/core/embedding.ts` and uses
  `@huggingface/transformers` without Python.
- Dataset preparation and evaluation code lives under `experiments/<dataset>/`.
- The Apache-2.0 license is preserved from the upstream project.
