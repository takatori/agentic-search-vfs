# AgenticSearchVfs: Virtual Filesystem over OpenSearch

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

## Experiments

Dataset preparation and evaluation details live with each experiment:

- [QA](experiments/qa/README.md): JaQuAD/JSQuAD QA cases with local Ollama tool
  calling.
- [JQaRA](experiments/jqara/README.md): Japanese full-passage retrieval and
  agent comparison.
- [EnterpriseRAG-Bench](experiments/enterprise-rag/README.md): enterprise-like
  retrieval and agent comparison, including the fixed 20k Nomic subset.

## Configuration

Runtime profiles and embedding settings are documented in `.env.example`. Copy it
to `.env` during setup and override values there. Experiment-specific settings
live in the experiment READMEs and package scripts.

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
