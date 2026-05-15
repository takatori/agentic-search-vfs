import type { CommandContext, ExecResult } from 'just-bash';
import type { Client } from '@opensearch-project/opensearch';
import yargsParser from 'yargs-parser';
import {
  getOpenSearchFsIndexNames,
  type OpenSearchFsIndexNames,
} from '../opensearchfs-constants.js';
import { createEmbeddingProvider, type EmbeddingProvider } from './embedding.js';
import type {
  OpenSearchFs,
  SearchScopeFilter,
} from './opensearchfs.js';
import { normalizePath, pathToSlug, slugToPath } from './path-tree.js';

const DEFAULT_SEMANTIC_SEARCH_RESULT_LIMIT = 5;

export interface ParsedSemanticSearchArgv {
  query: string;
  pathArg: string;
  limit: number;
}

export interface RankedFileHit {
  slug: string;
  score: number;
  content: string;
  chunkId?: number;
}

type SearchHit = {
  _score?: number;
  _source?: {
    slug?: string;
  };
  inner_hits?: {
    chunks?: {
      hits?: {
        hits?: NestedChunkHit[];
      };
    };
  };
};

type NestedChunkHit = {
  _source?:
    | {
        chunk_id?: number;
        text?: string;
      }
    | {
        chunks?: {
          chunk_id?: number;
          text?: string;
        };
      };
  fields?: Record<string, unknown[]>;
};

type SearchBody = {
  hits?: {
    hits?: SearchHit[];
  };
};

function unwrapSearchBody(res: unknown): SearchBody {
  return (res as { body?: SearchBody }).body ?? (res as SearchBody);
}

function extractNestedChunkHit(
  hit: SearchHit,
): { chunkId?: number; text: string } | undefined {
  const nestedHit = hit.inner_hits?.chunks?.hits?.hits?.[0];
  if (!nestedHit) return undefined;
  const source = nestedHit._source;
  if (source && 'text' in source && typeof source.text === 'string') {
    return {
      chunkId: typeof source.chunk_id === 'number' ? source.chunk_id : undefined,
      text: source.text,
    };
  }
  if (
    source &&
    'chunks' in source &&
    source.chunks &&
    typeof source.chunks.text === 'string'
  ) {
    return {
      chunkId:
        typeof source.chunks.chunk_id === 'number'
          ? source.chunks.chunk_id
          : undefined,
      text: source.chunks.text,
    };
  }
  const fieldText = nestedHit.fields?.['chunks.text']?.[0];
  if (typeof fieldText === 'string') {
    const fieldChunkId = nestedHit.fields?.['chunks.chunk_id']?.[0];
    return {
      chunkId: typeof fieldChunkId === 'number' ? fieldChunkId : undefined,
      text: fieldText,
    };
  }
  return undefined;
}

export class OpenSearchSemanticSearcher {
  private readonly client: Client;
  private readonly indexNames: OpenSearchFsIndexNames;
  private embeddings: EmbeddingProvider | undefined;

  constructor(options: {
    client: Client;
    embeddings?: EmbeddingProvider;
    indexNames?: OpenSearchFsIndexNames;
  }) {
    this.client = options.client;
    this.embeddings = options.embeddings;
    this.indexNames = options.indexNames ?? getOpenSearchFsIndexNames();
  }

  async findMatchingFilesWithScope(
    pattern: string,
    scopeFilter: SearchScopeFilter,
    limit: number,
  ): Promise<RankedFileHit[]> {
    const k = Math.max(1, limit);
    const vector = await this.getEmbeddings().embed(pattern, 'query');
    const embeddingQuery: Record<string, unknown> = {
      vector,
      k: Math.max(k, Math.min(k * 10, 1000)),
    };
    const res = await this.searchNestedSemanticFiles(
      embeddingQuery,
      scopeFilter,
      k,
    );
    return this.parseRankedHits(res);
  }

  private getEmbeddings(): EmbeddingProvider {
    this.embeddings ??= createEmbeddingProvider();
    return this.embeddings;
  }

  private async searchNestedSemanticFiles(
    embeddingQuery: Record<string, unknown>,
    scopeFilter: SearchScopeFilter,
    size: number,
  ): Promise<unknown> {
    const nestedChunkQuery = {
      nested: {
        path: 'chunks',
        score_mode: 'max',
        query: {
          knn: {
            'chunks.embedding': embeddingQuery,
          },
        },
        inner_hits: {
          size: 1,
          _source: ['chunks.chunk_id', 'chunks.text'],
        },
      },
    };
    return this.client.search({
      index: this.indexNames.files,
      body: {
        size,
        _source: ['slug'],
        query: scopeFilter
          ? {
              bool: {
                filter: [scopeFilter],
                must: [nestedChunkQuery],
              },
            }
          : nestedChunkQuery,
      },
    } as never);
  }

  private parseRankedHits(res: unknown): RankedFileHit[] {
    const hits = unwrapSearchBody(res).hits?.hits ?? [];
    return hits
      .map((hit) => {
        const chunk = extractNestedChunkHit(hit);
        return {
          slug: hit._source?.slug ?? '',
          score: hit._score ?? 0,
          content: chunk?.text ?? '',
          chunkId: chunk?.chunkId,
        };
      })
      .filter((hit) => hit.slug.length > 0 && hit.content.length > 0);
  }
}

export function parseSemanticSearchLimit(
  raw = process.env.SEMANTIC_SEARCH_RESULT_LIMIT,
): number {
  if (raw === undefined || raw === '') return DEFAULT_SEMANTIC_SEARCH_RESULT_LIMIT;
  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(
      `Invalid SEMANTIC_SEARCH_RESULT_LIMIT "${raw}". Expected a positive integer.`,
    );
  }
  return parsed;
}

export function parseSemanticSearchArgv(
  args: string[],
): ParsedSemanticSearchArgv {
  const parsed = yargsParser(args, {
    configuration: {
      'short-option-groups': true,
      'camel-case-expansion': false,
      'unknown-options-as-args': true,
    },
    string: ['limit', 'n'],
    alias: {
      limit: 'n',
    },
  });
  const pos = parsed._.map(String);
  if (pos.length === 0 || pos[0] === '') {
    throw new Error('semantic_search: missing query');
  }
  if (pos.length > 2) {
    throw new Error('semantic_search: too many arguments');
  }
  const limitRaw =
    typeof parsed.limit === 'string'
      ? parsed.limit
      : typeof parsed.n === 'string'
        ? parsed.n
        : undefined;
  return {
    query: pos[0],
    pathArg: pos[1] ?? '.',
    limit: parseSemanticSearchLimit(limitRaw),
  };
}

export async function runOpenSearchSemanticSearch(
  args: string[],
  ctx: CommandContext,
  opensearchFs: OpenSearchFs,
  searcher: OpenSearchSemanticSearcher,
): Promise<ExecResult> {
  let parsed: ParsedSemanticSearchArgv;
  try {
    parsed = parseSemanticSearchArgv(args);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return { stdout: '', stderr: `${msg}\n`, exitCode: 2 };
  }

  const scope = await resolveSemanticSearchScope(
    opensearchFs,
    ctx.cwd,
    parsed.pathArg,
  );
  if (!scope.ok) {
    return { stdout: '', stderr: scope.stderr, exitCode: scope.exitCode };
  }

  try {
    const hits = await searcher.findMatchingFilesWithScope(
      parsed.query,
      scope.scopeFilter,
      parsed.limit,
    );
    return formatSemanticSearchOutput(hits, parsed.query);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return { stdout: '', stderr: `semantic_search: ${msg}\n`, exitCode: 2 };
  }
}

async function resolveSemanticSearchScope(
  fs: OpenSearchFs,
  cwd: string,
  pathArg: string,
): Promise<
  | { ok: true; scopeFilter: SearchScopeFilter }
  | { ok: false; stderr: string; exitCode: number }
> {
  const abs = normalizePath(fs.resolvePath(cwd, pathArg));
  if (!(await fs.exists(abs))) {
    return {
      ok: false,
      stderr: `semantic_search: ${pathArg}: No such file or directory\n`,
      exitCode: 2,
    };
  }
  const st = await fs.stat(abs);
  if (st.isFile) {
    return { ok: true, scopeFilter: { term: { slug: pathToSlug(abs) } } };
  }
  if (st.isDirectory) {
    return {
      ok: true,
      scopeFilter:
        abs === '/' ? undefined : { prefix: { slug: `${abs.slice(1)}/` } },
    };
  }
  return {
    ok: false,
    stderr: `semantic_search: ${pathArg}: Unsupported file type\n`,
    exitCode: 2,
  };
}

function formatSemanticSearchOutput(
  hits: RankedFileHit[],
  query: string,
): ExecResult {
  if (hits.length === 0) return { stdout: '', stderr: '', exitCode: 1 };
  const stdout = hits
    .map((hit) => `${slugToPath(hit.slug)}:${selectSnippet(hit.content, query)}`)
    .join('\n');
  return { stdout: `${stdout}\n`, stderr: '', exitCode: 0 };
}

function selectSnippet(content: string, query: string): string {
  const body = content.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n/u, '');
  const lines = body
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length === 0) return '';

  const queryTerms = query
    .normalize('NFKC')
    .toLowerCase()
    .split(/[^\p{L}\p{N}_-]+/u)
    .filter(Boolean);
  if (queryTerms.length === 0) return clipSnippet(lines[0]);

  let best = lines[0];
  let bestScore = -1;
  for (const line of lines) {
    const normalized = line.normalize('NFKC').toLowerCase();
    const score = queryTerms.filter((term) => normalized.includes(term)).length;
    if (score > bestScore) {
      best = line;
      bestScore = score;
    }
  }
  return clipSnippet(best);
}

function clipSnippet(value: string): string {
  const singleLine = value.replace(/\s+/gu, ' ').trim();
  return singleLine.length <= 240 ? singleLine : `${singleLine.slice(0, 240)}...`;
}
