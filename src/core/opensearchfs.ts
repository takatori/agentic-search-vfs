import nodePath from 'node:path/posix';
import type { Client } from '@opensearch-project/opensearch';
import type {
  BufferEncoding,
  CpOptions,
  FileContent,
  FsStat,
  IFileSystem,
  MkdirOptions,
  RmOptions,
} from 'just-bash';
import {
  getOpenSearchFsIndexNames,
  type OpenSearchFsIndexNames,
} from '../opensearchfs-constants.js';
import { createEmbeddingProvider, type EmbeddingProvider } from './embedding.js';
import type {
  DirentEntry,
  ReadFileOptions,
  WriteFileOptions,
} from './just-bash-fs-types.js';
import { normalizePath, pathToSlug } from './path-tree.js';

// POSIX EROFS — mutating operations are not allowed on this read-only VFS.
function erofs(): Error {
  const err = new Error(
    'EROFS: read-only file system',
  ) as NodeJS.ErrnoException;
  err.code = 'EROFS';
  return err;
}

// POSIX ENOTDIR — path refers to a file where a directory was required (e.g. `ls` on a file).
function enotdir(): Error {
  const err = new Error('ENOTDIR: not a directory') as NodeJS.ErrnoException;
  err.code = 'ENOTDIR';
  return err;
}

// POSIX ENOENT — path is not present in the tree (no such file or directory).
function enoent(): Error {
  const err = new Error(
    'ENOENT: no such file or directory',
  ) as NodeJS.ErrnoException;
  err.code = 'ENOENT';
  return err;
}

// POSIX EINVAL — e.g. `readlink` on a path that exists but is not a symlink.
function einval(message: string): Error {
  const err = new Error(message) as NodeJS.ErrnoException;
  err.code = 'EINVAL';
  return err;
}

interface FileHitSource {
  content?: string;
  slug?: string;
}

export interface RankedFileHit {
  slug: string;
  score: number;
  content: string;
  chunkId?: number;
}

/**
 * Coarse grep filter used by {@link OpenSearchFs.findMatchingFiles}.
 */
export interface GrepCoarseFilter {
  pattern: string;
  ignoreCase: boolean;
  fixedStrings: boolean;
}

export type OpenSearchQuery = Record<string, unknown>;
export type SearchScopeFilter = OpenSearchQuery | undefined;

const SEARCH_PAGE_SIZE = 1000;

function hasRegexMeta(pattern: string): boolean {
  return /[\\^$.*+?()[\]{}|]/u.test(pattern);
}

function escapeRegexpLiteral(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/gu, '\\$&');
}

type SearchHit = {
  _score?: number;
  _source?: FileHitSource;
  inner_hits?: {
    chunks?: {
      hits?: {
        hits?: NestedChunkHit[];
      };
    };
  };
};

type NestedChunkHit = {
  _score?: number;
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

type SemanticVectorLayout = 'nested_chunks' | 'legacy_embedding';

function unwrapSearchBody(res: unknown): SearchBody {
  return (
    (res as { body?: SearchBody }).body ??
    (res as SearchBody)
  );
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

function isMissingNestedChunksError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  if (message.includes('failed to find nested object under path [chunks]')) {
    return true;
  }
  const body = (error as { meta?: { body?: unknown } } | undefined)?.meta?.body;
  return JSON.stringify(body ?? {}).includes(
    'failed to find nested object under path [chunks]',
  );
}

/**
 * Read-only virtual filesystem backed by OpenSearch file documents and a preloaded path tree.
 */
export class OpenSearchFs implements IFileSystem {
  private files = new Set<string>();
  private dirs = new Map<string, string[]>();
  private readonly client: Client;
  private readonly embeddings: EmbeddingProvider;
  private readonly indexNames: OpenSearchFsIndexNames;
  private semanticVectorLayout: SemanticVectorLayout | undefined;

  constructor(options: {
    client: Client;
    files: ReadonlySet<string>;
    dirs: ReadonlyMap<string, string[]>;
    embeddings?: EmbeddingProvider;
    indexNames?: OpenSearchFsIndexNames;
  }) {
    this.client = options.client;
    this.files = new Set(options.files);
    this.dirs = new Map(options.dirs);
    this.embeddings = options.embeddings ?? createEmbeddingProvider();
    this.indexNames = options.indexNames ?? getOpenSearchFsIndexNames();
  }

  /** Map a normalized path to canonical tree file key (`files` stores `/<slug>.mdx`). */
  private resolveTreeFileKey(normalized: string): string | undefined {
    return this.files.has(normalized) ? normalized : undefined;
  }

  /**
   * Path must exist as a file in the pruned tree (QUERY_SPEC);
   * this validates a specific path for reading.
   * @returns Ingest `slug` for file content documents (e.g. `auth/oauth`).
   */
  private resolveReadFileSlug(path: string): string {
    const normalized = normalizePath(path);
    if (this.dirs.has(normalized)) {
      throw enotdir();
    }
    const treeKey = this.resolveTreeFileKey(normalized);
    if (treeKey === undefined) {
      throw enoent();
    }
    return pathToSlug(treeKey);
  }

  /**
   * Ingest `slug` for a visible canonical file, or `null` if not in the tree.
   */
  getFileSlug(vfsPath: string): string | null {
    const normalized = normalizePath(vfsPath);
    const treeKey = this.resolveTreeFileKey(normalized);
    if (treeKey === undefined) return null;
    return pathToSlug(treeKey);
  }

  /**
   * Normalized visible file paths in the pruned tree (same keys as `files`).
   */
  getVisibleFilePaths(): string[] {
    return [...this.files].sort();
  }

  /**
   * Paginate over the files index with `search_after`, processing each page via callbacks.
   * Stops when a page is empty, shorter than `SEARCH_PAGE_SIZE`, or the cursor extractor returns `undefined`.
   */
  private async searchAllPages(
    params: Record<string, unknown>,
    extractCursor: (
      hits: { _source?: FileHitSource }[],
    ) => unknown[] | undefined,
    onPage: (hits: { _source?: FileHitSource }[]) => void,
  ): Promise<void> {
    let searchAfter: unknown[] | undefined;
    while (true) {
      const body = {
        ...((params.body as Record<string, unknown> | undefined) ?? {}),
        size: SEARCH_PAGE_SIZE,
        ...(searchAfter !== undefined ? { search_after: searchAfter } : {}),
      };
      const res = await this.client.search({
        ...params,
        body,
      } as never);
      const hits = unwrapSearchBody(res).hits?.hits ?? [];
      if (hits.length === 0) break;
      onPage(hits);
      if (hits.length < SEARCH_PAGE_SIZE) break;
      const cursor = extractCursor(hits);
      if (cursor === undefined) break;
      searchAfter = cursor;
    }
  }

  /**
   * Coarse stage for `grep`: distinct file `slug` values that may match.
   *
   * @param slugsUnderDirs In-scope ingest slugs (e.g. `auth/oauth`).
   * @returns Slugs that passed the coarse query and optional match_phrase/regexp filter.
   */
  async findMatchingFiles(
    coarseFilter: GrepCoarseFilter,
    slugsUnderDirs: string[],
  ): Promise<string[]> {
    if (slugsUnderDirs.length === 0) return [];
    return this.findMatchingFilesWithScope(coarseFilter, {
      terms: { slug: slugsUnderDirs },
    });
  }

  async findMatchingFilesWithScope(
    coarseFilter: GrepCoarseFilter,
    scopeFilter?: SearchScopeFilter,
  ): Promise<string[]> {
    const isLiteralPattern =
      coarseFilter.fixedStrings || !hasRegexMeta(coarseFilter.pattern);

    const query = {
      bool: {
        ...(scopeFilter ? { filter: [scopeFilter] } : {}),
        must: isLiteralPattern
          ? coarseFilter.ignoreCase
            ? [
                {
                  match_phrase: {
                    content: coarseFilter.pattern,
                  },
                },
              ]
            : [
                {
                  regexp: {
                    'content.pattern': {
                      value: `.*(${escapeRegexpLiteral(coarseFilter.pattern)}).*`,
                      case_insensitive: false,
                    },
                  },
                },
              ]
          : [
              {
                regexp: {
                  'content.pattern': {
                    value: `.*(${coarseFilter.pattern}).*`,
                    case_insensitive: coarseFilter.ignoreCase,
                  },
                },
              },
            ],
      },
    };

    const slugs = new Set<string>();
    await this.searchAllPages(
      {
        index: this.indexNames.files,
        body: {
          track_total_hits: false,
          _source: ['slug'],
          sort: [{ slug: { order: 'asc' } }],
          query,
        },
      },
      (hits) => {
        const last = hits[hits.length - 1];
        const lastSlug = last?._source?.slug;
        if (typeof lastSlug !== 'string' || lastSlug.length === 0) {
          return undefined;
        }
        return [lastSlug];
      },
      (hits) => {
        for (const hit of hits) {
          const s = hit._source?.slug;
          if (typeof s === 'string' && s.length > 0) slugs.add(s);
        }
      },
    );
    return [...slugs];
  }

  async findSemanticMatchingFiles(
    pattern: string,
    slugsUnderDirs: string[],
    limit: number,
  ): Promise<RankedFileHit[]> {
    if (slugsUnderDirs.length === 0) return [];
    const k = Math.max(1, Math.min(limit, slugsUnderDirs.length));
    return this.findSemanticMatchingFilesWithScope(
      pattern,
      { terms: { slug: slugsUnderDirs } },
      k,
    );
  }

  async findSemanticMatchingFilesWithScope(
    pattern: string,
    scopeFilter: SearchScopeFilter,
    limit: number,
  ): Promise<RankedFileHit[]> {
    const k = Math.max(1, limit);
    const vector = await this.embeddings.embed(pattern, 'query');
    const embeddingQuery: Record<string, unknown> = {
      vector,
      k: Math.max(k, Math.min(k * 10, 1000)),
    };

    if (this.semanticVectorLayout === 'legacy_embedding') {
      const legacyRes = await this.searchLegacySemanticFiles(
        embeddingQuery,
        scopeFilter,
        k,
      );
      return this.parseRankedHits(legacyRes);
    }

    let res: unknown;
    try {
      res = await this.searchNestedSemanticFiles(embeddingQuery, scopeFilter, k);
      this.semanticVectorLayout = 'nested_chunks';
    } catch (error) {
      if (!isMissingNestedChunksError(error)) {
        throw error;
      }
      this.semanticVectorLayout = 'legacy_embedding';
      res = await this.searchLegacySemanticFiles(embeddingQuery, scopeFilter, k);
    }
    return this.parseRankedHits(res);
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

  private async searchLegacySemanticFiles(
    embeddingQuery: Record<string, unknown>,
    scopeFilter: SearchScopeFilter,
    size: number,
  ): Promise<unknown> {
    const legacyEmbeddingQuery = {
      knn: {
        embedding: embeddingQuery,
      },
    };
    const res = await this.client.search({
      index: this.indexNames.files,
      body: {
        size,
        _source: ['slug', 'content'],
        query: scopeFilter
          ? {
              bool: {
                filter: [scopeFilter],
                must: [legacyEmbeddingQuery],
              },
            }
          : legacyEmbeddingQuery,
      },
    } as never);
    return res;
  }

  /**
   * Read the contents of a file as a string (default: utf8)
   * @throws Error if file doesn't exist or is a directory
   */
  async readFile(
    path: string,
    options?: ReadFileOptions | BufferEncoding,
  ): Promise<string> {
    void options;
    const slug = this.resolveReadFileSlug(path);

    const res = await this.client.search({
      index: this.indexNames.files,
      body: {
        size: 1,
        _source: ['content'],
        query: { bool: { filter: [{ term: { slug } }] } },
      },
    } as never);
    const hit = unwrapSearchBody(res).hits?.hits?.[0];
    const content = hit?._source?.content;
    if (content === undefined) {
      throw enoent();
    }

    return content;
  }

  private parseRankedHits(res: unknown): RankedFileHit[] {
    const hits = unwrapSearchBody(res).hits?.hits ?? [];
    return hits
      .map((hit) => {
        const chunk = extractNestedChunkHit(hit);
        return {
          slug: hit._source?.slug ?? '',
          score: hit._score ?? 0,
          content: chunk?.text ?? hit._source?.content ?? '',
          chunkId: chunk?.chunkId,
        };
      })
      .filter((hit) => hit.slug.length > 0 && hit.content.length > 0);
  }

  /**
   * Read the contents of a file as a Uint8Array (binary)
   * Same logical file as {@link readFile}, as UTF-8 bytes (corpus is text in ES).
   * Implemented by reusing `readFile` then `TextEncoder`.
   * @throws Error if file doesn't exist or is a directory
   */
  async readFileBuffer(path: string): Promise<Uint8Array> {
    const text = await this.readFile(path);
    return new TextEncoder().encode(text);
  }

  /**
   * Does not write content to a file, nor create it if it doesn't exist.
   * @throws Error if called to enforce read-only interaction.
   */
  async writeFile(
    path: string,
    content: FileContent,
    options?: WriteFileOptions | BufferEncoding,
  ): Promise<void> {
    void path;
    void content;
    void options;
    throw erofs();
  }

  /**
   * Does not append content to a file, nor create it if it doesn't exist.
   * @throws Error if called to enforce read-only interaction.
   */
  async appendFile(
    path: string,
    content: FileContent,
    options?: WriteFileOptions | BufferEncoding,
  ): Promise<void> {
    void path;
    void content;
    void options;
    throw erofs();
  }

  /**
   * Check if a path exists
   */
  async exists(path: string): Promise<boolean> {
    const normalized = normalizePath(path);

    // Directory: any path that appears as a parent in `buildFileTree` (see `path-tree.ts`).
    if (this.dirs.has(normalized)) return true;

    // File: canonical `/<slug>.mdx` only.
    if (this.resolveTreeFileKey(normalized) !== undefined) return true;

    return false;
  }

  /**
   * Get file/directory information
   * @throws Error if path doesn't exist
   */
  async stat(path: string): Promise<FsStat> {
    const normalized = normalizePath(path);

    if (this.dirs.has(normalized)) {
      return {
        isFile: false,
        isDirectory: true,
        isSymbolicLink: false,
        mode: 0o40755, // directory, user rwx / group+other rx (placeholder; not enforced on this VFS)
        size: 0, // POSIX often reports 0 for dirs; we have no per-dir byte size in the tree
        mtime: new Date(0), // synthetic — virtual dirs are inferred from file paths, not stored in ES
      };
    }

    const treeKey = this.resolveTreeFileKey(normalized);
    if (treeKey !== undefined) {
      return {
        isFile: true,
        isDirectory: false,
        isSymbolicLink: false,
        mode: 0o100644, // regular file, user rw / group+other r (placeholder; not enforced on this VFS)
        size: 0,
        mtime: new Date(0),
      };
    }

    throw enoent();
  }

  /**
   * Does not create a directory.
   * @throws Error if called to enforce read-only interaction.
   */
  async mkdir(path: string, options?: MkdirOptions): Promise<void> {
    void path;
    void options;
    throw erofs();
  }

  /**
   * Read directory contents
   * @returns Array of entry names (not full paths)
   * @throws Error if path doesn't exist or is not a directory
   */
  async readdir(path: string): Promise<string[]> {
    const normalized = normalizePath(path);

    const names = this.dirs.get(normalized);

    if (names !== undefined) {
      return [...names];
    }

    if (this.files.has(normalized)) {
      throw enotdir();
    }

    throw enoent();
  }

  /**
   * Read directory contents with file type information (optional)
   * This is more efficient than readdir + stat for each entry
   * @returns Array of DirentEntry objects with name and type
   * @throws Error if path doesn't exist or is not a directory
   */
  async readdirWithFileTypes(path: string): Promise<DirentEntry[]> {
    const names = await this.readdir(path);
    const normalized = normalizePath(path);
    const out: DirentEntry[] = [];

    for (const name of names) {
      const childPath = normalizePath(nodePath.join(normalized, name));
      const isDirectory = this.dirs.has(childPath);
      const isFile = !isDirectory && this.files.has(childPath);

      out.push({
        name,
        isFile,
        isDirectory,
        isSymbolicLink: false,
      });
    }

    return out;
  }

  /**
   * Does not remove a file or directory.
   * @throws Error if called to enforce read-only interaction.
   */
  async rm(path: string, options?: RmOptions): Promise<void> {
    void path;
    void options;
    throw erofs();
  }

  /**
   * Does not copy a file or directory.
   * @throws Error if called to enforce read-only interaction.
   */
  async cp(src: string, dest: string, options?: CpOptions): Promise<void> {
    void src;
    void dest;
    void options;
    throw erofs();
  }

  /**
   * Does not move or rename a file or directory.
   * @throws Error if called to enforce read-only interaction.
   */
  async mv(src: string, dest: string): Promise<void> {
    void src;
    void dest;
    throw erofs();
  }

  /**
   * Resolve a relative path against a base path
   */
  resolvePath(base: string, path: string): string {
    const rel = path.trim();
    if (rel.startsWith('/')) {
      return normalizePath(rel);
    }
    return normalizePath(nodePath.join(normalizePath(base), rel));
  }

  /**
   * Get all paths in the filesystem (useful for glob matching)
   * Optional - implementations may return empty array if not supported
   */
  getAllPaths(): string[] {
    return [...this.files, ...this.dirs.keys()].sort();
  }

  /**
   * Does not change file or directory permissions.
   * @throws Error if called to enforce read-only interaction.
   */
  async chmod(path: string, mode: number): Promise<void> {
    void path;
    void mode;
    throw erofs();
  }

  /**
   * Does not create a symbolic link.
   * @throws Error if called to enforce read-only interaction.
   */
  async symlink(target: string, linkPath: string): Promise<void> {
    void target;
    void linkPath;
    throw erofs();
  }

  /**
   * Does not create a hard link.
   * @throws Error if called to enforce read-only interaction.
   */
  async link(existingPath: string, newPath: string): Promise<void> {
    void existingPath;
    void newPath;
    throw erofs();
  }

  /**
   * Read the target of a symbolic link
   * @throws Error if path doesn't exist or is not a symlink
   */
  async readlink(path: string): Promise<string> {
    const n = normalizePath(path);
    if (!(await this.exists(n))) {
      throw enoent();
    }
    throw einval('readlink: not a symbolic link');
  }

  /**
   * Get file/directory information without following symlinks
   * Same as {@link stat} — this VFS does not model symlinks (`symlink` is read-only / unsupported),
   * so there is nothing to follow; `lstat` and `stat` return identical `FsStat` values.
   * @throws Error if path doesn't exist
   */
  async lstat(path: string): Promise<FsStat> {
    return this.stat(path);
  }

  /**
   * Resolve all symlinks in a path to get the canonical physical path.
   * This is equivalent to POSIX realpath() - it resolves all symlinks
   * in the path and returns the absolute physical path.
   * Used by pwd -P and cd -P for symlink resolution.
   * @throws Error if path doesn't exist or contains a broken symlink
   */
  async realpath(path: string): Promise<string> {
    const n = normalizePath(path);
    if (!(await this.exists(n))) {
      throw enoent();
    }
    return n;
  }

  /**
   * Does not set access or modification times of a file.
   * @throws Error if called to enforce read-only interaction.
   */
  async utimes(path: string, atime: Date, mtime: Date): Promise<void> {
    void path;
    void atime;
    void mtime;
    throw erofs();
  }
}
