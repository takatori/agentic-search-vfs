/**
 * Two-stage `grep` for just-bash over OpenSearchFs: coarse + fine in-memory match.
 */

import type { CommandContext, ExecResult } from 'just-bash';
import type { Client } from '@opensearch-project/opensearch';
import yargsParser from 'yargs-parser';
import {
  getOpenSearchFsIndexNames,
  type OpenSearchFsIndexNames,
} from '../opensearchfs-constants.js';
import type { OpenSearchFs, SearchScopeFilter } from './opensearchfs.js';
import { normalizePath, pathToSlug, slugToPath } from './path-tree.js';

const SEARCH_PAGE_SIZE = 1000;

/** Returns true if the string contains any regex metacharacters (used to decide whether to treat a pattern as literal or regex). */
export function hasRegexMeta(pattern: string): boolean {
  return /[\\^$.*+?()[\]{}|]/.test(pattern);
}

/** Escapes all regex metacharacters in `value` so it can be embedded safely in a `RegExp` as a literal string. */
export function escapeRegexpLiteral(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

export interface GrepCoarseFilter {
  pattern: string;
  ignoreCase: boolean;
  fixedStrings: boolean;
}

type FileHitSource = {
  slug?: string;
};

type SearchHit = {
  _source?: FileHitSource;
};

type SearchBody = {
  hits?: {
    hits?: SearchHit[];
  };
};

function unwrapSearchBody(res: unknown): SearchBody {
  return (res as { body?: SearchBody }).body ?? (res as SearchBody);
}

type OpenSearchGrepSearchOptions = {
  indexNames?: OpenSearchFsIndexNames;
};

class OpenSearchGrepSearcher {
  private readonly client: Client;
  private readonly indexNames: OpenSearchFsIndexNames;

  constructor(options: {
    client: Client;
    indexNames?: OpenSearchFsIndexNames;
  }) {
    this.client = options.client;
    this.indexNames = options.indexNames ?? getOpenSearchFsIndexNames();
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

  /**
   * Paginate over the files index with `search_after`, processing each page via callbacks.
   * Stops when a page is empty, shorter than `SEARCH_PAGE_SIZE`, or the cursor extractor returns `undefined`.
   */
  private async searchAllPages(
    params: Record<string, unknown>,
    extractCursor: (hits: SearchHit[]) => unknown[] | undefined,
    onPage: (hits: SearchHit[]) => void,
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
}

export async function findGrepMatchingFiles(
  client: Client,
  coarseFilter: GrepCoarseFilter,
  slugsUnderDirs: string[],
  options?: OpenSearchGrepSearchOptions,
): Promise<string[]> {
  return new OpenSearchGrepSearcher({
    client,
    indexNames: options?.indexNames,
  }).findMatchingFiles(coarseFilter, slugsUnderDirs);
}

export async function findGrepMatchingFilesWithScope(
  client: Client,
  coarseFilter: GrepCoarseFilter,
  scopeFilter?: SearchScopeFilter,
  options?: OpenSearchGrepSearchOptions,
): Promise<string[]> {
  return new OpenSearchGrepSearcher({
    client,
    indexNames: options?.indexNames,
  }).findMatchingFilesWithScope(coarseFilter, scopeFilter);
}

// --- argv (yargs-parser) ---

interface ParsedGrepArgv {
  pattern: string;
  fileArgs: string[];
  ignoreCase: boolean;
  recursive: boolean;
  filesWithMatches: boolean;
  lineNumber: boolean;
  fixedStrings: boolean;
  invertMatch: boolean;
  quiet: boolean;
}

/**
 * Parse `grep` argv (command name already stripped).
 * Supports `grep [flags] pattern [files...]` and `grep [flags] -e pattern [files...]`.
 */
export function parseGrepArgv(
  args: string[],
  defaultFixedStrings?: boolean,
): ParsedGrepArgv {
  const parsed = yargsParser(args, {
    configuration: {
      'short-option-groups': true,
      'camel-case-expansion': false,
      'unknown-options-as-args': true,
    },
    boolean: [
      'i',
      'ignore-case',
      'r',
      'R',
      'recursive',
      'l',
      'files-with-matches',
      'n',
      'line-number',
      'F',
      'fixed-strings',
      'v',
      'invert-match',
      'q',
      'quiet',
    ],
    string: ['e'],
    alias: {
      'ignore-case': 'i',
      recursive: ['r', 'R'],
      'files-with-matches': 'l',
      'line-number': 'n',
      'fixed-strings': 'F',
      'invert-match': 'v',
      quiet: 'q',
    },
  });

  const fixedStrings = Boolean(parsed.F ?? defaultFixedStrings);
  const hasExplicitPattern = typeof parsed.e === 'string' && parsed.e !== '';

  let pattern: string | undefined;
  const pos = parsed._ as string[];
  if (hasExplicitPattern) {
    pattern = parsed.e;
  } else if (pos.length > 0) {
    pattern = pos[0];
  }

  const fileArgs = hasExplicitPattern ? pos : pos.slice(1);

  if (pattern === undefined || pattern === '') {
    throw new Error('grep: missing pattern');
  }

  return {
    pattern,
    fileArgs,
    ignoreCase: Boolean(parsed.i),
    recursive: Boolean(parsed.r ?? parsed.R ?? parsed.recursive),
    filesWithMatches: Boolean(parsed.l),
    lineNumber: Boolean(parsed.n),
    fixedStrings,
    invertMatch: Boolean(parsed.v),
    quiet: Boolean(parsed.q),
  };
}

/**
 * Resolve CLI path arguments to vfs file paths for grep.
 * Directories require `-r` / `-R` / `--recursive` (matches GNU grep).
 */
async function listVfsFilesForGrep(
  fs: OpenSearchFs,
  cwd: string,
  fileArgs: string[],
  recursive: boolean,
): Promise<
  | { ok: true; vfsPaths: string[]; scopeFilter: SearchScopeFilter }
  | { ok: false; stderr: string; exitCode: number }
> {
  const roots = fileArgs.length > 0 ? fileArgs : ['.'];
  const out = new Set<string>();
  const filters: SearchScopeFilter[] = [];
  let includesAllVisibleFiles = false;
  const allFiles = fs.getVisibleFilePaths();

  for (const r of roots) {
    const abs = normalizePath(fs.resolvePath(cwd, r));
    if (!(await fs.exists(abs))) {
      return {
        ok: false,
        stderr: `grep: ${r}: No such file or directory\n`,
        exitCode: 2,
      };
    }
    const st = await fs.stat(abs);
    if (st.isFile) {
      out.add(abs);
      filters.push({ term: { slug: pathToSlug(abs) } });
      continue;
    }
    if (st.isDirectory) {
      if (!recursive) {
        return {
          ok: false,
          stderr: `grep: ${r}: Is a directory\n`,
          exitCode: 2,
        };
      }
      const prefix = abs;
      if (prefix === '/') {
        includesAllVisibleFiles = true;
      } else {
        filters.push({
          prefix: { slug: `${prefix.slice(1)}/` },
        });
      }
      for (const fp of allFiles) {
        if (prefix === '/' || fp === prefix || fp.startsWith(`${prefix}/`)) {
          out.add(fp);
        }
      }
    }
  }

  return {
    ok: true,
    vfsPaths: [...out].sort(),
    scopeFilter: includesAllVisibleFiles ? undefined : combineScopeFilters(filters),
  };
}

function combineScopeFilters(filters: SearchScopeFilter[]): SearchScopeFilter {
  const concreteFilters = filters.filter(
    (filter): filter is Exclude<SearchScopeFilter, undefined> =>
      filter !== undefined,
  );
  if (concreteFilters.length === 0) return undefined;
  if (concreteFilters.length === 1) return concreteFilters[0];
  return {
    bool: {
      should: concreteFilters,
      minimum_should_match: 1,
    },
  };
}

/**
 * Builds a per-line match function from grep pattern options.
 * Handles `fixedStrings` (literal match), `ignoreCase`, and `invertMatch` modes.
 */
export function buildLinePredicate(
  pattern: string,
  opts: {
    fixedStrings: boolean;
    ignoreCase: boolean;
    invertMatch: boolean;
  },
): (line: string) => boolean {
  let test: (line: string) => boolean;

  if (opts.fixedStrings) {
    const needle = opts.ignoreCase ? pattern.toLowerCase() : pattern;
    test = (line) => {
      const h = opts.ignoreCase ? line.toLowerCase() : line;
      return h.includes(needle);
    };
  } else {
    const flags = opts.ignoreCase ? 'imu' : 'mu';
    const re = new RegExp(pattern, flags);
    test = (line) => re.test(line);
  }

  if (!opts.invertMatch) return test;
  return (line) => !test(line);
}

interface GrepLineHit {
  lineNo: number;
  line: string;
}

/** Runs `predicate` against each line of `content` and returns 1-indexed line numbers with the matching text. */
function findMatchingLines(
  content: string,
  predicate: (line: string) => boolean,
): GrepLineHit[] {
  const lines = content.split(/\r?\n/u);
  const out: GrepLineHit[] = [];
  for (const [i, line] of lines.entries()) {
    if (predicate(line)) {
      out.push({ lineNo: i + 1, line });
    }
  }
  return out;
}

/**
 * Formats grep hits as a string following GNU grep output conventions.
 * In `filesWithMatches` mode returns only the path; otherwise formats each hit as
 * `[path:][lineNo:]line`, prefixing the path only when `multiFile` is true.
 */
export function formatGrepOutput(
  vfsPath: string,
  hits: GrepLineHit[],
  opts: {
    filesWithMatches: boolean;
    lineNumber: boolean;
    multiFile: boolean;
  },
): string {
  if (opts.filesWithMatches) {
    return hits.length > 0 ? `${vfsPath}\n` : '';
  }
  let buf = '';
  for (const h of hits) {
    if (opts.multiFile) {
      buf += opts.lineNumber
        ? `${vfsPath}:${h.lineNo}:${h.line}\n`
        : `${vfsPath}:${h.line}\n`;
    } else {
      buf += opts.lineNumber ? `${h.lineNo}:${h.line}\n` : `${h.line}\n`;
    }
  }
  return buf;
}

/**
 * Fine stage: in-memory line match over `readFile`
 * (no stock just-bash `grep` builtin; that would recurse into our custom command).
 */
async function execBuiltin(
  parsed: ParsedGrepArgv,
  vfsPaths: string[],
  fs: OpenSearchFs,
  forceMultiFileOutput = false,
): Promise<ExecResult> {
  if (vfsPaths.length === 0) {
    return { stdout: '', stderr: '', exitCode: 1 };
  }

  let predicate: (line: string) => boolean;
  try {
    predicate = buildLinePredicate(parsed.pattern, {
      fixedStrings: parsed.fixedStrings,
      ignoreCase: parsed.ignoreCase,
      invertMatch: parsed.invertMatch,
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return {
      stdout: '',
      stderr: `grep: invalid regular expression: ${msg}\n`,
      exitCode: 2,
    };
  }

  let stdout = '';
  let matchedAny = false;

  for (const vfsPath of vfsPaths) {
    try {
      const text = await fs.readFile(vfsPath);
      const hits = findMatchingLines(text, predicate);
      if (hits.length > 0) {
        matchedAny = true;
        if (parsed.quiet) {
          break;
        }
        stdout += formatGrepOutput(vfsPath, hits, {
          filesWithMatches: parsed.filesWithMatches,
          lineNumber: parsed.lineNumber,
          multiFile: forceMultiFileOutput || vfsPaths.length > 1,
        });
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      return { stdout: '', stderr: `grep: ${vfsPath}: ${msg}\n`, exitCode: 2 };
    }
  }

  return { stdout, stderr: '', exitCode: matchedAny ? 0 : 1 };
}

/**
 * Two-stage OpenSearchFs `grep` implementation for use with `defineCommand('grep', …)` (see
 * `scripts/just-bash-opensearchfs-quickstart.ts`).
 */
export async function runOpenSearchGrep(
  args: string[],
  ctx: CommandContext,
  opensearchFs: OpenSearchFs,
  client: Client,
): Promise<ExecResult> {
  // Parse arguments
  let scannedArgs: ParsedGrepArgv;
  try {
    scannedArgs = parseGrepArgv(args);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return { stdout: '', stderr: `${msg}\n`, exitCode: 2 };
  }

  // List visible files for grep
  const scope = await listVfsFilesForGrep(
    opensearchFs,
    ctx.cwd,
    scannedArgs.fileArgs,
    scannedArgs.recursive,
  );

  // If no visible files, return early
  if (!scope.ok) {
    return { stdout: '', stderr: scope.stderr, exitCode: scope.exitCode };
  }

  // Get visible file paths
  const vfsPaths = scope.vfsPaths;
  if (vfsPaths.length === 0) {
    return { stdout: '', stderr: '', exitCode: 1 };
  }
  const shouldPrefixFilePath = vfsPaths.length > 1;

  // Build coarse filter
  const coarseFilter = {
    pattern: scannedArgs.pattern, // The raw user pattern text. Example: "OAuth.*token" (regex-like) or "OAuth token" (literal phrase).
    ignoreCase: scannedArgs.ignoreCase, // If true, matching ignores letter case. Example: pattern "OAuth" can match "oauth", "OAUTH", "oAuth".
    fixedStrings: scannedArgs.fixedStrings, // If true, metacharacters are literal text. Example: pattern "a+b" matches "a+b" (not regex "one or more a, then b").
  };
  const isRegexPattern =
    !scannedArgs.fixedStrings && hasRegexMeta(scannedArgs.pattern);

  // Coarse Filter: Ask backing store for slugs matching the string/regex
  let matchedSlugs: string[];
  try {
    matchedSlugs = await findGrepMatchingFilesWithScope(
      client,
      coarseFilter,
      scope.scopeFilter,
    );
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return isRegexPattern
      ? {
          stdout: '',
          stderr: `grep: invalid regular expression: ${msg}\n`,
          exitCode: 2,
        }
      : { stdout: '', stderr: `grep: ${msg}\n`, exitCode: 2 };
  }
  if (matchedSlugs.length === 0) return { stdout: '', stderr: '', exitCode: 1 };

  // Fine Filter: Narrow to resolved hit paths.
  const matchedPaths = matchedSlugs.map((slug) => slugToPath(slug));

  // Exec: Let the in-memory RegExp engine format the final output
  return execBuiltin(
    scannedArgs,
    matchedPaths,
    opensearchFs,
    shouldPrefixFilePath,
  );
}
