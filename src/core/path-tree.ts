import { readFile } from 'node:fs/promises';
import path from 'node:path/posix';

export type JsonObject = Record<string, unknown>;

type PathTreeEntry = {
  isPublic: boolean;
  groups: string[];
};

export type PathTreePolicy = Record<string, PathTreeEntry>;

export type CompiledAccessPlan = {
  publicSlugs: string[];
  groupSlugs: Record<string, string[]>;
};

/** Normalises backslashes to forward slashes, collapses repeated slashes, and strips a leading `./` and any trailing slashes. */
function normalizeSlashesAndDots(input: string): string {
  return input
    .replace(/\\/g, '/')
    .replace(/\/+/g, '/')
    .replace(/^\.\//, '')
    .replace(/\/+$/g, '');
}

/*
 * Normalize a slug to a relative slug.
 */
export function normalizeSlug(slug: string): string {
  const normalized = normalizeSlashesAndDots(slug);
  const absolute =
    normalized === ''
      ? '/'
      : normalized.startsWith('/')
        ? normalized
        : `/${normalized}`;
  if (absolute === '/') {
    throw new Error('Invalid slug: empty string.');
  }
  return absolute.slice(1);
}

/**
 * Normalize to a generic absolute path.
 */
export function normalizePath(filePath: string): string {
  const normalized = normalizeSlashesAndDots(filePath);
  return normalized === ''
    ? '/'
    : normalized.startsWith('/')
      ? normalized
      : `/${normalized}`;
}

/**
 * Convert an .mdx path back to its slug.
 * "/blog/hello.mdx"   -> "blog/hello"
 */
export function pathToSlug(filePath: string): string {
  const normalized = normalizePath(filePath);
  if (!normalized.endsWith('.mdx')) {
    throw new Error(`Path must end with .mdx: "${filePath}"`);
  }
  return normalized.slice(1, -'.mdx'.length);
}

/**
 * Convert a slug to its corresponding absolute .mdx path.
 * "blog/hello" -> "/blog/hello.mdx"
 */
export function slugToPath(slug: string): string {
  const normalizedSlug = normalizeSlug(slug);
  return normalizedSlug.endsWith('.mdx')
    ? `/${normalizedSlug}`
    : `/${normalizedSlug}.mdx`;
}

/**
 * Builds an in-memory file tree from a collection of slugs.
 * Returns a `files` set of absolute `.mdx` paths and a `dirs` map of directory path → sorted child names,
 * with intermediate parent directories synthesised automatically.
 */
export function buildFileTreeFromSlugs(slugs: Iterable<string>): {
  files: Set<string>;
  dirs: Map<string, string[]>;
} {
  const files = new Set<string>();
  const dirToChildren = new Map<string, Set<string>>();

  function addChild(dir: string, name: string): void {
    const d = dir === '' ? '/' : normalizePath(dir);
    let set = dirToChildren.get(d);
    if (!set) {
      set = new Set();
      dirToChildren.set(d, set);
    }
    set.add(name);
  }

  for (const rawSlug of slugs) {
    const fp = slugToPath(rawSlug);
    files.add(fp);

    const base = path.basename(fp);
    const dir = path.dirname(fp);
    addChild(dir, base);

    let current = dir;
    while (current !== '/' && current !== '') {
      const parent = path.dirname(current);
      const seg = path.basename(current);
      const p = parent === '' || parent === '.' ? '/' : parent;
      addChild(p, seg);
      current = p;
    }
  }

  const dirs = new Map<string, string[]>();
  for (const [d, children] of dirToChildren) {
    dirs.set(d, [...children].sort());
  }

  return { files, dirs };
}

/** Strips trailing commas before object/array closers, then parses as JSON. Allows lax JSON that editors commonly produce. */
function parseJsonWithTrailingCommaSupport(raw: string): unknown {
  const sanitized = raw.replace(/,\s*([}\]])/g, '$1');
  return JSON.parse(sanitized) as unknown;
}

/** Validates and coerces a raw JSON value into a path tree policy entry. */
function asPathTreeEntry(value: unknown, slug: string): PathTreeEntry {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`Invalid path tree entry for "${slug}": expected object.`);
  }
  const obj = value as JsonObject;
  if (typeof obj.isPublic !== 'boolean') {
    throw new Error(
      `Invalid path tree entry for "${slug}": "isPublic" must be boolean.`,
    );
  }
  if (
    !Array.isArray(obj.groups) ||
    obj.groups.some((entry) => typeof entry !== 'string')
  ) {
    throw new Error(
      `Invalid path tree entry for "${slug}": "groups" must be string[].`,
    );
  }
  const groups = obj.groups.map((group) => group.trim()).filter(Boolean);
  return { isPublic: obj.isPublic, groups: [...new Set(groups)].sort() };
}

/** Reads a JSON file from disk and parses it as a `PathTreePolicy`. Accepts trailing commas. */
export async function loadPathTreeAccessPolicy(
  path: string,
): Promise<PathTreePolicy> {
  const raw = await readFile(path, 'utf8');
  const parsed = parseJsonWithTrailingCommaSupport(raw);
  return parsePathTreeAccessPolicy(parsed);
}

/** Validates a parsed JSON value as a `PathTreePolicy`, normalising each slug via `normalizeSlug`. */
export function parsePathTreeAccessPolicy(parsed: unknown): PathTreePolicy {
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('Path tree policy must be an object.');
  }

  const out: PathTreePolicy = {};
  for (const [rawSlug, value] of Object.entries(parsed as JsonObject)) {
    const slug = normalizeSlug(rawSlug);
    out[slug] = asPathTreeEntry(value, slug);
  }
  return out;
}

/**
 * Derives per-profile slug assignments from a `PathTreePolicy`.
 * Each group slug list is the union of group slugs and all public slugs,
 * so a group profile can always read public content.
 */
export function compileAccessPlanFromPolicy(
  policy: PathTreePolicy,
): CompiledAccessPlan {
  const publicSet = new Set<string>();
  const groupSets = new Map<string, Set<string>>();

  for (const [slug, entry] of Object.entries(policy)) {
    if (entry.isPublic) {
      publicSet.add(slug);
    }
    for (const group of entry.groups) {
      let set = groupSets.get(group);
      if (!set) {
        set = new Set();
        groupSets.set(group, set);
      }
      set.add(slug);
    }
  }

  const publicSlugs = [...publicSet].sort();
  const groupSlugs: Record<string, string[]> = {};
  for (const [group, slugs] of groupSets) {
    const merged = new Set<string>(publicSet);
    for (const slug of slugs) merged.add(slug);
    groupSlugs[group] = [...merged].sort();
  }

  return { publicSlugs, groupSlugs };
}
