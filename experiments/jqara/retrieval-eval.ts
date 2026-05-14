import 'dotenv/config';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import yargsParser from 'yargs-parser';
import type { GrepCoarseFilter } from '../../src/core/opensearchfs.js';
import { OpenSearchFs, type SearchScopeFilter } from '../../src/core/opensearchfs.js';
import { normalizePath, pathToSlug, slugToPath } from '../../src/core/path-tree.js';
import { createOpenSearchClient } from '../../src/opensearch-adapter/client.js';
import { initSessionTree } from '../../src/session.js';

type EvalCase = {
  id: string;
  q_id: string;
  dataset: 'jqara';
  question: string;
  answers: string[];
  positivePaths: string[];
};

type Arm = 'lexical' | 'semantic_search';

type EvalRow = EvalCase & {
  arm: Arm;
  retrievedPaths: string[];
  ndcgAt10: number;
  mrrAt10: number;
  recallAt1: number;
  recallAt5: number;
  recallAt10: number;
  elapsedMs: number;
};

const args = yargsParser(process.argv.slice(2), {
  string: ['cases', 'profile', 'arms', 'output', 'path'],
  number: ['limit', 'result-limit'],
  default: {
    cases: 'data_eval/jqara/eval_cases.jsonl',
    profile: 'PUBLIC',
    arms: 'lexical,semantic_search',
    output: 'reports/jqara-retrieval-eval.jsonl',
    path: '/jqara/test',
    limit: 0,
    'result-limit': 10,
  },
});

const profile = String(args.profile);
const casesPath = String(args.cases);
const outputPath = String(args.output);
const searchPath = String(args.path);
const resultLimit = Number(args['result-limit']);
const arms = parseArms(String(args.arms));
const cases = (await readCases(casesPath)).slice(
  0,
  Number(args.limit) > 0 ? Number(args.limit) : undefined,
);
if (cases.length === 0) {
  throw new Error(`No JQaRA eval cases found in ${casesPath}.`);
}

const client = createOpenSearchClient(profile);
const rows: EvalRow[] = [];
try {
  const session = await initSessionTree(client, profile);
  const fs = new OpenSearchFs({
    client,
    files: session.files,
    dirs: session.dirs,
  });
  const scopeFilter = await resolveScopeFilter(fs, searchPath);

  for (const arm of arms) {
    for (const testCase of cases) {
      const startedAt = Date.now();
      const retrievedPaths = await runRetrievalArm(
        fs,
        arm,
        testCase.question,
        scopeFilter,
        resultLimit,
      );
      const row = buildRow(
        testCase,
        arm,
        retrievedPaths,
        Date.now() - startedAt,
      );
      rows.push(row);
      console.log(
        `[${arm}] ${testCase.q_id}: nDCG@10=${row.ndcgAt10.toFixed(3)} MRR@10=${row.mrrAt10.toFixed(3)}`,
      );
    }
  }
} finally {
  await client.close();
}

await writeReports(outputPath, rows);

async function runRetrievalArm(
  fs: OpenSearchFs,
  arm: Arm,
  question: string,
  scopeFilter: SearchScopeFilter,
  limit: number,
): Promise<string[]> {
  if (arm === 'semantic_search') {
    const hits = await fs.findSemanticMatchingFilesWithScope(
      question,
      scopeFilter,
      limit,
    );
    return hits.map((hit) => slugToPath(hit.slug));
  }

  const coarseFilter: GrepCoarseFilter = {
    pattern: question,
    ignoreCase: true,
    fixedStrings: true,
  };
  const slugs = await fs.findMatchingFilesWithScope(coarseFilter, scopeFilter);
  return slugs.slice(0, limit).map((slug) => slugToPath(slug));
}

async function resolveScopeFilter(
  fs: OpenSearchFs,
  pathArg: string,
): Promise<SearchScopeFilter> {
  const abs = normalizePath(fs.resolvePath('/', pathArg));
  if (!(await fs.exists(abs))) {
    throw new Error(`Search path does not exist for profile ${profile}: ${pathArg}`);
  }
  const stat = await fs.stat(abs);
  if (stat.isFile) return { term: { slug: pathToSlug(abs) } };
  if (stat.isDirectory) {
    return abs === '/' ? undefined : { prefix: { slug: `${abs.slice(1)}/` } };
  }
  throw new Error(`Unsupported search path type: ${pathArg}`);
}

function buildRow(
  testCase: EvalCase,
  arm: Arm,
  retrievedPaths: string[],
  elapsedMs: number,
): EvalRow {
  return {
    ...testCase,
    arm,
    retrievedPaths,
    ndcgAt10: ndcgAtK(retrievedPaths, testCase.positivePaths, 10),
    mrrAt10: mrrAtK(retrievedPaths, testCase.positivePaths, 10),
    recallAt1: recallAtK(retrievedPaths, testCase.positivePaths, 1),
    recallAt5: recallAtK(retrievedPaths, testCase.positivePaths, 5),
    recallAt10: recallAtK(retrievedPaths, testCase.positivePaths, 10),
    elapsedMs,
  };
}

function ndcgAtK(
  retrievedPaths: string[],
  positivePaths: string[],
  k: number,
): number {
  const positives = new Set(positivePaths);
  let dcg = 0;
  for (const [index, pathValue] of retrievedPaths.slice(0, k).entries()) {
    if (positives.has(pathValue)) {
      dcg += 1 / Math.log2(index + 2);
    }
  }
  const idealHits = Math.min(positivePaths.length, k);
  let idcg = 0;
  for (let index = 0; index < idealHits; index += 1) {
    idcg += 1 / Math.log2(index + 2);
  }
  return idcg === 0 ? 0 : dcg / idcg;
}

function mrrAtK(
  retrievedPaths: string[],
  positivePaths: string[],
  k: number,
): number {
  const positives = new Set(positivePaths);
  const index = retrievedPaths
    .slice(0, k)
    .findIndex((pathValue) => positives.has(pathValue));
  return index === -1 ? 0 : 1 / (index + 1);
}

function recallAtK(
  retrievedPaths: string[],
  positivePaths: string[],
  k: number,
): number {
  if (positivePaths.length === 0) return 0;
  const retrieved = new Set(retrievedPaths.slice(0, k));
  const found = positivePaths.filter((pathValue) => retrieved.has(pathValue)).length;
  return found / positivePaths.length;
}

async function readCases(filePath: string): Promise<EvalCase[]> {
  const raw = await readFile(filePath, 'utf8');
  return raw
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as EvalCase);
}

async function writeReports(jsonlPath: string, rows: EvalRow[]): Promise<void> {
  const resolvedJsonlPath = path.resolve(jsonlPath);
  await mkdir(path.dirname(resolvedJsonlPath), { recursive: true });
  await writeFile(
    resolvedJsonlPath,
    `${rows.map((row) => JSON.stringify(row)).join('\n')}\n`,
    'utf8',
  );
  const mdPath = resolvedJsonlPath.replace(/\.jsonl$/u, '.md');
  await writeFile(mdPath, toMarkdown(rows), 'utf8');
  console.log(`Wrote ${resolvedJsonlPath}`);
  console.log(`Wrote ${mdPath}`);
}

function toMarkdown(rows: EvalRow[]): string {
  const lines = [
    '# JQaRA retrieval eval',
    '',
    `search path: \`${searchPath}\``,
    '',
    '| arm | cases | nDCG@10 | MRR@10 | Recall@1 | Recall@5 | Recall@10 | avg ms |',
    '|---|---:|---:|---:|---:|---:|---:|---:|',
  ];
  for (const row of summarize(rows)) {
    lines.push(
      `| ${row.arm} | ${row.cases} | ${row.ndcgAt10.toFixed(3)} | ${row.mrrAt10.toFixed(3)} | ${row.recallAt1.toFixed(3)} | ${row.recallAt5.toFixed(3)} | ${row.recallAt10.toFixed(3)} | ${Math.round(row.avgMs)} |`,
    );
  }
  lines.push('', '## Cases', '');
  lines.push('| q_id | arm | nDCG@10 | MRR@10 | positives | top hits |');
  lines.push('|---|---:|---:|---:|---|---|');
  for (const row of rows) {
    lines.push(
      `| ${escapeMd(row.q_id)} | ${row.arm} | ${row.ndcgAt10.toFixed(3)} | ${row.mrrAt10.toFixed(3)} | ${escapeMd(row.positivePaths.join('<br>'))} | ${escapeMd(row.retrievedPaths.slice(0, 5).join('<br>'))} |`,
    );
  }
  lines.push('');
  return lines.join('\n');
}

function summarize(rows: EvalRow[]): Array<{
  arm: Arm;
  cases: number;
  ndcgAt10: number;
  mrrAt10: number;
  recallAt1: number;
  recallAt5: number;
  recallAt10: number;
  avgMs: number;
}> {
  const grouped = new Map<Arm, EvalRow[]>();
  for (const row of rows) {
    grouped.set(row.arm, [...(grouped.get(row.arm) ?? []), row]);
  }
  return [...grouped.entries()].map(([arm, group]) => ({
    arm,
    cases: group.length,
    ndcgAt10: average(group.map((row) => row.ndcgAt10)),
    mrrAt10: average(group.map((row) => row.mrrAt10)),
    recallAt1: average(group.map((row) => row.recallAt1)),
    recallAt5: average(group.map((row) => row.recallAt5)),
    recallAt10: average(group.map((row) => row.recallAt10)),
    avgMs: average(group.map((row) => row.elapsedMs)),
  }));
}

function average(values: number[]): number {
  return values.length === 0
    ? 0
    : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function parseArms(value: string): Arm[] {
  const arms = value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
  for (const arm of arms) {
    if (arm !== 'lexical' && arm !== 'semantic_search') {
      throw new Error(`Invalid arm: ${arm}`);
    }
  }
  return arms as Arm[];
}

function escapeMd(value: string): string {
  return value.replace(/\|/gu, '\\|').replace(/\n/gu, '<br>');
}
