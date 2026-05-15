import 'dotenv/config';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { Bash, defineCommand } from 'just-bash';
import yargsParser from 'yargs-parser';
import { runOpenSearchGrep } from '../../src/core/grep.js';
import { OpenSearchFs } from '../../src/core/opensearchfs.js';
import { createOpenSearchClient } from '../../src/opensearch-adapter/client.js';
import { initSessionTree } from '../../src/session.js';

type EvalMode = 'lexical';

type EvalCase = {
  id: string;
  dataset: string;
  question: string;
  answers: string[];
  expectedPath: string;
  title: string;
};

type ToolCallLog = {
  name: string;
  arguments: Record<string, unknown>;
  result: string;
  elapsedMs: number;
};

type EvalResult = EvalCase & {
  mode: EvalMode;
  model: string;
  finalAnswer: string;
  citedPaths: string[];
  toolCalls: ToolCallLog[];
  grepCalls: number;
  catPaths: string[];
  elapsedMs: number;
  answerCorrect: boolean;
  citationCorrect: boolean;
  correct: boolean;
  error?: string;
};

type OllamaToolCall = {
  function?: {
    name?: string;
    arguments?: unknown;
  };
};

type OllamaMessage = {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content?: string;
  thinking?: string;
  tool_name?: string;
  tool_calls?: OllamaToolCall[];
};

type OllamaChatResponse = {
  message?: OllamaMessage;
  done?: boolean;
};

const args = yargsParser(process.argv.slice(2), {
  string: ['cases', 'profile', 'model', 'output', 'ollama-url', 'dataset'],
  number: ['limit', 'max-tool-calls', 'temperature'],
  default: {
    cases: 'data_eval/qa/eval_cases.jsonl',
    profile: 'PUBLIC',
    model: process.env.OLLAMA_MODEL ?? 'gemma4:e4b',
    output: 'reports/ollama-qa-eval.jsonl',
    'ollama-url': process.env.OLLAMA_BASE_URL ?? 'http://localhost:11434',
    dataset: '',
    limit: 0,
    'max-tool-calls': Number(process.env.MAX_TOOL_CALLS ?? 12),
    temperature: 0,
  },
});

const casesPath = String(args.cases);
const profile = String(args.profile);
const model = String(args.model);
const ollamaUrl = String(args['ollama-url']).replace(/\/+$/u, '');
const maxToolCalls = Number(args['max-tool-calls']);
const temperature = Number(args.temperature);
const outputPath = String(args.output);
const mode: EvalMode = 'lexical';
const datasetFilter = String(args.dataset).trim();
const cases = (await readCases(casesPath))
  .filter((testCase) => !datasetFilter || testCase.dataset === datasetFilter)
  .slice(
  0,
  Number(args.limit) > 0 ? Number(args.limit) : undefined,
  );

if (cases.length === 0) {
  throw new Error(`No eval cases found in ${casesPath}.`);
}

const results: EvalResult[] = [];
const client = createOpenSearchClient(profile);
try {
  const session = await initSessionTree(client, profile);
  const fs = new OpenSearchFs({
    client,
    files: session.files,
    dirs: session.dirs,
  });
  const grep = defineCommand('grep', (commandArgs, ctx) =>
    runOpenSearchGrep(commandArgs, ctx, fs, client),
  );
  const bash = new Bash({ fs, cwd: '/', customCommands: [grep] });

  for (const testCase of cases) {
    const result = await runCase(mode, testCase, bash);
    results.push(result);
    console.log(
      `[${mode}] ${testCase.dataset}/${testCase.id}: ${result.correct ? 'correct' : 'wrong'} (${result.elapsedMs}ms)`,
    );
  }
} finally {
  await client.close();
}

await writeReports(outputPath, results);

async function runCase(
  mode: EvalMode,
  testCase: EvalCase,
  bash: Bash,
): Promise<EvalResult> {
  const startedAt = Date.now();
  const toolCalls: ToolCallLog[] = [];
  const messages: OllamaMessage[] = [
    { role: 'system', content: systemPrompt() },
    {
      role: 'user',
      content: `次の質問に答えてください。\n質問: ${testCase.question}`,
    },
  ];

  try {
    for (let turn = 0; turn <= maxToolCalls; turn += 1) {
      const response = await chat(messages);
      const message = response.message;
      if (!message) throw new Error('Ollama returned no message.');
      messages.push(message);

      const calls = message.tool_calls ?? [];
      if (calls.length === 0) {
        return buildResult(mode, testCase, message.content ?? '', toolCalls, startedAt);
      }

      if (toolCalls.length + calls.length > maxToolCalls) {
        return buildResult(
          mode,
          testCase,
          message.content ?? '',
          toolCalls,
          startedAt,
          `Exceeded max tool calls (${maxToolCalls}).`,
        );
      }

      for (const call of calls) {
        const name = call.function?.name ?? '';
        const callArgs = parseToolArguments(call.function?.arguments);
        const toolStartedAt = Date.now();
        const result =
          name === 'docs_bash'
            ? await runDocsBash(bash, String(callArgs.command ?? ''))
            : `Unknown tool: ${name}`;
        toolCalls.push({
          name,
          arguments: callArgs,
          result,
          elapsedMs: Date.now() - toolStartedAt,
        });
        messages.push({
          role: 'tool',
          tool_name: name,
          content: result,
        });
      }
    }

    return buildResult(
      mode,
      testCase,
      '',
      toolCalls,
      startedAt,
      `Exceeded max tool calls (${maxToolCalls}).`,
    );
  } catch (error) {
    return buildResult(
      mode,
      testCase,
      '',
      toolCalls,
      startedAt,
      error instanceof Error ? error.message : String(error),
    );
  }
}

async function runDocsBash(bash: Bash, command: string): Promise<string> {
  if (command.trim().length === 0) {
    return 'exitCode: 2\n[stderr]\nempty command';
  }
  const { stdout, stderr, exitCode } = await bash.exec(command);
  const text =
    `exitCode: ${exitCode}\n` +
    (stdout || '(empty stdout)') +
    (stderr ? `\n[stderr]\n${stderr}` : '');
  return clipToolOutput(text, Number(process.env.OLLAMA_QA_MAX_TOOL_OUTPUT_CHARS ?? 8000));
}

async function chat(messages: OllamaMessage[]): Promise<OllamaChatResponse> {
  const response = await fetch(`${ollamaUrl}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      messages,
      stream: false,
      think: false,
      options: { temperature },
      tools: [
        {
          type: 'function',
          function: {
            name: 'docs_bash',
            description:
              'Run a read-only bash command on the documentation virtual filesystem rooted at "/". Available commands include ls, cat, grep, pwd, cd, head, tail, and find.',
            parameters: {
              type: 'object',
              required: ['command'],
              properties: {
                command: {
                  type: 'string',
                  description: 'Bash command to execute, for example: ls / or grep -ri "検索語" /qa',
                },
              },
            },
          },
        },
      ],
    }),
  });
  if (!response.ok) {
    throw new Error(`Ollama chat failed: ${response.status} ${await response.text()}`);
  }
  return (await response.json()) as OllamaChatResponse;
}

function buildResult(
  mode: EvalMode,
  testCase: EvalCase,
  finalAnswer: string,
  toolCalls: ToolCallLog[],
  startedAt: number,
  error?: string,
): EvalResult {
  const citedPaths = extractCitedPaths(finalAnswer);
  const answerCorrect = testCase.answers.some((answer) =>
    normalizeForMatch(finalAnswer).includes(normalizeForMatch(answer)),
  );
  const citationCorrect =
    finalAnswer.includes(testCase.expectedPath) ||
    citedPaths.includes(testCase.expectedPath);
  const catPaths = extractCatPaths(toolCalls);
  return {
    ...testCase,
    mode,
    model,
    finalAnswer,
    citedPaths,
    toolCalls,
    grepCalls: toolCalls.filter((call) =>
      String(call.arguments.command ?? '').trim().startsWith('grep '),
    ).length,
    catPaths,
    elapsedMs: Date.now() - startedAt,
    answerCorrect,
    citationCorrect,
    correct: answerCorrect && citationCorrect && !error,
    ...(error ? { error } : {}),
  };
}

function systemPrompt(): string {
  return `You are a Japanese QA assistant with access to a virtual filesystem of QA documents via the docs_bash tool.
- You must inspect the filesystem before answering.
- Start with ls / and use grep -ri "<term>" /qa to search.
- You must use cat <path> to read at least one candidate file before giving the final answer.
- The only search command you should rely on is grep; its internal implementation may change between runs.
- Answer in Japanese.
- Include the exact evidence file path you used in the final answer.
- If the answer is not supported by the files you can access, say that you could not find enough evidence.`;
}

async function readCases(filePath: string): Promise<EvalCase[]> {
  const raw = await readFile(filePath, 'utf8');
  return raw
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as EvalCase);
}

async function writeReports(jsonlPath: string, rows: EvalResult[]): Promise<void> {
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

function toMarkdown(rows: EvalResult[]): string {
  const lines = [
    '# Ollama QA eval',
    '',
    `model: \`${model}\``,
    '',
    '| dataset | mode | cases | correct | answer | citation | avg tools | avg ms |',
    '|---|---:|---:|---:|---:|---:|---:|---:|',
  ];
  for (const row of summarize(rows)) {
    lines.push(
      `| ${row.dataset} | ${row.mode} | ${row.cases} | ${percent(row.correct, row.cases)} | ${percent(row.answerCorrect, row.cases)} | ${percent(row.citationCorrect, row.cases)} | ${row.avgTools.toFixed(1)} | ${Math.round(row.avgMs)} |`,
    );
  }
  lines.push('', '## Cases', '');
  lines.push('| dataset | id | mode | ok | expected path | answers | cited paths | final answer |');
  lines.push('|---|---|---:|---:|---|---|---|---|');
  for (const row of rows) {
    lines.push(
      `| ${escapeMd(row.dataset)} | ${escapeMd(row.id)} | ${row.mode} | ${row.correct ? 'yes' : 'no'} | ${escapeMd(row.expectedPath)} | ${escapeMd(row.answers.join(' / '))} | ${escapeMd(row.citedPaths.join('<br>') || '(none)')} | ${escapeMd(row.finalAnswer.slice(0, 180))} |`,
    );
  }
  lines.push('');
  return lines.join('\n');
}

function summarize(rows: EvalResult[]): Array<{
  dataset: string;
  mode: EvalMode;
  cases: number;
  correct: number;
  answerCorrect: number;
  citationCorrect: number;
  avgTools: number;
  avgMs: number;
}> {
  const keys = new Map<string, EvalResult[]>();
  for (const row of rows) {
    const key = `${row.dataset}\t${row.mode}`;
    keys.set(key, [...(keys.get(key) ?? []), row]);
  }
  return [...keys.entries()].map(([key, group]) => {
    const [dataset, mode] = key.split('\t') as [string, EvalMode];
    return {
      dataset,
      mode,
      cases: group.length,
      correct: group.filter((row) => row.correct).length,
      answerCorrect: group.filter((row) => row.answerCorrect).length,
      citationCorrect: group.filter((row) => row.citationCorrect).length,
      avgTools:
        group.reduce((sum, row) => sum + row.toolCalls.length, 0) / group.length,
      avgMs: group.reduce((sum, row) => sum + row.elapsedMs, 0) / group.length,
    };
  });
}

function parseToolArguments(value: unknown): Record<string, unknown> {
  if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  if (typeof value === 'string' && value.trim()) {
    try {
      const parsed = JSON.parse(value) as unknown;
      if (parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      return {};
    }
  }
  return {};
}

function extractCitedPaths(text: string): string[] {
  const paths = new Set<string>();
  for (const match of text.matchAll(/\/[^\s"'`),。、]+\.mdx/gu)) {
    paths.add(match[0]);
  }
  return [...paths];
}

function extractCatPaths(toolCalls: ToolCallLog[]): string[] {
  const paths = new Set<string>();
  for (const call of toolCalls) {
    const command = String(call.arguments.command ?? '').trim();
    const match = command.match(/^cat\s+(.+)$/u);
    if (match?.[1]) paths.add(match[1].trim());
  }
  return [...paths];
}

function normalizeForMatch(value: string): string {
  return value
    .normalize('NFKC')
    .toLowerCase()
    .replace(/\s+/gu, '');
}

function percent(numerator: number, denominator: number): string {
  if (denominator === 0) return '0.0%';
  return `${((numerator / denominator) * 100).toFixed(1)}%`;
}

function escapeMd(value: string): string {
  return value.replace(/\|/gu, '\\|').replace(/\n/gu, '<br>');
}

function clipToolOutput(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  return `${text.slice(0, maxChars)}\n...[truncated ${text.length - maxChars} chars]`;
}
