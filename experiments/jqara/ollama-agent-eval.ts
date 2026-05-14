import 'dotenv/config';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { Bash, defineCommand } from 'just-bash';
import yargsParser from 'yargs-parser';
import { runOpenSearchGrep } from '../../src/core/grep.js';
import { OpenSearchFs } from '../../src/core/opensearchfs.js';
import { runOpenSearchSemanticSearch } from '../../src/core/semantic-search.js';
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

type Arm = 'grep-only' | 'grep+semantic_search';

type ToolCallLog = {
  name: string;
  arguments: Record<string, unknown>;
  result: string;
  elapsedMs: number;
};

type EvalResult = EvalCase & {
  arm: Arm;
  model: string;
  finalAnswer: string;
  citedPaths: string[];
  toolCalls: ToolCallLog[];
  grepCalls: number;
  semanticSearchCalls: number;
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
  string: ['cases', 'profile', 'model', 'arms', 'output', 'ollama-url', 'search-path'],
  number: ['limit', 'max-tool-calls', 'temperature', 'semantic-result-limit'],
  default: {
    cases: 'data_eval/jqara/eval_cases.jsonl',
    profile: 'PUBLIC',
    model: process.env.OLLAMA_MODEL ?? 'gemma4:e4b',
    arms: 'grep-only,grep+semantic_search',
    output: 'reports/ollama-jqara-agent-eval.jsonl',
    'ollama-url': process.env.OLLAMA_BASE_URL ?? 'http://localhost:11434',
    'search-path': '/jqara/test',
    limit: 100,
    'max-tool-calls': Number(process.env.MAX_TOOL_CALLS ?? 12),
    temperature: 0,
    'semantic-result-limit': Number(process.env.SEMANTIC_SEARCH_RESULT_LIMIT ?? 5),
  },
});

const casesPath = String(args.cases);
const profile = String(args.profile);
const model = String(args.model);
const ollamaUrl = String(args['ollama-url']).replace(/\/+$/u, '');
const maxToolCalls = Number(args['max-tool-calls']);
const temperature = Number(args.temperature);
const outputPath = String(args.output);
const searchPath = String(args['search-path']);
const semanticResultLimit = Number(args['semantic-result-limit']);
const arms = parseArms(String(args.arms));
const cases = (await readCases(casesPath)).slice(
  0,
  Number(args.limit) > 0 ? Number(args.limit) : undefined,
);

if (cases.length === 0) {
  throw new Error(`No JQaRA eval cases found in ${casesPath}.`);
}

const results: EvalResult[] = [];
for (const arm of arms) {
  const previousSemanticLimit = process.env.SEMANTIC_SEARCH_RESULT_LIMIT;
  process.env.SEMANTIC_SEARCH_RESULT_LIMIT = String(semanticResultLimit);

  const client = createOpenSearchClient(profile);
  try {
    const session = await initSessionTree(client, profile);
    const fs = new OpenSearchFs({
      client,
      files: session.files,
      dirs: session.dirs,
    });
    const grep = defineCommand('grep', (commandArgs, ctx) =>
      runOpenSearchGrep(commandArgs, ctx, fs),
    );
    const customCommands = [grep];
    if (arm === 'grep+semantic_search') {
      customCommands.push(
        defineCommand('semantic_search', (commandArgs, ctx) =>
          runOpenSearchSemanticSearch(commandArgs, ctx, fs),
        ),
      );
    }
    const bash = new Bash({ fs, cwd: '/', customCommands });

    for (const testCase of cases) {
      const result = await runCase(arm, testCase, bash);
      results.push(result);
      console.log(
        `[${arm}] ${testCase.q_id}: ${result.correct ? 'correct' : 'wrong'} (${result.elapsedMs}ms)`,
      );
    }
  } finally {
    await client.close();
    restoreEnv('SEMANTIC_SEARCH_RESULT_LIMIT', previousSemanticLimit);
  }
}

await writeReports(outputPath, results);

async function runCase(
  arm: Arm,
  testCase: EvalCase,
  bash: Bash,
): Promise<EvalResult> {
  const startedAt = Date.now();
  const toolCalls: ToolCallLog[] = [];
  const messages: OllamaMessage[] = [
    { role: 'system', content: systemPrompt(arm) },
    {
      role: 'user',
      content: `次の質問に答えてください。\n質問: ${testCase.question}`,
    },
  ];

  try {
    for (let turn = 0; turn <= maxToolCalls; turn += 1) {
      const response = await chat(messages, arm);
      const message = response.message;
      if (!message) throw new Error('Ollama returned no message.');
      messages.push(message);

      const calls = message.tool_calls ?? [];
      if (calls.length === 0) {
        return buildResult(arm, testCase, message.content ?? '', toolCalls, startedAt);
      }

      if (toolCalls.length + calls.length > maxToolCalls) {
        return buildResult(
          arm,
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
      arm,
      testCase,
      '',
      toolCalls,
      startedAt,
      `Exceeded max tool calls (${maxToolCalls}).`,
    );
  } catch (error) {
    return buildResult(
      arm,
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

async function chat(
  messages: OllamaMessage[],
  arm: Arm,
): Promise<OllamaChatResponse> {
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
              arm === 'grep+semantic_search'
                ? 'Run a read-only bash command on the documentation virtual filesystem rooted at "/". Available commands include ls, cat, grep, semantic_search, pwd, cd, head, tail, and find.'
                : 'Run a read-only bash command on the documentation virtual filesystem rooted at "/". Available commands include ls, cat, grep, pwd, cd, head, tail, and find.',
            parameters: {
              type: 'object',
              required: ['command'],
              properties: {
                command: {
                  type: 'string',
                  description:
                    arm === 'grep+semantic_search'
                      ? 'Bash command to execute, for example: ls /, grep -ri "検索語" /jqara/test, semantic_search "自然文の質問" /jqara/test, or cat <path>'
                      : 'Bash command to execute, for example: ls /, grep -ri "検索語" /jqara/test, or cat <path>',
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
  arm: Arm,
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
  const citationCorrect = testCase.positivePaths.some(
    (pathValue) => finalAnswer.includes(pathValue) || citedPaths.includes(pathValue),
  );
  const catPaths = extractCatPaths(toolCalls);
  return {
    ...testCase,
    arm,
    model,
    finalAnswer,
    citedPaths,
    toolCalls,
    grepCalls: toolCalls.filter((call) =>
      String(call.arguments.command ?? '').trim().startsWith('grep '),
    ).length,
    semanticSearchCalls: toolCalls.filter((call) =>
      String(call.arguments.command ?? '').trim().startsWith('semantic_search '),
    ).length,
    catPaths,
    elapsedMs: Date.now() - startedAt,
    answerCorrect,
    citationCorrect,
    correct: answerCorrect && citationCorrect && !error,
    ...(error ? { error } : {}),
  };
}

function systemPrompt(arm: Arm): string {
  const semanticInstruction =
    arm === 'grep+semantic_search'
      ? `\n- semantic_search "<natural language query>" ${searchPath} is available when exact keywords are uncertain, the question is phrased naturally, or grep does not find enough candidates. You do not need to start with semantic_search if grep or directory exploration is more appropriate.\n- Do not finalize from a weak candidate file. If the files found by grep do not directly answer the question, try semantic_search before giving the final answer.`
      : '';
  return `You are a Japanese QA assistant with access to a virtual filesystem of JQaRA passages via the docs_bash tool.
- You must inspect the filesystem before answering.
- Start with ls / and search under ${searchPath}.
- Use grep -ri "<term>" ${searchPath} for keyword search.${semanticInstruction}
- You must use cat <path> to read at least one candidate file before giving the final answer.
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
    '# Ollama JQaRA agent eval',
    '',
    `model: \`${model}\``,
    '',
    '| arm | cases | correct | answer | citation | avg tools | avg grep | avg semantic | avg cat paths | avg ms |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
  ];
  for (const row of summarize(rows)) {
    lines.push(
      `| ${row.arm} | ${row.cases} | ${percent(row.correct, row.cases)} | ${percent(row.answerCorrect, row.cases)} | ${percent(row.citationCorrect, row.cases)} | ${row.avgTools.toFixed(1)} | ${row.avgGrep.toFixed(1)} | ${row.avgSemantic.toFixed(1)} | ${row.avgCatPaths.toFixed(1)} | ${Math.round(row.avgMs)} |`,
    );
  }
  lines.push('', '## Cases', '');
  lines.push('| q_id | arm | ok | positive paths | answers | cited paths | final answer |');
  lines.push('|---|---:|---:|---|---|---|---|');
  for (const row of rows) {
    lines.push(
      `| ${escapeMd(row.q_id)} | ${row.arm} | ${row.correct ? 'yes' : 'no'} | ${escapeMd(row.positivePaths.join('<br>'))} | ${escapeMd(row.answers.join(' / '))} | ${escapeMd(row.citedPaths.join('<br>') || '(none)')} | ${escapeMd(row.finalAnswer.slice(0, 180))} |`,
    );
  }
  lines.push('');
  return lines.join('\n');
}

function summarize(rows: EvalResult[]): Array<{
  arm: Arm;
  cases: number;
  correct: number;
  answerCorrect: number;
  citationCorrect: number;
  avgTools: number;
  avgGrep: number;
  avgSemantic: number;
  avgCatPaths: number;
  avgMs: number;
}> {
  const keys = new Map<Arm, EvalResult[]>();
  for (const row of rows) {
    keys.set(row.arm, [...(keys.get(row.arm) ?? []), row]);
  }
  return [...keys.entries()].map(([arm, group]) => ({
    arm,
    cases: group.length,
    correct: group.filter((row) => row.correct).length,
    answerCorrect: group.filter((row) => row.answerCorrect).length,
    citationCorrect: group.filter((row) => row.citationCorrect).length,
    avgTools:
      group.reduce((sum, row) => sum + row.toolCalls.length, 0) / group.length,
    avgGrep: group.reduce((sum, row) => sum + row.grepCalls, 0) / group.length,
    avgSemantic:
      group.reduce((sum, row) => sum + row.semanticSearchCalls, 0) /
      group.length,
    avgCatPaths:
      group.reduce((sum, row) => sum + row.catPaths.length, 0) / group.length,
    avgMs: group.reduce((sum, row) => sum + row.elapsedMs, 0) / group.length,
  }));
}

function parseArms(value: string): Arm[] {
  const arms = value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
  for (const arm of arms) {
    if (arm !== 'grep-only' && arm !== 'grep+semantic_search') {
      throw new Error(`Invalid arm: ${arm}`);
    }
  }
  return arms as Arm[];
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

function restoreEnv(key: string, previous: string | undefined): void {
  if (previous === undefined) {
    delete process.env[key];
  } else {
    process.env[key] = previous;
  }
}
