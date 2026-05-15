import 'dotenv/config';
import { appendFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { Bash, defineCommand } from 'just-bash';
import yargsParser from 'yargs-parser';
import { runOpenSearchGrep } from '../../src/core/grep.js';
import { OpenSearchFs } from '../../src/core/opensearchfs.js';
import {
  OpenSearchSemanticSearcher,
  runOpenSearchSemanticSearch,
} from '../../src/core/semantic-search.js';
import {
  documentRecall as calculateDocumentRecall,
  extractDocumentIdsFromPaths,
  invalidExtraDocCount,
  parseJudgeResponseText,
  type FactJudgment,
} from './metrics.js';
import { createOpenSearchClient } from '../../src/opensearch-adapter/client.js';
import { initSessionTree } from '../../src/session.js';

type EvalCase = {
  id: string;
  question_id: string;
  dataset: 'enterprise-rag-bench';
  questionType: string;
  sourceTypes: string[];
  question: string;
  goldAnswer: string;
  answerFacts: string[];
  expectedDocIds: string[];
  positivePaths: string[];
};

type Arm = 'grep-only' | 'grep+semantic_search';

type ToolCallLog = {
  turn: number;
  name: string;
  arguments: Record<string, unknown>;
  result: string;
  resultRawLength: number;
  resultTruncated: boolean;
  elapsedMs: number;
};

type AssistantTurnLog = {
  turn: number;
  content: string;
  thinking?: string;
  toolCalls: Array<{
    name: string;
    arguments: Record<string, unknown>;
  }>;
};

type JudgeResult = {
  answer_correct: boolean;
  fact_judgments: FactJudgment[];
  completeness: number;
  reason: string;
  prompt: {
    system: string;
    user: string;
  };
  rawResponse: string;
  error?: string;
};

type EvalResult = EvalCase & {
  arm: Arm;
  model: string;
  judgeModel: string;
  answer: string;
  finalAnswer: string;
  citedPaths: string[];
  document_ids: string[];
  readDocumentIds: string[];
  documentRecall: number;
  invalidExtraDocs: number;
  toolCalls: ToolCallLog[];
  grepCalls: number;
  semanticSearchCalls: number;
  catPaths: string[];
  elapsedMs: number;
  answerFactRecall: number;
  answerCompleteness: number;
  answerCorrect: boolean;
  overallScore: number;
  citationCorrect: boolean;
  judgeResult: JudgeResult;
  judgeError?: string;
  correct: boolean;
  error?: string;
};

type TrajectoryRecord = {
  question_id: string;
  arm: Arm;
  model: string;
  judgeModel: string;
  questionType: string;
  sourceTypes: string[];
  question: string;
  goldAnswer: string;
  answerFacts: string[];
  expectedDocIds: string[];
  positivePaths: string[];
  prompts: {
    system: string;
    user: string;
  };
  assistantTurns: AssistantTurnLog[];
  toolCalls: ToolCallLog[];
  judgeResult: JudgeResult;
  finalAnswer: string;
  citedPaths: string[];
  document_ids: string[];
  readDocumentIds: string[];
  documentRecall: number;
  invalidExtraDocs: number;
  catPaths: string[];
  grepCalls: number;
  semanticSearchCalls: number;
  elapsedMs: number;
  answerFactRecall: number;
  answerCompleteness: number;
  answerCorrect: boolean;
  overallScore: number;
  citationCorrect: boolean;
  judgeError?: string;
  correct: boolean;
  error?: string;
};

type RunCaseOutput = {
  result: EvalResult;
  trajectory: TrajectoryRecord;
};

type DocsBashResult = {
  text: string;
  rawLength: number;
  truncated: boolean;
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

const rawArgv = process.argv.slice(2);
const args = yargsParser(rawArgv, {
  string: [
    'cases',
    'profile',
    'model',
    'arms',
    'output',
    'trajectory-output',
    'ollama-url',
    'search-path',
    'judge-model',
  ],
  number: [
    'limit',
    'max-tool-calls',
    'temperature',
    'semantic-result-limit',
    'judge-temperature',
  ],
  default: {
    cases: 'data_eval/enterprise_rag/eval_cases.jsonl',
    profile: 'PUBLIC',
    model: process.env.OLLAMA_MODEL ?? 'gemma4:e4b',
    arms: 'grep-only,grep+semantic_search',
    output: 'reports/ollama-enterprise-rag-agent-eval.jsonl',
    'ollama-url': process.env.OLLAMA_BASE_URL ?? 'http://localhost:11434',
    'judge-model':
      process.env.OLLAMA_JUDGE_MODEL ??
      process.env.OLLAMA_MODEL,
    'search-path': '/enterprise-rag/test',
    limit: 100,
    'max-tool-calls': Number(process.env.MAX_TOOL_CALLS ?? 12),
    temperature: 0,
    'judge-temperature': 0,
    'semantic-result-limit': Number(process.env.SEMANTIC_SEARCH_RESULT_LIMIT ?? 5),
  },
});

const casesPath = stringArg(cliArgValue('cases', args.cases));
const profile = stringArg(cliArgValue('profile', args.profile));
const model = stringArg(cliArgValue('model', args.model));
const judgeModel = stringArg(cliArgValue('judge-model', args['judge-model'] ?? model));
const ollamaUrl = stringArg(cliArgValue('ollama-url', args['ollama-url'])).replace(/\/+$/u, '');
const maxToolCalls = numberArg(cliArgValue('max-tool-calls', args['max-tool-calls']));
const temperature = numberArg(cliArgValue('temperature', args.temperature));
const judgeTemperature = numberArg(
  cliArgValue('judge-temperature', args['judge-temperature']),
);
const outputPath = stringArg(cliArgValue('output', args.output));
const trajectoryOutputPath =
  optionalStringArg(cliArgValue('trajectory-output', args['trajectory-output'])) ??
  deriveTrajectoryOutputPath(outputPath);
const searchPath = stringArg(cliArgValue('search-path', args['search-path']));
const semanticResultLimit = numberArg(
  cliArgValue('semantic-result-limit', args['semantic-result-limit']),
);
const limit = numberArg(cliArgValue('limit', args.limit));
const arms = parseArms(stringArg(cliArgValue('arms', args.arms)));
const cases = (await readCases(casesPath)).slice(
  0,
  limit > 0 ? limit : undefined,
);

if (cases.length === 0) {
  throw new Error(`No EnterpriseRAG-Bench eval cases found in ${casesPath}.`);
}

const results: EvalResult[] = [];
const trajectories: TrajectoryRecord[] = [];
await resetOutputFiles([
  outputPath,
  markdownPathForJsonl(outputPath),
  trajectoryOutputPath,
  markdownPathForJsonl(trajectoryOutputPath),
]);
const previousSemanticLimit = process.env.SEMANTIC_SEARCH_RESULT_LIMIT;
process.env.SEMANTIC_SEARCH_RESULT_LIMIT = String(semanticResultLimit);

const runtimes: Array<{
  arm: Arm;
  client: ReturnType<typeof createOpenSearchClient>;
  bash: Bash;
}> = [];
try {
  for (const arm of arms) {
    const client = createOpenSearchClient(profile);
    const session = await initSessionTree(client, profile);
    const fs = new OpenSearchFs({
      client,
      files: session.files,
      dirs: session.dirs,
    });
    const semanticSearcher = new OpenSearchSemanticSearcher({ client });
    const grep = defineCommand('grep', (commandArgs, ctx) =>
      runOpenSearchGrep(commandArgs, ctx, fs),
    );
    const customCommands = [grep];
    if (arm === 'grep+semantic_search') {
      customCommands.push(
        defineCommand('semantic_search', (commandArgs, ctx) =>
          runOpenSearchSemanticSearch(commandArgs, ctx, fs, semanticSearcher),
        ),
      );
    }
    runtimes.push({
      arm,
      client,
      bash: new Bash({ fs, cwd: '/', customCommands }),
    });
  }

  for (const [caseIndex, testCase] of cases.entries()) {
    const orderedRuntimes =
      caseIndex % 2 === 0 ? runtimes : [...runtimes].reverse();
    for (const runtime of orderedRuntimes) {
      const { result, trajectory } = await runCase(
        runtime.arm,
        testCase,
        runtime.bash,
      );
      results.push(result);
      trajectories.push(trajectory);
      await appendJsonl(outputPath, result);
      await appendJsonl(trajectoryOutputPath, trajectory);
      console.log(
        `[${runtime.arm}] ${testCase.question_id}: overall=${result.overallScore.toFixed(3)} answer=${result.answerCorrect ? 'correct' : 'wrong'} docRecall=${result.documentRecall.toFixed(3)} (${result.elapsedMs}ms)`,
      );
    }
  }
} finally {
  await Promise.all(runtimes.map((runtime) => runtime.client.close()));
  restoreEnv('SEMANTIC_SEARCH_RESULT_LIMIT', previousSemanticLimit);
}

await writeReportMarkdown(outputPath, results);
await writeTrajectoryMarkdown(trajectoryOutputPath, trajectories);

async function runCase(
  arm: Arm,
  testCase: EvalCase,
  bash: Bash,
): Promise<RunCaseOutput> {
  const startedAt = Date.now();
  const toolCalls: ToolCallLog[] = [];
  const assistantTurns: AssistantTurnLog[] = [];
  const system = systemPrompt(arm);
  const user = `Answer the following question.\nQuestion: ${testCase.question}`;
  const messages: OllamaMessage[] = [
    { role: 'system', content: system },
    { role: 'user', content: user },
  ];

  const finish = async (
    finalAnswer: string,
    error?: string,
  ): Promise<RunCaseOutput> => {
    const judgeResult = await judgeAnswer(testCase, finalAnswer);
    const result = buildResult(
      arm,
      testCase,
      finalAnswer,
      toolCalls,
      judgeResult,
      startedAt,
      error,
    );
    return {
      result,
      trajectory: buildTrajectoryRecord(result, system, user, assistantTurns),
    };
  };

  try {
    for (let turn = 0; turn <= maxToolCalls; turn += 1) {
      const response = await chat(messages, arm);
      const message = response.message;
      if (!message) throw new Error('Ollama returned no message.');
      messages.push(message);

      const calls = message.tool_calls ?? [];
      const parsedCalls = calls.map((call) => ({
        name: call.function?.name ?? '',
        arguments: parseToolArguments(call.function?.arguments),
      }));
      assistantTurns.push({
        turn,
        content: message.content ?? '',
        ...(message.thinking ? { thinking: message.thinking } : {}),
        toolCalls: parsedCalls,
      });

      if (calls.length === 0) {
        return await finish(message.content ?? '');
      }

      if (toolCalls.length + calls.length > maxToolCalls) {
        return await finish(
          message.content ?? '',
          `Exceeded max tool calls (${maxToolCalls}).`,
        );
      }

      for (const call of parsedCalls) {
        const name = call.name;
        const callArgs = call.arguments;
        const toolStartedAt = Date.now();
        const result =
          name === 'docs_bash'
            ? await runDocsBash(bash, String(callArgs.command ?? ''))
            : {
                text: `Unknown tool: ${name}`,
                rawLength: `Unknown tool: ${name}`.length,
                truncated: false,
              };
        toolCalls.push({
          turn,
          name,
          arguments: callArgs,
          result: result.text,
          resultRawLength: result.rawLength,
          resultTruncated: result.truncated,
          elapsedMs: Date.now() - toolStartedAt,
        });
        messages.push({
          role: 'tool',
          tool_name: name,
          content: result.text,
        });
      }
    }

    return await finish('', `Exceeded max tool calls (${maxToolCalls}).`);
  } catch (error) {
    return await finish('', error instanceof Error ? error.message : String(error));
  }
}

async function runDocsBash(
  bash: Bash,
  command: string,
): Promise<DocsBashResult> {
  if (command.trim().length === 0) {
    return clipToolOutput('exitCode: 2\n[stderr]\nempty command', 8000);
  }
  const { stdout, stderr, exitCode } = await bash.exec(command);
  const text =
    `exitCode: ${exitCode}\n` +
    (stdout || '(empty stdout)') +
    (stderr ? `\n[stderr]\n${stderr}` : '');
  return clipToolOutput(
    text,
    Number(process.env.OLLAMA_QA_MAX_TOOL_OUTPUT_CHARS ?? 8000),
  );
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
                ? 'Run a read-only bash command on the enterprise documentation virtual filesystem rooted at "/". Available commands include ls, cat, grep, semantic_search, pwd, cd, head, tail, and find.'
                : 'Run a read-only bash command on the enterprise documentation virtual filesystem rooted at "/". Available commands include ls, cat, grep, pwd, cd, head, tail, and find.',
            parameters: {
              type: 'object',
              required: ['command'],
              properties: {
                command: {
                  type: 'string',
                  description:
                    arm === 'grep+semantic_search'
                      ? `Bash command to execute, for example: ls /, grep -ri "search term" ${searchPath}, semantic_search "natural language question" ${searchPath}, or cat <path>`
                      : `Bash command to execute, for example: ls /, grep -ri "search term" ${searchPath}, or cat <path>`,
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

async function judgeAnswer(
  testCase: EvalCase,
  finalAnswer: string,
): Promise<JudgeResult> {
  const prompt = judgePrompt(testCase, finalAnswer);
  let rawResponse = '';
  let lastError = '';
  let messages: OllamaMessage[] = [
    { role: 'system', content: prompt.system },
    { role: 'user', content: prompt.user },
  ];

  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const response = await judgeChat(messages);
      rawResponse = response.message?.content ?? '';
      const parsed = parseJudgeResponseText(rawResponse, testCase.answerFacts);
      return {
        ...parsed,
        prompt,
        rawResponse,
      };
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
      if (attempt === 0) {
        messages = [
          ...messages,
          { role: 'assistant', content: rawResponse },
          {
            role: 'user',
            content:
              'The previous response was invalid. Return only one valid JSON object with keys answer_correct, fact_judgments, completeness, and reason.',
          },
        ];
      }
    }
  }

  return {
    answer_correct: false,
    fact_judgments: testCase.answerFacts.map((fact) => ({
      fact,
      supported: false,
      reason: 'Judge failed before producing a valid fact judgment.',
    })),
    completeness: 0,
    reason: 'Judge failed before producing valid JSON.',
    prompt,
    rawResponse,
    error: lastError || 'Judge failed before producing valid JSON.',
  };
}

async function judgeChat(messages: OllamaMessage[]): Promise<OllamaChatResponse> {
  const response = await fetch(`${ollamaUrl}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: judgeModel,
      messages,
      stream: false,
      think: false,
      format: 'json',
      options: { temperature: judgeTemperature },
    }),
  });
  if (!response.ok) {
    throw new Error(`Ollama judge failed: ${response.status} ${await response.text()}`);
  }
  return (await response.json()) as OllamaChatResponse;
}

function judgePrompt(
  testCase: EvalCase,
  finalAnswer: string,
): { system: string; user: string } {
  return {
    system: `You are a strict evaluator for EnterpriseRAG-Bench style question answering.
Return only valid JSON. Do not use markdown.
Judge semantic equivalence, not exact wording.
The answer is correct only if it directly answers the question and is consistent with the gold answer.
For each answer fact, mark supported=true only when the model answer clearly states that fact or an equivalent fact.
Do not give credit for facts that appear only in the gold answer, only for facts present in the model answer.`,
    user: JSON.stringify(
      {
        question_id: testCase.question_id,
        question: testCase.question,
        gold_answer: testCase.goldAnswer,
        answer_facts: testCase.answerFacts,
        model_answer: finalAnswer,
        required_schema: {
          answer_correct: 'boolean',
          fact_judgments:
            'array with one item per answer_facts item: {fact:string,supported:boolean,reason:string}',
          completeness: 'number from 0 to 1',
          reason: 'short string',
        },
      },
      null,
      2,
    ),
  };
}

function buildResult(
  arm: Arm,
  testCase: EvalCase,
  finalAnswer: string,
  toolCalls: ToolCallLog[],
  judgeResult: JudgeResult,
  startedAt: number,
  error?: string,
): EvalResult {
  const citedPaths = extractCitedPaths(finalAnswer);
  const documentIds = extractDocumentIdsFromPaths(citedPaths);
  const catPaths = extractCatPaths(toolCalls);
  const readDocumentIds = extractDocumentIdsFromPaths(catPaths);
  const documentRecall = calculateDocumentRecall(
    documentIds,
    testCase.expectedDocIds,
  );
  const invalidExtraDocs = invalidExtraDocCount(
    documentIds,
    testCase.expectedDocIds,
  );
  const answerCompleteness = judgeResult.completeness;
  const answerCorrect = judgeResult.answer_correct;
  const overallScore = answerCorrect && !error ? answerCompleteness : 0;
  const citationCorrect = isAnswerable(testCase)
    ? testCase.positivePaths.some(
        (pathValue) => finalAnswer.includes(pathValue) || citedPaths.includes(pathValue),
      )
    : true;
  return {
    ...testCase,
    arm,
    model,
    judgeModel,
    answer: finalAnswer,
    finalAnswer,
    citedPaths,
    document_ids: documentIds,
    readDocumentIds,
    documentRecall,
    invalidExtraDocs,
    toolCalls,
    grepCalls: toolCalls.filter((call) =>
      String(call.arguments.command ?? '').trim().startsWith('grep '),
    ).length,
    semanticSearchCalls: toolCalls.filter((call) =>
      String(call.arguments.command ?? '').trim().startsWith('semantic_search '),
    ).length,
    catPaths,
    elapsedMs: Date.now() - startedAt,
    answerFactRecall: answerCompleteness,
    answerCompleteness,
    answerCorrect,
    overallScore,
    citationCorrect,
    judgeResult,
    ...(judgeResult.error ? { judgeError: judgeResult.error } : {}),
    correct: answerCorrect && !error,
    ...(error ? { error } : {}),
  };
}

function buildTrajectoryRecord(
  result: EvalResult,
  system: string,
  user: string,
  assistantTurns: AssistantTurnLog[],
): TrajectoryRecord {
  return {
    question_id: result.question_id,
    arm: result.arm,
    model: result.model,
    judgeModel: result.judgeModel,
    questionType: result.questionType,
    sourceTypes: result.sourceTypes,
    question: result.question,
    goldAnswer: result.goldAnswer,
    answerFacts: result.answerFacts,
    expectedDocIds: result.expectedDocIds,
    positivePaths: result.positivePaths,
    prompts: { system, user },
    assistantTurns,
    toolCalls: result.toolCalls,
    judgeResult: result.judgeResult,
    finalAnswer: result.finalAnswer,
    citedPaths: result.citedPaths,
    document_ids: result.document_ids,
    readDocumentIds: result.readDocumentIds,
    documentRecall: result.documentRecall,
    invalidExtraDocs: result.invalidExtraDocs,
    catPaths: result.catPaths,
    grepCalls: result.grepCalls,
    semanticSearchCalls: result.semanticSearchCalls,
    elapsedMs: result.elapsedMs,
    answerFactRecall: result.answerFactRecall,
    answerCompleteness: result.answerCompleteness,
    answerCorrect: result.answerCorrect,
    overallScore: result.overallScore,
    citationCorrect: result.citationCorrect,
    ...(result.judgeError ? { judgeError: result.judgeError } : {}),
    correct: result.correct,
    ...(result.error ? { error: result.error } : {}),
  };
}

function systemPrompt(arm: Arm): string {
  const semanticInstruction =
    arm === 'grep+semantic_search'
      ? `\n- semantic_search "<natural language query>" ${searchPath} is available when exact keywords are uncertain, the question is phrased naturally, or grep does not find enough candidates. You do not need to start with semantic_search if grep or directory exploration is more appropriate.\n- Do not finalize from a weak candidate file. If the files found by grep do not directly answer the question, try semantic_search before giving the final answer.`
      : '';
  return `You are an English QA assistant with access to a virtual filesystem of enterprise documents via the docs_bash tool.
- You must inspect the filesystem before answering.
- Start with ls / and search under ${searchPath}.
- Use grep -ri "<term>" ${searchPath} for keyword search.${semanticInstruction}
- You must use cat <path> to read at least one candidate file before giving the final answer.
- Answer in English.
- Preserve numbers, identifiers, metric names, timestamps, and dates exactly as written in the evidence documents.
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

async function writeReportMarkdown(
  jsonlPath: string,
  rows: EvalResult[],
): Promise<void> {
  const resolvedJsonlPath = path.resolve(jsonlPath);
  const mdPath = markdownPathForJsonl(resolvedJsonlPath);
  await writeFile(mdPath, toMarkdown(rows), 'utf8');
  console.log(`Wrote ${mdPath}`);
}

async function writeTrajectoryMarkdown(
  jsonlPath: string,
  rows: TrajectoryRecord[],
): Promise<void> {
  const resolvedJsonlPath = path.resolve(jsonlPath);
  const mdPath = markdownPathForJsonl(resolvedJsonlPath);
  await writeFile(mdPath, toTrajectoryMarkdown(rows), 'utf8');
  console.log(`Wrote ${resolvedJsonlPath}`);
  console.log(`Wrote ${mdPath}`);
}

function toMarkdown(rows: EvalResult[]): string {
  const lines = [
    '# Ollama EnterpriseRAG-Bench agent eval',
    '',
    `model: \`${model}\``,
    `judge model: \`${judgeModel}\``,
    `search path: \`${searchPath}\``,
    '',
    '| arm | cases | overall score | answer correctness | answer completeness | document recall | invalid extra docs | citation debug | avg tools | avg grep | avg semantic | avg cat paths | avg ms |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
  ];
  for (const row of summarize(rows)) {
    lines.push(
      `| ${row.arm} | ${row.cases} | ${formatRate(row.avgOverallScore)} | ${percent(row.answerCorrect, row.cases)} | ${formatRate(row.avgAnswerCompleteness)} | ${formatRate(row.avgDocumentRecall)} | ${row.avgInvalidExtraDocs.toFixed(2)} | ${percent(row.citationCorrect, row.cases)} | ${row.avgTools.toFixed(1)} | ${row.avgGrep.toFixed(1)} | ${row.avgSemantic.toFixed(1)} | ${row.avgCatPaths.toFixed(1)} | ${Math.round(row.avgMs)} |`,
    );
  }
  lines.push('', '## Cases', '');
  lines.push(
    '| question_id | type | arm | overall | answer | completeness | doc recall | invalid docs | document_ids | expectedDocIds | judge reason | final answer |',
  );
  lines.push('|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|');
  for (const row of rows) {
    lines.push(
      `| ${escapeMd(row.question_id)} | ${escapeMd(row.questionType)} | ${row.arm} | ${row.overallScore.toFixed(3)} | ${row.answerCorrect ? 'yes' : 'no'} | ${row.answerCompleteness.toFixed(3)} | ${row.documentRecall.toFixed(3)} | ${row.invalidExtraDocs} | ${escapeMd(row.document_ids.join('<br>') || '(none)')} | ${escapeMd(row.expectedDocIds.join('<br>') || '(none)')} | ${escapeMd((row.judgeError ?? row.judgeResult.reason).slice(0, 180))} | ${escapeMd(row.finalAnswer.slice(0, 180))} |`,
    );
  }
  lines.push('');
  return lines.join('\n');
}

function toTrajectoryMarkdown(rows: TrajectoryRecord[]): string {
  const lines = [
    '# Ollama EnterpriseRAG-Bench trajectory',
    '',
    `model: \`${model}\``,
    `judge model: \`${judgeModel}\``,
    `search path: \`${searchPath}\``,
    '',
  ];

  for (const row of rows) {
    lines.push(
      `## ${row.question_id} / ${row.arm} / ${row.correct ? 'correct' : 'wrong'}`,
      '',
      `- type: \`${row.questionType}\``,
      `- answerCorrect: \`${row.answerCorrect}\``,
      `- answerCompleteness: \`${row.answerCompleteness.toFixed(3)}\``,
      `- overallScore: \`${row.overallScore.toFixed(3)}\``,
      `- documentRecall: \`${row.documentRecall.toFixed(3)}\``,
      `- invalidExtraDocs: \`${row.invalidExtraDocs}\``,
      `- citationCorrectDebug: \`${row.citationCorrect}\``,
      `- grepCalls: \`${row.grepCalls}\``,
      `- semanticSearchCalls: \`${row.semanticSearchCalls}\``,
      `- catPaths: \`${row.catPaths.length}\``,
      ...(row.judgeError ? [`- judgeError: \`${escapeInlineCode(row.judgeError)}\``] : []),
      ...(row.error ? [`- error: \`${escapeInlineCode(row.error)}\``] : []),
      '',
      '### Question',
      '',
      row.question,
      '',
      '### Gold Answer',
      '',
      clipForMarkdown(row.goldAnswer, 1200),
      '',
      '### Positive Paths',
      '',
      row.positivePaths.length > 0
        ? row.positivePaths.map((pathValue) => `- \`${pathValue}\``).join('\n')
        : '(none)',
      '',
      '### Document IDs',
      '',
      `- expected: ${row.expectedDocIds.map((docId) => `\`${docId}\``).join(', ') || '(none)'}`,
      `- cited: ${row.document_ids.map((docId) => `\`${docId}\``).join(', ') || '(none)'}`,
      `- read: ${row.readDocumentIds.map((docId) => `\`${docId}\``).join(', ') || '(none)'}`,
      '',
      '### Judge',
      '',
      `reason: ${row.judgeResult.reason || row.judgeError || '(none)'}`,
      '',
      'fact judgments:',
      '',
      row.judgeResult.fact_judgments.length > 0
        ? row.judgeResult.fact_judgments
            .map(
              (judgment) =>
                `- ${judgment.supported ? 'yes' : 'no'}: ${escapeMd(judgment.fact)}${judgment.reason ? ` (${escapeMd(judgment.reason)})` : ''}`,
            )
            .join('\n')
        : '(none)',
      '',
      'judge prompt:',
      '',
      '```text',
      escapeCodeFence(
        clipForMarkdown(
          `${row.judgeResult.prompt.system}\n\n${row.judgeResult.prompt.user}`,
          1800,
        ),
      ),
      '```',
      '',
      'judge raw response:',
      '',
      '```text',
      escapeCodeFence(clipForMarkdown(row.judgeResult.rawResponse || '(empty)', 1200)),
      '```',
      '',
      '### Commands',
      '',
      row.toolCalls.length > 0
        ? row.toolCalls
            .map((call, index) => {
              const command = String(call.arguments.command ?? '');
              return `${index + 1}. \`${escapeInlineCode(command || call.name)}\``;
            })
            .join('\n')
        : '(none)',
      '',
      '### Tool Results',
      '',
    );

    for (const [index, call] of row.toolCalls.entries()) {
      const command = String(call.arguments.command ?? call.name);
      lines.push(
        `#### ${index + 1}. ${escapeMd(command)}`,
        '',
        `- elapsedMs: \`${call.elapsedMs}\``,
        `- rawLength: \`${call.resultRawLength}\``,
        `- truncated: \`${call.resultTruncated}\``,
        '',
        '```text',
        escapeCodeFence(clipForMarkdown(call.result, 1600)),
        '```',
        '',
      );
    }

    lines.push('### Assistant Turns', '');
    if (row.assistantTurns.length === 0) {
      lines.push('(none)', '');
    } else {
      for (const turn of row.assistantTurns) {
        const commands = turn.toolCalls
          .map((call) => String(call.arguments.command ?? call.name))
          .join(' | ');
        lines.push(
          `#### turn ${turn.turn}`,
          '',
          commands
            ? `commands: \`${escapeInlineCode(commands)}\``
            : 'commands: `(none)`',
          '',
          '```text',
          escapeCodeFence(clipForMarkdown(turn.content || '(empty)', 1200)),
          '```',
          '',
        );
      }
    }

    lines.push(
      '### Final Answer',
      '',
      '```text',
      escapeCodeFence(clipForMarkdown(row.finalAnswer || '(empty)', 2000)),
      '```',
      '',
    );
  }

  return lines.join('\n');
}

function summarize(rows: EvalResult[]): Array<{
  arm: Arm;
  cases: number;
  correct: number;
  answerCorrect: number;
  citationCorrect: number;
  avgOverallScore: number;
  avgAnswerCompleteness: number;
  avgDocumentRecall: number;
  avgInvalidExtraDocs: number;
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
    avgOverallScore: average(group.map((row) => row.overallScore)),
    avgAnswerCompleteness: average(group.map((row) => row.answerCompleteness)),
    avgDocumentRecall: average(group.map((row) => row.documentRecall)),
    avgInvalidExtraDocs: average(group.map((row) => row.invalidExtraDocs)),
    avgTools: average(group.map((row) => row.toolCalls.length)),
    avgGrep: average(group.map((row) => row.grepCalls)),
    avgSemantic: average(group.map((row) => row.semanticSearchCalls)),
    avgCatPaths: average(group.map((row) => row.catPaths.length)),
    avgMs: average(group.map((row) => row.elapsedMs)),
  }));
}

async function resetOutputFiles(filePaths: string[]): Promise<void> {
  for (const filePath of filePaths) {
    const resolved = path.resolve(filePath);
    await mkdir(path.dirname(resolved), { recursive: true });
    await writeFile(resolved, '', 'utf8');
  }
}

async function appendJsonl(filePath: string, value: unknown): Promise<void> {
  const resolved = path.resolve(filePath);
  await appendFile(resolved, `${JSON.stringify(value)}\n`, 'utf8');
}

function deriveTrajectoryOutputPath(jsonlPath: string): string {
  if (jsonlPath.endsWith('.jsonl')) {
    return jsonlPath.replace(/\.jsonl$/u, '.trajectory.jsonl');
  }
  return `${jsonlPath}.trajectory.jsonl`;
}

function markdownPathForJsonl(filePath: string): string {
  if (filePath.endsWith('.jsonl')) {
    return filePath.replace(/\.jsonl$/u, '.md');
  }
  return `${filePath}.md`;
}

function cliArgValue(name: string, fallback: unknown): unknown {
  const flag = `--${name}`;
  const prefix = `${flag}=`;
  let value: string | undefined;
  for (let index = 0; index < rawArgv.length; index += 1) {
    const token = rawArgv[index];
    if (token === flag && rawArgv[index + 1] !== undefined) {
      value = rawArgv[index + 1];
      index += 1;
    } else if (token.startsWith(prefix)) {
      value = token.slice(prefix.length);
    }
  }
  return value ?? fallback;
}

function lastArg(value: unknown): unknown {
  return Array.isArray(value) ? value[value.length - 1] : value;
}

function stringArg(value: unknown): string {
  return String(lastArg(value));
}

function optionalStringArg(value: unknown): string | undefined {
  const valueOrLast = lastArg(value);
  return typeof valueOrLast === 'string' && valueOrLast.length > 0
    ? valueOrLast
    : undefined;
}

function numberArg(value: unknown): number {
  return Number(lastArg(value));
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

function isAnswerable(testCase: EvalCase): boolean {
  return testCase.positivePaths.length > 0;
}

function average(values: number[]): number {
  return values.length === 0
    ? 0
    : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function percent(numerator: number, denominator: number): string {
  if (denominator === 0) return '0.0%';
  return `${((numerator / denominator) * 100).toFixed(1)}%`;
}

function formatRate(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function escapeMd(value: string): string {
  return value.replace(/\|/gu, '\\|').replace(/\n/gu, '<br>');
}

function clipToolOutput(text: string, maxChars: number): DocsBashResult {
  if (text.length <= maxChars) {
    return { text, rawLength: text.length, truncated: false };
  }
  return {
    text: `${text.slice(0, maxChars)}\n...[truncated ${text.length - maxChars} chars]`,
    rawLength: text.length,
    truncated: true,
  };
}

function clipForMarkdown(value: string, maxChars: number): string {
  if (value.length <= maxChars) return value;
  return `${value.slice(0, maxChars)}\n...[truncated ${value.length - maxChars} chars]`;
}

function escapeInlineCode(value: string): string {
  return value.replace(/`/gu, '\\`');
}

function escapeCodeFence(value: string): string {
  return value.replace(/```/gu, '``\\`');
}

function restoreEnv(key: string, previous: string | undefined): void {
  if (previous === undefined) {
    delete process.env[key];
  } else {
    process.env[key] = previous;
  }
}
