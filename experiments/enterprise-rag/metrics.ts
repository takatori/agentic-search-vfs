export type FactJudgment = {
  fact: string;
  supported: boolean;
  reason: string;
};

export type ParsedJudgeResponse = {
  answer_correct: boolean;
  fact_judgments: FactJudgment[];
  completeness: number;
  reason: string;
};

export function extractDocumentIdsFromPaths(paths: readonly string[]): string[] {
  const ids = new Set<string>();
  for (const pathValue of paths) {
    const withoutSuffix = pathValue.split(/[?#]/u)[0] ?? pathValue;
    const basename = withoutSuffix.split('/').pop() ?? withoutSuffix;
    const basenameDocId = basename.replace(/\.mdx$/u, '');
    if (isEnterpriseDocId(basenameDocId)) {
      ids.add(basenameDocId);
      continue;
    }
    for (const match of pathValue.matchAll(/dsid_[A-Za-z0-9]+/gu)) {
      ids.add(match[0]);
    }
  }
  return [...ids];
}

export function documentRecall(
  documentIds: readonly string[],
  expectedDocIds: readonly string[],
): number {
  const expected = unique(expectedDocIds);
  if (expected.length === 0) return 1;
  const returned = new Set(documentIds);
  const found = expected.filter((docId) => returned.has(docId)).length;
  return found / expected.length;
}

export function invalidExtraDocCount(
  documentIds: readonly string[],
  expectedDocIds: readonly string[],
): number {
  const expected = new Set(expectedDocIds);
  return unique(documentIds).filter((docId) => !expected.has(docId)).length;
}

export function parseJudgeResponseText(
  text: string,
  answerFacts: readonly string[],
): ParsedJudgeResponse {
  const rawJson = extractFirstJsonObject(text);
  if (!rawJson) {
    throw new Error('Judge response did not contain a JSON object.');
  }

  const parsed = JSON.parse(rawJson) as unknown;
  if (!isRecord(parsed)) {
    throw new Error('Judge response JSON must be an object.');
  }

  const answerCorrect = coerceBoolean(parsed.answer_correct);
  if (answerCorrect === undefined) {
    throw new Error('Judge response missing boolean answer_correct.');
  }

  const rawJudgments = Array.isArray(parsed.fact_judgments)
    ? parsed.fact_judgments
    : [];
  const factJudgments = alignFactJudgments(answerFacts, rawJudgments);
  const completeness = completenessFromFactJudgments(
    answerFacts,
    factJudgments,
  );

  return {
    answer_correct: answerCorrect,
    fact_judgments: factJudgments,
    completeness,
    reason: typeof parsed.reason === 'string' ? parsed.reason : '',
  };
}

export function completenessFromFactJudgments(
  answerFacts: readonly string[],
  factJudgments: readonly FactJudgment[],
): number {
  const facts = answerFacts.filter((fact) => fact.trim().length > 0);
  if (facts.length === 0) return 1;
  const supported = factJudgments
    .slice(0, facts.length)
    .filter((judgment) => judgment.supported).length;
  return supported / facts.length;
}

function alignFactJudgments(
  answerFacts: readonly string[],
  rawJudgments: readonly unknown[],
): FactJudgment[] {
  const byFact = new Map<string, FactJudgment>();
  const parsedJudgments = rawJudgments.map(parseFactJudgment);
  for (const judgment of parsedJudgments) {
    byFact.set(normalizeFact(judgment.fact), judgment);
  }

  return answerFacts.map((fact, index) => {
    const matched = byFact.get(normalizeFact(fact)) ?? parsedJudgments[index];
    return {
      fact,
      supported: matched?.supported ?? false,
      reason: matched?.reason ?? '',
    };
  });
}

function parseFactJudgment(value: unknown): FactJudgment {
  if (!isRecord(value)) {
    return { fact: '', supported: false, reason: '' };
  }
  return {
    fact: typeof value.fact === 'string' ? value.fact : '',
    supported: coerceBoolean(value.supported) ?? false,
    reason: typeof value.reason === 'string' ? value.reason : '',
  };
}

function extractFirstJsonObject(text: string): string | undefined {
  const start = text.indexOf('{');
  if (start === -1) return undefined;

  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let index = start; index < text.length; index += 1) {
    const char = text[index];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (char === '\\') {
      escaped = inString;
      continue;
    }
    if (char === '"') {
      inString = !inString;
      continue;
    }
    if (inString) continue;
    if (char === '{') depth += 1;
    if (char === '}') {
      depth -= 1;
      if (depth === 0) return text.slice(start, index + 1);
    }
  }
  return undefined;
}

function coerceBoolean(value: unknown): boolean | undefined {
  if (typeof value === 'boolean') return value;
  if (typeof value !== 'string') return undefined;
  const normalized = value.trim().toLowerCase();
  if (normalized === 'true') return true;
  if (normalized === 'false') return false;
  return undefined;
}

function isEnterpriseDocId(value: string): boolean {
  return /^dsid_[A-Za-z0-9]+$/u.test(value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function normalizeFact(value: string): string {
  return value.normalize('NFKC').trim().toLowerCase().replace(/\s+/gu, ' ');
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values)];
}
