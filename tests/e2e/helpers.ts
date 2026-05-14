import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { Bash, defineCommand } from 'just-bash';
import { afterAll, beforeAll, describe, expect } from 'vitest';
import { OpenSearchFs } from '../../src/core/opensearchfs.js';
import { runOpenSearchGrep } from '../../src/core/grep.js';
import { runOpenSearchSemanticSearch } from '../../src/core/semantic-search.js';
import { createOpenSearchClient } from '../../src/opensearch-adapter/client.js';
import { initSessionTree } from '../../src/session.js';

export type Profile = 'PUBLIC' | 'BILLING' | 'INTERNAL' | 'SYSTEM';

export type CommandExpectation = {
  exitCode: number;
  stdoutContains?: string[];
  stdoutExact?: string;
  stderrContains?: string[];
  stderrExact?: string;
};

const envPath = resolve(process.cwd(), '.env');
if (existsSync(envPath)) {
  for (const line of readFileSync(envPath, 'utf8').split('\n')) {
    if (!line || line.startsWith('#')) continue;
    const [key, value] = line.split('=', 2);
    if (key?.startsWith('OPENSEARCH_') && value) process.env[key] = value;
  }
}

interface OpenSearchSession {
  client: ReturnType<typeof createOpenSearchClient>;
  fs: OpenSearchFs;
}

type CommandCase = {
  command: string;
  expected: CommandExpectation;
};

export function hasProfileEnv(profile: Profile): boolean {
  return Boolean(
    process.env.OPENSEARCH_URL &&
      process.env[`OPENSEARCH_USERNAME_${profile}`] &&
      process.env[`OPENSEARCH_PASSWORD_${profile}`],
  );
}

export function hasProfilesEnv(profiles: readonly Profile[]): boolean {
  return profiles.every((profile) => hasProfileEnv(profile));
}

export async function createOpenSearchSession(
  profile: Profile,
): Promise<OpenSearchSession> {
  const client = createOpenSearchClient(profile);
  const session = await initSessionTree(client, profile);
  const fs = new OpenSearchFs({
    client,
    files: session.files,
    dirs: session.dirs,
  });
  return { client, fs };
}

export async function createBashSession(
  profile: Profile,
): Promise<OpenSearchSession & { bash: Bash }> {
  const { client, fs } = await createOpenSearchSession(profile);
  const grep = defineCommand('grep', async (args, ctx) =>
    runOpenSearchGrep(args, ctx, fs),
  );
  const semanticSearch = defineCommand('semantic_search', async (args, ctx) =>
    runOpenSearchSemanticSearch(args, ctx, fs),
  );
  const bash = new Bash({
    fs,
    cwd: '/',
    customCommands: [grep, semanticSearch],
  });
  return { client, fs, bash };
}

function runContainsChecks(
  label: string,
  value: string,
  needles: string[],
): string[] {
  const failures: string[] = [];
  for (const needle of needles) {
    if (!value.includes(needle)) {
      failures.push(
        `${label} missing expected snippet: ${JSON.stringify(needle)}`,
      );
    }
  }
  return failures;
}

export async function assertCommandCase(
  bash: Bash,
  testCase: CommandCase,
): Promise<{ exitCode: number; stdout: string; stderr: string }> {
  const actual = await bash.exec(testCase.command);
  const failures: string[] = [];

  if (actual.exitCode !== testCase.expected.exitCode) {
    failures.push(
      `exitCode mismatch: expected ${testCase.expected.exitCode}, got ${actual.exitCode}`,
    );
  }
  if (
    testCase.expected.stdoutExact !== undefined &&
    actual.stdout !== testCase.expected.stdoutExact
  ) {
    failures.push('stdout exact mismatch');
  }
  if (
    testCase.expected.stderrExact !== undefined &&
    actual.stderr !== testCase.expected.stderrExact
  ) {
    failures.push('stderr exact mismatch');
  }
  if (testCase.expected.stdoutContains) {
    failures.push(
      ...runContainsChecks(
        'stdout',
        actual.stdout,
        testCase.expected.stdoutContains,
      ),
    );
  }
  if (testCase.expected.stderrContains) {
    failures.push(
      ...runContainsChecks(
        'stderr',
        actual.stderr,
        testCase.expected.stderrContains,
      ),
    );
  }

  expect(failures).toEqual([]);
  return actual;
}

/**
 * Conditionally run a describe block with a SYSTEM bash session, skipping when
 * the SYSTEM profile env vars are absent. Registers beforeAll/afterAll for the
 * session lifecycle. Use `ctx.bash` inside `it` callbacks.
 */
export function describeSystemBash(
  suite: string,
  tests: (ctx: { bash: Bash }) => void,
): void {
  const run = hasProfilesEnv(['SYSTEM']) ? describe : describe.skip;
  run(suite, () => {
    const ctx = {} as { bash: Bash };
    let closeClient: (() => Promise<void>) | null = null;

    beforeAll(async () => {
      const session = await createBashSession('SYSTEM');
      ctx.bash = session.bash;
      closeClient = async () => session.client.close();
    });

    afterAll(async () => {
      await closeClient?.();
    });

    tests(ctx);
  });
}
