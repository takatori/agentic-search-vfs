import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import {
  assertCommandCase,
  createBashSession,
  describeSystemBash,
  hasProfileEnv,
} from './helpers.js';

const SUITE = 'grep e2e (SYSTEM)';

describeSystemBash(SUITE, (ctx) => {
  it('keeps the grep surface working', async () => {
    const actual = await ctx.bash.exec('grep -ri "OAuth" /auth');
    expect(actual.exitCode).toBe(0);
    expect(actual.stdout).toContain('/auth/oauth.mdx');
  });

  it('supports grep -ri "access_token" /', async () => {
    await assertCommandCase(ctx.bash, {
      command: 'grep -ri "access_token" /',
      expected: {
        exitCode: 0,
        stdoutContains: ['/auth/oauth.mdx', '/api-reference/users.mdx'],
      },
    });
  });

  it('supports grep -ri "webhook" /', async () => {
    await assertCommandCase(ctx.bash, {
      command: 'grep -ri "webhook" /',
      expected: {
        exitCode: 0,
        stdoutContains: ['Verify webhook signatures'],
      },
    });
  });

  it('supports grep -ri "billing" /', async () => {
    await assertCommandCase(ctx.bash, {
      command: 'grep -ri "billing" /',
      expected: {
        exitCode: 0,
        stdoutContains: ['/api-reference/payments.mdx'],
      },
    });
  });

  it('distinguishes regex matching from fixed-string matching', async () => {
    await assertCommandCase(ctx.bash, {
      command: 'grep -r "access.token" /auth',
      expected: {
        exitCode: 0,
        stdoutContains: ['/auth/oauth.mdx'],
      },
    });
    await assertCommandCase(ctx.bash, {
      command: 'grep -rF "access.token" /auth',
      expected: {
        exitCode: 1,
        stdoutExact: '',
      },
    });
  });

  it('returns grep parse/runtime errors with exit code 2', async () => {
    await assertCommandCase(ctx.bash, {
      command: 'grep -r "(" /auth',
      expected: {
        exitCode: 2,
        stderrContains: ['invalid regular expression'],
      },
    });
  });
});

const publicRun = hasProfileEnv('PUBLIC') ? describe : describe.skip;
publicRun('grep e2e (PUBLIC DLS)', () => {
  let session: Awaited<ReturnType<typeof createBashSession>> | null = null;

  beforeAll(async () => {
    session = await createBashSession('PUBLIC');
  });

  afterAll(async () => {
    await session?.client.close();
  });

  it('does not expose denied documents', async () => {
    const actual = await session!.bash.exec('grep -ri "billing" /');
    expect(actual.stdout).not.toContain('/internal/billing.mdx');
    expect(actual.stdout).not.toContain('/api-reference/payments.mdx');
  });
});
