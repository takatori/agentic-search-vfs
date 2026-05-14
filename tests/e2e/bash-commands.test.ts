import { it } from 'vitest';
import { assertCommandCase, describeSystemBash } from './helpers.js';

const SUITE = 'bash commands e2e (SYSTEM)';

describeSystemBash(SUITE, (ctx) => {
  it('supports pwd and cd', async () => {
    await assertCommandCase(ctx.bash, {
      command: 'pwd',
      expected: { exitCode: 0, stdoutContains: ['/'] },
    });
    await assertCommandCase(ctx.bash, {
      command: 'cd /auth && pwd',
      expected: { exitCode: 0, stdoutContains: ['/auth'] },
    });
  });

  it('lists visible entries with ls', async () => {
    await assertCommandCase(ctx.bash, {
      command: 'ls /',
      expected: {
        exitCode: 0,
        stdoutContains: ['auth', 'api-reference'],
      },
    });
  });

  it('reads public files with cat', async () => {
    await assertCommandCase(ctx.bash, {
      command: 'cat /auth/oauth.mdx',
      expected: {
        exitCode: 0,
        stdoutContains: ['# OAuth'],
      },
    });
  });

  it('reads internal files with system profile', async () => {
    await assertCommandCase(ctx.bash, {
      command: 'cat /internal/audit-log.mdx',
      expected: {
        exitCode: 0,
        stdoutContains: ['# Audit log'],
      },
    });
  });

  it('find enumerates files under a directory', async () => {
    await assertCommandCase(ctx.bash, {
      command: 'find /auth -type f',
      expected: {
        exitCode: 0,
        stdoutContains: ['/auth/oauth.mdx', '/auth/api-keys.mdx'],
      },
    });
  });
});
