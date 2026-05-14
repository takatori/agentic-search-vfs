import { describe, expect, it, beforeAll, afterAll } from 'vitest';
import {
  createBashSession,
  describeSystemBash,
  hasProfileEnv,
} from './helpers.js';

const runRuri = process.env.OPENSEARCHFS_RUN_RURI_E2E === '1';

if (runRuri) {
  describeSystemBash('semantic_search e2e (SYSTEM)', (ctx) => {
    it('returns cat-readable paths for a scoped semantic query', async () => {
      const actual = await ctx.bash.exec('semantic_search "OAuth token" /auth');
      expect(actual.exitCode).toBe(0);
      expect(actual.stdout).toContain('/auth/oauth.mdx:');

      const firstPath = actual.stdout.match(/^(\/[^:\n]+\.mdx):/u)?.[1];
      expect(firstPath).toBeTruthy();
      const cat = await ctx.bash.exec(`cat ${firstPath}`);
      expect(cat.exitCode).toBe(0);
      expect(cat.stdout).toContain('OAuth');
    });
  });
} else {
  describe.skip('semantic_search e2e (SYSTEM)', () => {});
}

const publicRun = runRuri && hasProfileEnv('PUBLIC') ? describe : describe.skip;
publicRun('semantic_search e2e (PUBLIC DLS)', () => {
  let session: Awaited<ReturnType<typeof createBashSession>> | null = null;

  beforeAll(async () => {
    session = await createBashSession('PUBLIC');
  });

  afterAll(async () => {
    await session?.client.close();
  });

  it('does not expose denied documents', async () => {
    const actual = await session!.bash.exec('semantic_search "billing payment" /');
    expect(actual.stdout).not.toContain('/internal/billing.mdx');
    expect(actual.stdout).not.toContain('/api-reference/payments.mdx');
  });
});
