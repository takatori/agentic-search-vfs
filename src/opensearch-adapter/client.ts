import { Client } from '@opensearch-project/opensearch';

/**
 * Creates an OpenSearch client using `OPENSEARCH_URL` plus profile-specific
 * basic auth credentials from the environment. Throws if required variables
 * are missing.
 */
export function createOpenSearchClient(profile: string): Client {
  const url = process.env.OPENSEARCH_URL;
  const username = process.env[`OPENSEARCH_USERNAME_${profile}`];
  const password = process.env[`OPENSEARCH_PASSWORD_${profile}`];

  if (!url) {
    throw new Error('Missing OPENSEARCH_URL');
  }

  if (!username || !password) {
    throw new Error(
      `Missing OPENSEARCH_USERNAME_${profile} or OPENSEARCH_PASSWORD_${profile}. Define both to use profile "${profile}".`,
    );
  }
  return new Client({
    node: url,
    auth: { username, password },
    ssl: { rejectUnauthorized: false },
  });
}
