import 'dotenv/config';
import { Bash, defineCommand } from 'just-bash';
import { OpenSearchFs } from '../src/core/opensearchfs.js';
import { runOpenSearchGrep } from '../src/core/grep.js';
import { runOpenSearchSemanticSearch } from '../src/core/semantic-search.js';
import { createOpenSearchClient } from '../src/opensearch-adapter/client.js';
import { initSessionTree } from '../src/session.js';

const profile = process.argv[2];
if (!profile) {
  throw new Error('Missing profile. Pass it as argv[2] (example: PUBLIC).');
}

const client = createOpenSearchClient(profile);
const authResponse = await client.transport.request({
  method: 'GET',
  path: '/_plugins/_security/authinfo',
} as never);
const auth =
  (authResponse as { body?: { user_name?: string; username?: string; roles?: string[] } })
    .body ?? {};
const currentUser = auth.user_name ?? auth.username ?? '(unknown)';
const currentRoles = Array.isArray(auth.roles) ? auth.roles.join(', ') : '';
console.log(
  `Authenticated as: ${currentUser}${currentRoles ? ` (roles: ${currentRoles})` : ''} [profile=${profile}]`,
);

const session = await initSessionTree(client, profile);

// Set up virtual filesystem
const opensearchFs = new OpenSearchFs({
  client,
  files: session.files,
  dirs: session.dirs,
});

// Define custom grep command
const grep = defineCommand('grep', async (args, ctx) =>
  runOpenSearchGrep(args, ctx, opensearchFs, client),
);
const semanticSearch = defineCommand('semantic_search', async (args, ctx) =>
  runOpenSearchSemanticSearch(args, ctx, opensearchFs, client),
);

// Set up virtual bash environment
const bash = new Bash({
  fs: opensearchFs,
  cwd: '/',
  customCommands: [grep, semanticSearch],
});

async function run(command: string, clip = 500): Promise<void> {
  console.log(`Command: ${command}`);
  const { stdout, stderr } = await bash.exec(command);
  console.log(`stdout:\n${stdout.slice(0, clip)}`);
  console.log(`stderr:\n${stderr}`);
}

await run('grep -ri "OAuth" /auth');
await run('semantic_search "OAuth token" /auth');
await run('cat /auth/oauth.mdx', 100);
await run('ls /api-reference');

console.log('\nArticle smoke checks');
await run('grep -ri "出張費" /handbook');
await run('semantic_search "出張費を精算するときに必要な書類" /handbook');
await run('semantic_search "秘密情報が外部に出たときの初動" /');
await run('grep -ri "給与" /');
