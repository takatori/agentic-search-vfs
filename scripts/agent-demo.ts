/**
 * AI agent demo over OpenSearchFs using Claude Agent SDK.
 *
 * No ANTHROPIC_API_KEY is needed when this runs from an authenticated
 * `claude` CLI environment; the SDK can reuse that auth.
 *
 * Usage:
 *   npx tsx scripts/agent-demo.ts <PROFILE> "<prompt>"
 *
 * Example:
 *   npx tsx scripts/agent-demo.ts BILLING "請求関連のドキュメントを要約して"
 */
import 'dotenv/config';
import { z } from 'zod';
import {
  createSdkMcpServer,
  query,
  tool,
} from '@anthropic-ai/claude-agent-sdk';
import { Bash, defineCommand } from 'just-bash';
import { runOpenSearchGrep } from '../src/core/grep.js';
import { OpenSearchFs } from '../src/core/opensearchfs.js';
import { runOpenSearchSemanticSearch } from '../src/core/semantic-search.js';
import { createOpenSearchClient } from '../src/opensearch-adapter/client.js';
import { initSessionTree } from '../src/session.js';

const profile = process.argv[2];
const userPrompt = process.argv[3];
if (!profile || !userPrompt) {
  console.error('Usage: tsx scripts/agent-demo.ts <PROFILE> "<prompt>"');
  process.exit(1);
}

const MAX_TOOL_OUTPUT_CHARS = Number(
  process.env.AGENT_DEMO_MAX_TOOL_OUTPUT_CHARS ?? 8000,
);

const client = createOpenSearchClient(profile);
const session = await initSessionTree(client, profile);
const fs = new OpenSearchFs({
  client,
  files: session.files,
  dirs: session.dirs,
});
const grep = defineCommand('grep', (args, ctx) =>
  runOpenSearchGrep(args, ctx, fs, client),
);
const semanticSearch = defineCommand('semantic_search', (args, ctx) =>
  runOpenSearchSemanticSearch(args, ctx, fs, client),
);
const bash = new Bash({ fs, cwd: '/', customCommands: [grep, semanticSearch] });

const osfsServer = createSdkMcpServer({
  name: 'osfs',
  version: '0.1.0',
  alwaysLoad: true,
  tools: [
    tool(
      'docs_bash',
      'Run a read-only bash command (ls, cat, grep, semantic_search, pwd, cd, head, tail, find) on the documentation virtual filesystem rooted at "/". Use this to explore docs and read files.',
      { command: z.string().describe('Bash command to execute') },
      async ({ command }) => {
        const { stdout, stderr, exitCode } = await bash.exec(command);
        const text =
          `exitCode: ${exitCode}\n` +
          (stdout || '(empty stdout)') +
          (stderr ? `\n[stderr]\n${stderr}` : '');
        return {
          content: [{ type: 'text', text: clipToolOutput(text) }],
        };
      },
    ),
  ],
});

const systemPrompt = `You are a documentation assistant with access to a virtual filesystem of internal docs via the "docs_bash" tool.
- Start by exploring with \`ls /\` to see what is available.
- Use \`grep -ri "<term>" /\` to search across files; use \`cat <path>\` to read specific files.
- Use \`semantic_search "<natural language query>" /\` when exact keywords are uncertain, the user asks a natural-language question, or grep does not find enough candidates. You do not need to start with semantic_search if grep or directory exploration is more appropriate.
- Some paths may not exist for the current user because access is controlled by OpenSearch document-level security. Do not assume hidden content.
- After gathering enough context, write the final answer in Japanese, citing file paths.`;

console.log(
  `\n[profile=${profile}] user: ${userPrompt}\n`,
);

const result = query({
  prompt: userPrompt,
  options: {
    cwd: process.cwd(),
    systemPrompt,
    tools: [],
    mcpServers: { osfs: osfsServer },
    permissionMode: 'bypassPermissions',
    allowDangerouslySkipPermissions: true,
  },
});

try {
  for await (const msg of result) {
    if (msg.type === 'assistant') {
      for (const block of msg.message.content) {
        if (block.type === 'text' && block.text.trim()) {
          console.log(`assistant: ${block.text}\n`);
        } else if (block.type === 'tool_use') {
          const input = JSON.stringify(block.input);
          console.log(`tool_use: ${block.name} ${input}`);
        }
      }
    } else if (msg.type === 'user') {
      const content = msg.message.content;
      if (Array.isArray(content)) {
        for (const block of content) {
          if (block.type === 'tool_result') {
            console.log(
              `tool_result:\n${extractToolResultText(block).slice(0, 500)}\n`,
            );
          }
        }
      }
    } else if (msg.type === 'result') {
      console.log(`\n--- done (${msg.subtype}) ---`);
    }
  }
} finally {
  await client.close();
}

function clipToolOutput(text: string): string {
  if (text.length <= MAX_TOOL_OUTPUT_CHARS) return text;
  return `${text.slice(0, MAX_TOOL_OUTPUT_CHARS)}\n...[truncated ${text.length - MAX_TOOL_OUTPUT_CHARS} chars]`;
}

function extractToolResultText(block: { content?: unknown }): string {
  if (typeof block.content === 'string') return block.content;
  if (Array.isArray(block.content)) {
    return block.content
      .map((entry) => {
        if (
          entry !== null &&
          typeof entry === 'object' &&
          'type' in entry &&
          entry.type === 'text' &&
          'text' in entry &&
          typeof entry.text === 'string'
        ) {
          return entry.text;
        }
        return '';
      })
      .join('');
  }
  return '';
}
