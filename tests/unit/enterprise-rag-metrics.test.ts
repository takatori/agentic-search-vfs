import { describe, expect, it } from 'vitest';
import {
  documentRecall,
  extractDocumentIdsFromPaths,
  invalidExtraDocCount,
  parseJudgeResponseText,
} from '../../experiments/enterprise-rag/metrics.js';

describe('EnterpriseRAG eval metrics', () => {
  it('extracts EnterpriseRAG document IDs from cited .mdx paths', () => {
    expect(
      extractDocumentIdsFromPaths([
        '/enterprise-rag/test/github/project/dsid_abc123.mdx',
        '/enterprise-rag/test/slack/topic/dsid_def456.mdx).',
        '/enterprise-rag/test/slack/topic/not-a-doc.mdx',
        '/enterprise-rag/test/slack/topic/dsid_abc123.mdx',
      ]),
    ).toEqual(['dsid_abc123', 'dsid_def456']);
  });

  it('calculates document recall and invalid extra docs', () => {
    const returned = ['dsid_a', 'dsid_extra', 'dsid_a'];
    const expected = ['dsid_a', 'dsid_b'];

    expect(documentRecall(returned, expected)).toBe(0.5);
    expect(invalidExtraDocCount(returned, expected)).toBe(1);
  });

  it('parses judge JSON with surrounding text and computes completeness', () => {
    const parsed = parseJudgeResponseText(
      [
        'Here is the JSON:',
        JSON.stringify({
          answer_correct: true,
          fact_judgments: [
            { fact: 'Fact A', supported: true, reason: 'stated' },
            { fact: 'Fact B', supported: false, reason: 'missing' },
          ],
          completeness: 1,
          reason: 'Mostly correct',
        }),
      ].join('\n'),
      ['Fact A', 'Fact B'],
    );

    expect(parsed.answer_correct).toBe(true);
    expect(parsed.completeness).toBe(0.5);
    expect(parsed.fact_judgments).toEqual([
      { fact: 'Fact A', supported: true, reason: 'stated' },
      { fact: 'Fact B', supported: false, reason: 'missing' },
    ]);
  });

  it('throws on unparseable judge responses', () => {
    expect(() => parseJudgeResponseText('not json', ['Fact A'])).toThrow(
      /JSON object/u,
    );
    expect(() =>
      parseJudgeResponseText('{"fact_judgments":[]}', ['Fact A']),
    ).toThrow(/answer_correct/u);
  });
});
