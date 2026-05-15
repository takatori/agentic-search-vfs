import type { Client } from '@opensearch-project/opensearch';
import { describe, expect, it, vi } from 'vitest';
import type { EmbeddingProvider } from '../../src/core/embedding.js';
import {
  findSemanticMatchingFilesWithScope,
  parseSemanticSearchArgv,
  parseSemanticSearchLimit,
} from '../../src/core/semantic-search.js';

type SearchResponse = {
  body: {
    hits: {
      hits: Array<{
        _score?: number;
        _source?: {
          slug?: string;
        };
        inner_hits?: {
          chunks?: {
            hits?: {
              hits?: Array<{
                _source?: {
                  chunk_id?: number;
                  text?: string;
                };
              }>;
            };
          };
        };
      }>;
    };
  };
};

const testEmbeddings: EmbeddingProvider = {
  async embed() {
    return new Array(768).fill(0);
  },
  async embedMany(texts) {
    return texts.map(() => new Array(768).fill(0));
  },
};

describe('semantic_search helpers', () => {
  it('parses query with the default path', () => {
    expect(parseSemanticSearchArgv(['文春砲'])).toEqual({
      query: '文春砲',
      pathArg: '.',
      limit: 5,
    });
  });

  it('parses query, optional path, and limit flag', () => {
    expect(parseSemanticSearchArgv(['-n', '3', '自然文の質問', '/jqara/test'])).toEqual({
      query: '自然文の質問',
      pathArg: '/jqara/test',
      limit: 3,
    });
  });

  it('rejects invalid limits', () => {
    expect(() => parseSemanticSearchLimit('0')).toThrow(
      /Invalid SEMANTIC_SEARCH_RESULT_LIMIT/u,
    );
    expect(() => parseSemanticSearchLimit('nan')).toThrow(
      /Invalid SEMANTIC_SEARCH_RESULT_LIMIT/u,
    );
  });

  it('rejects missing query and extra positional args', () => {
    expect(() => parseSemanticSearchArgv([])).toThrow(/missing query/u);
    expect(() => parseSemanticSearchArgv(['a', '/x', '/y'])).toThrow(
      /too many arguments/u,
    );
  });

  it('queries nested chunk vectors and returns the matched chunk text', async () => {
    const searchMock = vi
      .fn<(request: object) => Promise<SearchResponse>>()
      .mockResolvedValueOnce({
        body: {
          hits: {
            hits: [
              {
                _score: 12,
                _source: { slug: 'docs/runbook' },
                inner_hits: {
                  chunks: {
                    hits: {
                      hits: [
                        {
                          _source: {
                            chunk_id: 2,
                            text: 'Restart the streaming worker after quota errors.',
                          },
                        },
                      ],
                    },
                  },
                },
              },
            ],
          },
        },
      });

    const client = { search: searchMock } as object as Client;
    const hits = await findSemanticMatchingFilesWithScope(
      client,
      'streaming quota',
      { prefix: { slug: 'docs/' } },
      5,
      { embeddings: testEmbeddings },
    );

    expect(searchMock).toHaveBeenCalledWith(
      expect.objectContaining({
        body: expect.objectContaining({
          query: {
            bool: {
              filter: [{ prefix: { slug: 'docs/' } }],
              must: [
                {
                  nested: expect.objectContaining({
                    path: 'chunks',
                    query: {
                      knn: {
                        'chunks.embedding': expect.objectContaining({
                          k: 50,
                        }),
                      },
                    },
                    inner_hits: expect.objectContaining({ size: 1 }),
                  }),
                },
              ],
            },
          },
        }),
      }),
    );
    expect(hits).toEqual([
      {
        slug: 'docs/runbook',
        score: 12,
        content: 'Restart the streaming worker after quota errors.',
        chunkId: 2,
      },
    ]);
  });
});
