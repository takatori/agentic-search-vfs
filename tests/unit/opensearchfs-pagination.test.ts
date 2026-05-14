import type { Client } from '@opensearch-project/opensearch';
import { describe, expect, it, vi } from 'vitest';
import type { EmbeddingProvider } from '../../src/core/embedding.js';
import { OpenSearchFs } from '../../src/core/opensearchfs.js';

type SearchResponse = {
  body: {
    hits: {
      hits: Array<{
        _score?: number;
        _source?: {
          slug?: string;
          content?: string;
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

function makeFileHit(slug: string): SearchResponse['body']['hits']['hits'][number] {
  return { _source: { slug } };
}

const testEmbeddings: EmbeddingProvider = {
  async embed() {
    return new Array(768).fill(0);
  },
  async embedMany(texts) {
    return texts.map(() => new Array(768).fill(0));
  },
};

describe('OpenSearchFs pagination', () => {
  it('paginates coarse grep search and deduplicates slugs', async () => {
    const firstPage = [
      ...Array.from({ length: 999 }, (_, i) => makeFileHit(`alpha-${i}`)),
      makeFileHit('beta'),
    ];
    const secondPage = [makeFileHit('beta'), makeFileHit('gamma')];

    const searchMock = vi
      .fn<(request: object) => Promise<SearchResponse>>()
      .mockResolvedValueOnce({ body: { hits: { hits: firstPage } } })
      .mockResolvedValueOnce({ body: { hits: { hits: secondPage } } });

    const client = { search: searchMock } as object as Client;
    const fs = new OpenSearchFs({
      client,
      files: new Set(),
      dirs: new Map(),
      embeddings: testEmbeddings,
    });

    const scopeSlugs = [
      ...Array.from({ length: 999 }, (_, i) => `alpha-${i}`),
      'beta',
      'gamma',
    ];
    const out = await fs.findMatchingFiles(
      { pattern: 'access_token', ignoreCase: true, fixedStrings: false },
      scopeSlugs,
    );

    expect(new Set(out)).toEqual(new Set(scopeSlugs));
    expect(searchMock).toHaveBeenCalledTimes(2);

    expect(searchMock.mock.calls[0]?.[0]).toMatchObject({
      body: {
        size: 1000,
        sort: [{ slug: { order: 'asc' } }],
      },
    });
    expect(searchMock.mock.calls[0]?.[0]).not.toMatchObject({
      body: { search_after: expect.anything() },
    });
    expect(searchMock.mock.calls[1]?.[0]).toMatchObject({
      body: {
        search_after: ['beta'],
      },
    });
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
    const fs = new OpenSearchFs({
      client,
      files: new Set(),
      dirs: new Map(),
      embeddings: testEmbeddings,
    });

    const hits = await fs.findSemanticMatchingFilesWithScope(
      'streaming quota',
      { prefix: { slug: 'docs/' } },
      5,
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

  it('falls back to legacy top-level embeddings when nested chunks are absent', async () => {
    const missingNestedError = new Error(
      'failed to create query: [nested] failed to find nested object under path [chunks]',
    );
    const searchMock = vi
      .fn<(request: object) => Promise<SearchResponse>>()
      .mockRejectedValueOnce(missingNestedError)
      .mockResolvedValueOnce({
        body: {
          hits: {
            hits: [
              {
                _score: 7,
                _source: {
                  slug: 'jqara/test/passage',
                  content: 'Legacy passage content from a flat embedding index.',
                },
              },
            ],
          },
        },
      });

    const client = { search: searchMock } as object as Client;
    const fs = new OpenSearchFs({
      client,
      files: new Set(),
      dirs: new Map(),
      embeddings: testEmbeddings,
    });

    const hits = await fs.findSemanticMatchingFilesWithScope(
      'legacy query',
      { prefix: { slug: 'jqara/test/' } },
      3,
    );

    expect(searchMock).toHaveBeenCalledTimes(2);
    expect(searchMock.mock.calls[1]?.[0]).toMatchObject({
      body: {
        size: 3,
        _source: ['slug', 'content'],
        query: {
          bool: {
            filter: [{ prefix: { slug: 'jqara/test/' } }],
            must: [
              {
                knn: {
                  embedding: expect.objectContaining({
                    k: 30,
                  }),
                },
              },
            ],
          },
        },
      },
    });
    expect(hits).toEqual([
      {
        slug: 'jqara/test/passage',
        score: 7,
        content: 'Legacy passage content from a flat embedding index.',
        chunkId: undefined,
      },
    ]);
  });
});
