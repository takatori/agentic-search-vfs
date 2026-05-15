import type { Client } from '@opensearch-project/opensearch';
import { describe, expect, it, vi } from 'vitest';
import { OpenSearchFs } from '../../src/core/opensearchfs.js';

type SearchResponse = {
  body: {
    hits: {
      hits: Array<{
        _source?: {
          slug?: string;
          content?: string;
        };
      }>;
    };
  };
};

function makeFileHit(slug: string): SearchResponse['body']['hits']['hits'][number] {
  return { _source: { slug } };
}

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
});
