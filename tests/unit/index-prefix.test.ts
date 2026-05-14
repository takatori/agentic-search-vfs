import { describe, expect, it } from 'vitest';
import {
  getOpenSearchFsIndexNames,
  resolveOpenSearchFsIndexPrefix,
} from '../../src/opensearchfs-constants.js';

describe('OpenSearchFs index prefix', () => {
  it('defaults to the production index names', () => {
    expect(getOpenSearchFsIndexNames('opensearchfs')).toEqual({
      files: 'opensearchfs-chunks',
      meta: 'opensearchfs-meta',
    });
  });

  it('derives isolated experiment index names', () => {
    expect(getOpenSearchFsIndexNames('opensearchfs-jqara')).toEqual({
      files: 'opensearchfs-jqara-chunks',
      meta: 'opensearchfs-jqara-meta',
    });
  });

  it('rejects unsafe index prefixes', () => {
    expect(() => resolveOpenSearchFsIndexPrefix('BadPrefix')).toThrow(
      /Invalid OPENSEARCHFS_INDEX_PREFIX/u,
    );
    expect(() => resolveOpenSearchFsIndexPrefix('-bad')).toThrow(
      /Invalid OPENSEARCHFS_INDEX_PREFIX/u,
    );
  });
});
