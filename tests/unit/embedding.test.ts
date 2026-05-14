import { describe, expect, it } from 'vitest';
import { Tensor } from '@huggingface/transformers';
import {
  DEFAULT_EMBEDDING_PROVIDER_NAME,
  EMBEDDING_DIMS,
  NOMIC_MODEL_NAME,
  NOMIC_QUERY_MODEL_NAME,
  RURI_INDEX_MODEL_NAME,
  RURI_MODEL_NAME,
  RURI_QUERY_MODEL_NAME,
  RURI_QUERY_TOKENIZER_MODEL_NAME,
  assertEmbeddingDimensions,
  prefixTextForNomic,
  prefixTextForRuri,
  resolveEmbeddingProviderName,
  tensorToEmbeddingRows,
} from '../../src/core/embedding.js';

describe('embedding configuration', () => {
  it('uses Ruri by default and keeps split index/query model names', () => {
    expect(DEFAULT_EMBEDDING_PROVIDER_NAME).toBe('ruri');
    expect(RURI_INDEX_MODEL_NAME).toBe('cl-nagoya/ruri-v3-310m');
    expect(RURI_QUERY_MODEL_NAME).toBe('onnx-community/ruri-v3-310m-ONNX');
    expect(RURI_QUERY_TOKENIZER_MODEL_NAME).toBe(RURI_INDEX_MODEL_NAME);
    expect(RURI_MODEL_NAME).toBe(RURI_QUERY_MODEL_NAME);
    expect(EMBEDDING_DIMS).toBe(768);
  });

  it('defines Nomic as a selectable 768-dimensional provider', () => {
    expect(NOMIC_MODEL_NAME).toBe('nomic-ai/nomic-embed-text-v1.5');
    expect(NOMIC_QUERY_MODEL_NAME).toBe(NOMIC_MODEL_NAME);
    expect(resolveEmbeddingProviderName('nomic')).toBe('nomic');
    expect(resolveEmbeddingProviderName('RURI')).toBe('ruri');
    expect(() => resolveEmbeddingProviderName('unknown')).toThrow(
      /Invalid embedding provider/u,
    );
  });

  it('applies Ruri retrieval prefixes', () => {
    expect(prefixTextForRuri('認証情報を探す', 'query')).toBe(
      '検索クエリ: 認証情報を探す',
    );
    expect(prefixTextForRuri('APIキーの管理', 'document')).toBe(
      '検索文書: APIキーの管理',
    );
    expect(prefixTextForRuri('plain text', 'semantic')).toBe('plain text');
  });

  it('applies Nomic retrieval prefixes', () => {
    expect(prefixTextForNomic('retention policy', 'query')).toBe(
      'search_query: retention policy',
    );
    expect(prefixTextForNomic('The policy is retained for 30 days.', 'document')).toBe(
      'search_document: The policy is retained for 30 days.',
    );
    expect(prefixTextForNomic('plain text', 'semantic')).toBe('plain text');
  });

  it('validates fixed embedding dimensions', () => {
    expect(() =>
      assertEmbeddingDimensions(new Array(EMBEDDING_DIMS).fill(0)),
    ).not.toThrow();
    expect(() => assertEmbeddingDimensions([0])).toThrow(/768-dimensional/u);
  });

  it('converts pooled Transformers.js tensors into embedding rows', () => {
    const raw = new Float32Array(EMBEDDING_DIMS * 2);
    raw[0] = 0.25;
    raw[EMBEDDING_DIMS] = 0.5;
    const tensor = new Tensor('float32', raw, [2, EMBEDDING_DIMS]);

    const rows = tensorToEmbeddingRows(tensor, 2);

    expect(rows).toHaveLength(2);
    expect(rows[0]).toHaveLength(EMBEDDING_DIMS);
    expect(rows[0]?.[0]).toBe(0.25);
    expect(rows[1]?.[0]).toBe(0.5);
  });
});
