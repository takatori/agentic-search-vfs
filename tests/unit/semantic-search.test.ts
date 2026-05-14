import { describe, expect, it } from 'vitest';
import {
  parseSemanticSearchArgv,
  parseSemanticSearchLimit,
} from '../../src/core/semantic-search.js';

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
});
