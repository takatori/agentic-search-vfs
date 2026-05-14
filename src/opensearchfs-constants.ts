export const OPENSEARCHFS_DEFAULT_INDEX_PREFIX = 'opensearchfs';
export const OPENSEARCHFS_INDEX_PREFIX_ENV = 'OPENSEARCHFS_INDEX_PREFIX';
export const OPENSEARCHFS_PATH_TREE_DOC_ID = '__path_tree__';

export type OpenSearchFsIndexNames = {
  files: string;
  meta: string;
};

export function resolveOpenSearchFsIndexPrefix(
  value = process.env[OPENSEARCHFS_INDEX_PREFIX_ENV],
): string {
  const prefix = value?.trim() || OPENSEARCHFS_DEFAULT_INDEX_PREFIX;
  if (!/^[a-z0-9][a-z0-9_-]*$/u.test(prefix)) {
    throw new Error(
      `Invalid ${OPENSEARCHFS_INDEX_PREFIX_ENV} "${prefix}". Use lowercase letters, numbers, "_" or "-", starting with a letter or number.`,
    );
  }
  return prefix;
}

export function getOpenSearchFsIndexNames(
  prefix = resolveOpenSearchFsIndexPrefix(),
): OpenSearchFsIndexNames {
  return {
    files: `${prefix}-chunks`,
    meta: `${prefix}-meta`,
  };
}
