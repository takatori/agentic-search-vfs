import {
  AutoModel,
  AutoTokenizer,
  Tensor,
  layer_norm,
  mean_pooling,
  type DataType,
} from '@huggingface/transformers';

export const DEFAULT_EMBEDDING_PROVIDER_NAME = 'ruri';
export const RURI_INDEX_MODEL_NAME = 'cl-nagoya/ruri-v3-310m';
export const RURI_QUERY_MODEL_NAME = 'onnx-community/ruri-v3-310m-ONNX';
export const RURI_QUERY_TOKENIZER_MODEL_NAME = RURI_INDEX_MODEL_NAME;
export const RURI_MODEL_NAME = RURI_QUERY_MODEL_NAME;
export const NOMIC_MODEL_NAME = 'nomic-ai/nomic-embed-text-v1.5';
export const NOMIC_QUERY_MODEL_NAME = NOMIC_MODEL_NAME;
export const EMBEDDING_DIMS = 768;

export type EmbeddingProviderName = 'ruri' | 'nomic';
export type EmbeddingInputType = 'query' | 'document' | 'semantic';

export interface EmbeddingProvider {
  embed(text: string, inputType?: EmbeddingInputType): Promise<number[]>;
  embedMany(
    texts: string[],
    inputType?: EmbeddingInputType,
  ): Promise<number[][]>;
}

const QUERY_DTYPES = new Set<DataType>([
  'auto',
  'fp32',
  'fp16',
  'q8',
  'int8',
  'uint8',
  'q4',
  'bnb4',
  'q4f16',
  'q2',
  'q2f16',
  'q1',
  'q1f16',
]);

export function prefixTextForRuri(
  text: string,
  inputType: EmbeddingInputType,
): string {
  if (inputType === 'query') return `検索クエリ: ${text}`;
  if (inputType === 'document') return `検索文書: ${text}`;
  return text;
}

export function prefixTextForNomic(
  text: string,
  inputType: EmbeddingInputType,
): string {
  if (inputType === 'query') return `search_query: ${text}`;
  if (inputType === 'document') return `search_document: ${text}`;
  return text;
}

export function resolveEmbeddingProviderName(
  value =
    process.env.QUERY_EMBEDDING_PROVIDER ??
    process.env.EMBEDDING_PROVIDER ??
    DEFAULT_EMBEDDING_PROVIDER_NAME,
): EmbeddingProviderName {
  const normalized = value.trim().toLowerCase();
  if (normalized === 'ruri' || normalized === 'nomic') return normalized;
  throw new Error(
    `Invalid embedding provider "${value}". Expected one of: ruri, nomic.`,
  );
}

type RuriEmbeddingProviderOptions = {
  model?: string;
  tokenizerModel?: string;
  dtype?: DataType;
};

type RuriExtractor = {
  tokenizer: Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>;
  model: Awaited<ReturnType<typeof AutoModel.from_pretrained>>;
};

type ModelOutput = {
  last_hidden_state?: Tensor;
  logits?: Tensor;
  token_embeddings?: Tensor;
};

export class RuriEmbeddingProvider implements EmbeddingProvider {
  private extractorPromise?: Promise<RuriExtractor>;

  constructor(private readonly options: RuriEmbeddingProviderOptions = {}) {}

  async embed(
    text: string,
    inputType: EmbeddingInputType = 'semantic',
  ): Promise<number[]> {
    const [embedding] = await this.embedMany([text], inputType);
    if (!embedding) {
      throw new Error('Ruri query embedding provider returned no embedding.');
    }
    return embedding;
  }

  async embedMany(
    texts: string[],
    inputType: EmbeddingInputType = 'semantic',
  ): Promise<number[][]> {
    if (texts.length === 0) return [];

    const { tokenizer, model } = await this.getExtractor();
    const modelInputs = tokenizer(
      texts.map((text) => prefixTextForRuri(text, inputType)),
      { padding: true, truncation: true },
    ) as Record<string, Tensor>;
    const runModel = model as unknown as (
      inputs: Record<string, Tensor>,
    ) => Promise<ModelOutput>;
    const outputs = await runModel(modelInputs);
    const hidden =
      outputs.last_hidden_state ?? outputs.logits ?? outputs.token_embeddings;
    if (!hidden) {
      throw new Error('Ruri query model did not return hidden states.');
    }
    const attentionMask = modelInputs.attention_mask;
    if (!attentionMask) {
      throw new Error('Ruri query tokenizer did not return an attention mask.');
    }
    const output = mean_pooling(hidden, attentionMask).normalize(2, -1);
    return tensorToEmbeddingRows(output, texts.length);
  }

  private getExtractor(): Promise<RuriExtractor> {
    if (!this.extractorPromise) {
      const model =
        this.options.model ?? process.env.RURI_QUERY_MODEL ?? RURI_QUERY_MODEL_NAME;
      const tokenizerModel =
        this.options.tokenizerModel ??
        process.env.RURI_QUERY_TOKENIZER_MODEL ??
        RURI_QUERY_TOKENIZER_MODEL_NAME;
      const dtype =
        this.options.dtype ??
        parseQueryDtype(process.env.RURI_QUERY_DTYPE ?? 'fp32', 'RURI_QUERY_DTYPE');
      this.extractorPromise = Promise.all([
        AutoTokenizer.from_pretrained(tokenizerModel),
        AutoModel.from_pretrained(model, { dtype }),
      ]).then(([tokenizer, loadedModel]) => ({
        tokenizer,
        model: loadedModel,
      }));
    }
    return this.extractorPromise;
  }
}

type NomicEmbeddingProviderOptions = {
  model?: string;
  tokenizerModel?: string;
  dtype?: DataType;
};

export class NomicEmbeddingProvider implements EmbeddingProvider {
  private extractorPromise?: Promise<RuriExtractor>;

  constructor(private readonly options: NomicEmbeddingProviderOptions = {}) {}

  async embed(
    text: string,
    inputType: EmbeddingInputType = 'semantic',
  ): Promise<number[]> {
    const [embedding] = await this.embedMany([text], inputType);
    if (!embedding) {
      throw new Error('Nomic query embedding provider returned no embedding.');
    }
    return embedding;
  }

  async embedMany(
    texts: string[],
    inputType: EmbeddingInputType = 'semantic',
  ): Promise<number[][]> {
    if (texts.length === 0) return [];

    const { tokenizer, model } = await this.getExtractor();
    const modelInputs = tokenizer(
      texts.map((text) => prefixTextForNomic(text, inputType)),
      { padding: true, truncation: true },
    ) as Record<string, Tensor>;
    const runModel = model as unknown as (
      inputs: Record<string, Tensor>,
    ) => Promise<ModelOutput>;
    const outputs = await runModel(modelInputs);
    const hidden =
      outputs.last_hidden_state ?? outputs.logits ?? outputs.token_embeddings;
    if (!hidden) {
      throw new Error('Nomic query model did not return hidden states.');
    }
    const attentionMask = modelInputs.attention_mask;
    if (!attentionMask) {
      throw new Error('Nomic query tokenizer did not return an attention mask.');
    }
    const pooled = mean_pooling(hidden, attentionMask);
    const output = layer_norm(pooled, [pooled.dims[1] ?? EMBEDDING_DIMS]).normalize(
      2,
      -1,
    );
    return tensorToEmbeddingRows(output, texts.length, EMBEDDING_DIMS, 'Nomic');
  }

  private getExtractor(): Promise<RuriExtractor> {
    if (!this.extractorPromise) {
      const model =
        this.options.model ??
        process.env.NOMIC_QUERY_MODEL ??
        process.env.NOMIC_MODEL ??
        NOMIC_QUERY_MODEL_NAME;
      const tokenizerModel =
        this.options.tokenizerModel ??
        process.env.NOMIC_QUERY_TOKENIZER_MODEL ??
        model;
      const dtype =
        this.options.dtype ??
        parseQueryDtype(process.env.NOMIC_QUERY_DTYPE ?? 'fp32', 'NOMIC_QUERY_DTYPE');
      this.extractorPromise = Promise.all([
        AutoTokenizer.from_pretrained(tokenizerModel),
        AutoModel.from_pretrained(model, { dtype }),
      ]).then(([tokenizer, loadedModel]) => ({
        tokenizer,
        model: loadedModel,
      }));
    }
    return this.extractorPromise;
  }
}

export function createEmbeddingProvider(): EmbeddingProvider {
  const provider = resolveEmbeddingProviderName();
  if (provider === 'ruri') return new RuriEmbeddingProvider();
  return new NomicEmbeddingProvider();
}

export function assertEmbeddingDimensions(embedding: number[]): void {
  if (embedding.length !== EMBEDDING_DIMS) {
    throw new Error(
      `Expected ${EMBEDDING_DIMS}-dimensional embeddings, got ${embedding.length}.`,
    );
  }
}

export function tensorToEmbeddingRows(
  tensor: Tensor,
  expectedRows: number,
  expectedDims = EMBEDDING_DIMS,
  providerName = 'embedding model',
): number[][] {
  const [rows, dims] = tensor.dims;
  if (tensor.dims.length !== 2 || rows !== expectedRows || dims !== expectedDims) {
    throw new Error(
      `Expected ${providerName} output shape [${expectedRows}, ${expectedDims}], got [${tensor.dims.join(', ')}].`,
    );
  }

  const data = Array.from(tensor.data, Number);
  const embeddings: number[][] = [];
  for (let row = 0; row < rows; row += 1) {
    const start = row * expectedDims;
    const embedding = data.slice(start, start + expectedDims);
    assertEmbeddingDimensions(embedding);
    embeddings.push(embedding);
  }
  return embeddings;
}

function parseQueryDtype(value: string, envName: string): DataType {
  if (QUERY_DTYPES.has(value as DataType)) return value as DataType;
  throw new Error(
    `Invalid ${envName} "${value}". Expected one of: ${[...QUERY_DTYPES].join(', ')}.`,
  );
}
