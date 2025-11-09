# weaviate-sbert-vectorizer

CPU-only [Sentence Transformers](https://www.sbert.net/index.html) vectorizer (embedding) service for Weaviate.

> Vectorization / Embedding is the process by which human-understandable data (ie. text) is converted into machine-readable numerical representations (vectors) for use in AI systems. Vectors are stored in specialized [vector databases](https://weaviate.io/blog/what-is-a-vector-database) that are optimized for rapid and nuanced information retrieval.

This service is designed for CPU vectorization on a single core using the [ONNX Runtime](https://onnxruntime.ai/docs/). No GPU is required.

Built images contain only the minimum dependencies required, and are thus significantly smaller than those [provided](https://docs.weaviate.io/weaviate/model-providers/transformers/embeddings#available-models) by Weaviate (by ~7GB).

## Usage

The images of this service are intended to be used with Weaviate's `text2vec-transformers` module:

```yaml
services:
  weaviate:
    environment:
      ENABLE_MODULES: text2vec-transformers
      TRANSFORMERS_INFERENCE_API: http://wstv:8080
  wstv:
    image: ghcr.io/metabronx/weaviate-sbert-vectorizer:all-MiniLM-L6-v2_quint8_avx2
```

## Configuration

The embedding model is configurable via the `MODEL_NAME` and `FILE_PATH` environment variables, which correspond:

```python
SentenceTransformer(
    MODEL_NAME,
    model_kwargs={"file_name": FILE_PATH},
    backend="onnx",
)
```

The default model is the 8-bit-quantized AVX2-optimized (`quint8_avx2`) variant of `sentence-transformers/all-MiniLM-L6-v2`.
