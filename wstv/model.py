import os

from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("HF_MODEL_NAME", "all-MiniLM-L6-v2")
FILE_PATH = os.getenv("HF_FILE_PATH", "onnx/model_quint8_avx2.onnx")

MODEL = SentenceTransformer(
    MODEL_NAME,
    model_kwargs={
        "file_name": FILE_PATH,
        "provider": "CPUExecutionProvider",
    },
    device="cpu",
    backend="onnx",
)
