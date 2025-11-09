from typing import TYPE_CHECKING, cast

from fastapi import FastAPI, status
from pydantic import BaseModel

from .model import FILE_PATH, MODEL, MODEL_NAME

if TYPE_CHECKING:
    from numpy import dtype, float32, ndarray

    type Embeddings = ndarray[tuple[int, int], dtype[float32]]
    type Embedding = ndarray[tuple[int], dtype[float32]]


app = FastAPI()


@app.get("/.well-known/live", status_code=status.HTTP_204_NO_CONTENT)
@app.get("/.well-known/ready", status_code=status.HTTP_204_NO_CONTENT)
async def live_and_ready() -> None:
    return None


@app.get("/meta")
def meta() -> dict[str, dict[str, str | None] | str]:
    return {
        "model": {"model_name": MODEL_NAME, "model_type": None},
        "model_path": FILE_PATH,
    }


class VectorParamsConfig(BaseModel):
    pooling_strategy: str | None = None
    task_type: str | None = None
    dimensions: int | None = None


class VectorParams(BaseModel):
    text: str
    config: VectorParamsConfig | None = None

    def is_query(self) -> bool:
        return self.config is not None and self.config.task_type == "query"

    def get_output_dimensions(self) -> int | None:
        if self.config is not None and self.config.dimensions is not None:
            return self.config.dimensions


@app.post("/vectors")
@app.post("/vectors/")
async def vectorize(vector: VectorParams) -> dict[str, int | list[float] | str]:
    encode_fn = MODEL.encode_query if vector.is_query() else MODEL.encode_document
    embedding: "Embedding" = cast(
        "Embeddings",
        encode_fn(
            [vector.text],
            convert_to_tensor=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            truncate_dim=vector.get_output_dimensions(),
        ),
    )[0]

    return {"text": vector.text, "vector": embedding.tolist(), "dim": len(embedding)}
