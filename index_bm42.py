import os
import json

import tqdm
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from typing import Iterable

from ipdb import launch_ipdb_on_exception

DATASET = os.getenv("DATASET", "quora")


def read_file(file_name: str) -> Iterable[str]:
    with open(file_name, "r") as file:
        for line in file:
            row = json.loads(line)
            yield row["_id"], row["text"]


def read_embedded(file_name: str) -> Iterable[models.PointStruct]:
    model = SparseTextEmbedding(
        model_name="Qdrant/bm42-all-minilm-l6-v2-attentions"
    )

    def read_texts():
        n = 0
        for _, text in read_file(file_name):
            n += 1
            yield text

    for ((idx, text), embedding) in zip(read_file(file_name), model.embed(tqdm.tqdm(read_texts()), batch_size=32)):
        doc = models.PointStruct(
            id=int(idx),
            vector={
                "bm42": models.SparseVector(
                    values=embedding.values.tolist(),
                    indices=embedding.indices.tolist()
                )
            }
        )

        yield doc


def main():
    file_name = f"data/{DATASET}/corpus.jsonl"  # MS MARCO collection
    client = QdrantClient(url="http://localhost:6333", prefer_grpc=True)

    client.delete_collection(collection_name=DATASET)

    client.create_collection(
        collection_name=DATASET,
        vectors_config={},
        sparse_vectors_config={
            "bm42": models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        }
    )

    with launch_ipdb_on_exception():
        for point in tqdm.tqdm(read_embedded(file_name)):
            client.upsert(collection_name=DATASET, points=[point], wait=False)


if __name__ == "__main__":
    main()
