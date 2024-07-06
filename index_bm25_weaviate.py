import os
import json

from typing import Iterable

import tqdm
import weaviate

from weaviate.classes.config import Property, DataType


DATASET = os.getenv("DATASET", "quora")


def read_file(file_name: str) -> Iterable[str]:
    with open(file_name, "r") as file:
        for line in file:
            row = json.loads(line)
            yield row["_id"], row["text"]


def main():
    file_name = f"data/{DATASET}/corpus.jsonl"  # MS MARCO collection

    client = weaviate.connect_to_local()

    client.collections.delete(DATASET)

    client.collections.create(
        DATASET,
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="idx", data_type=DataType.INT)
        ]
    )

    collection = client.collections.get(DATASET)

    for idx, text in tqdm.tqdm(read_file(file_name)):
        collection.data.insert({
            "text": text,
            "idx": int(idx)
        })


if __name__ == "__main__":
    main()
