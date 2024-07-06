import json
import os
import weaviate

from ipdb import launch_ipdb_on_exception
from qdrant_client import models

DATASET = os.getenv("DATASET", "quora")

def load_queries():
    queries = {}

    with open(f"data/{DATASET}/queries.jsonl", "r") as file:
        for line in file:
            row = json.loads(line)
            queries[row["_id"]] = {**row, "doc_ids": []}

    with open(f"data/{DATASET}/qrels/test.tsv", "r") as file:
        next(file)
        for line in file:
            query_id, doc_id, score = line.strip().split("\t")
            if int(score) > 0:
                queries[query_id]["doc_ids"].append(doc_id)

    queries_filtered = {}
    for query_id, query in queries.items():
        if len(query["doc_ids"]) > 0:
            queries_filtered[query_id] = query

    return queries_filtered


def main():
    n = 0
    hits = 0
    limit = 10
    number_of_queries = 100_000

    queries = load_queries()

    client = weaviate.connect_to_local()

    collection = client.collections.get(DATASET)

    def search_sparse(query, limit):
        with launch_ipdb_on_exception():
            response = collection.query.bm25(
                query=query,
                query_properties=["text"],
                limit=limit
            )

            return [
                hit.properties for hit in response.objects
            ]

    recalls = []
    precisions = []
    num_queries = 0

    for idx, query in enumerate(queries.values()):
        if idx >= number_of_queries:
            print(f"Processed {number_of_queries} queries, stopping...")
            break

        num_queries += 1

        result = search_sparse(query["text"], limit)
        found_ids = []

        for hit in result:
            found_ids.append(str(hit["idx"]))

        query_hits = 0
        for doc_id in query["doc_ids"]:
            n += 1
            if doc_id in found_ids:
                hits += 1
                query_hits += 1

        recalls.append(
            query_hits / len(query["doc_ids"])
        )

        precisions.append(
            query_hits / limit
        )

        print(f"Processing query: {query}, hits: {query_hits}")

    print(f"Total hits: {hits} out of {n}, which is {hits/n}")

    print(f"Precision: {hits/(num_queries * limit)}")

    average_precision = sum(precisions) / len(precisions)

    print(f"Average precision: {average_precision}")

    average_recall = sum(recalls) / len(recalls)

    print(f"Average recall: {average_recall}")


if __name__ == "__main__":
    main()
