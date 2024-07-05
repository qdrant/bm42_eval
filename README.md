

# BM42 vs BM25 benchmark


## Introduction

Download dataset:

```bash
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip
mkdir -p data
mv quora.zip data/

cd data
unzip quora.zip
```

## Install dependencies

```bash
pip install -r requirements.txt
```

(Note: for gpu inference see [fastembed](https://github.com/qdrant/fastembed?tab=readme-ov-file#%EF%B8%8F-fastembed-on-a-gpu))


## BM25

Bm25 version uses `tantivy` library for indexing and search. 

```bash
python index_bm25.py

python evaluate-bm25.py
```

Results we got:

```
Total hits: 9394 out of 15675, which is 0.5992982456140351
Precision: 0.09394
Average precision: 0.09394000000001176
Average recall: 0.7131305557181846
```

## BM42

BM42 uses `fastembed` implementation for inference, and `qdrant` for indexing and search.
IDF are calculated using inside Qdrant.


```bash
# Run qdrant
docker run --rm -d --network=host qdrant/qdrant:v1.10.0

python index_bm42.py

python evaluate-bm42.py
```

Results we got:

```
Total hits: 11488 out of 15675, which is 0.7328867623604466
Precision: 0.11488
Average precision: 0.11488000000000238
Average recall: 0.8515208038970792
```
