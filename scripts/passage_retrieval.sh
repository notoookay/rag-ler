#!/bin/bash
# NOTE: use faiss-gpu will take approximately 70GB GPU memory

python retrieval/passage_retrieval.py \
    --model_name_or_path facebook/contriever-msmarco \
    --passages YOUR_CORPUS_DATA \
    --passages_embeddings "YOUR_CORPUS_EMBEDDINGS/wikipedia_embeddings/*" \
    --data "YOUR_RETRIEVAL_DATA/*" \
    --n_docs 30 \
    --validate_retrieval \
    --per_gpu_batch_size 128 \
    --use_faiss_gpu \
    --output_dir OUTPUT_DIR