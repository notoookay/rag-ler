#!/bin/bash

python retrieval/passage_retrieval_bm25.py \
    --passages YOUR_CORPUS_DATA \
    --passages_indexes "YOUR_CORPUS_INDEXES/wikipedia_indexes/*" \
    --data YOUR_RETRIEVAL_DATA \
    --n_docs 30 \
    --validate_retrieval \
    --output_dir OUTPUT_DIR