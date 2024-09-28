#!/bin/bash

python evaluae_passages.py \
    --data data/evaluation/dev \
    --rerank_model RERANK_MODEL \
    --reranker_revision RERANKER_REVISION \
    --ctxs_num 0