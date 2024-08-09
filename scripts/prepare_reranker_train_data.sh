#!/bin/bash

python data_creation/reranker_train.py \
    --input YOUR_TRAIN_DATA \
    --output YOUR_TRAIN_DATA_WITH_PROBS \
    --model YOUR_LM \
    --max_seq_length 2048 \
    --use_flash-attn
