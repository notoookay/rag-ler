#!/bin/bash

MODEL=YOUR_LLM_MODEL
RERANK_MODEL=YOUR_RERANK_MODEL
CTXS_NUM=5
EXPERIMENT_NAME="Evaluation"

python run_llm.py \
    --data YOUR_EVAL_DATA \
    --output OUTPUT_DIR \
    --model ${MODEL} \
    --use_slow_tokenizer \
    --num_proc 20 \
    --use_flash-attn \
    --max_new_tokens 100 \
    --use_context \
    --ctxs_num ${CTXS_NUM} \
    --rerank \
    --rerank_model ${RERANK_MODEL} \
    --experiment_name "${EXPERIMENT_NAME}" \
    --run_name "inference ${MODEL} with reranker ${RERANK_MODEL}" \
    --evaluate \
    --metric accuracy em
