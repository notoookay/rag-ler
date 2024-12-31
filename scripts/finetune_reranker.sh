export CUDA_VISIBLE_DEVICES=0,1

BASE_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
DATASET=reranker_train_labels_llama2_7b.jsonl
NUM_GPUS=2
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=64
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
EXPERIMENT_NAME="Fine-tuning Re-ranker"
echo "Training Re-ranker $BASE_MODEL using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory, 
# but it will trade off speed.
accelerate launch \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    finetune_reranker.py \
    --model_name_or_path ${BASE_MODEL} \
    --tokenizer_name ${BASE_MODEL} \
    --use_slow_tokenizer \
    --train_file ${DATASET} \
    --max_seq_length 512 \
    --ctxs_num 30 \
    --d_conf_threshold 0.8 \
    --preprocessing_num_workers 20 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --output_dir output/model \
    --with_tracking \
    --report_to wandb \
    --experiment_name "${EXPERIMENT_NAME}" \
    --logging_steps 1 \
    --checkpointing_steps epoch \