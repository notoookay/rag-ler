export CUDA_VISIBLE_DEVICES=0,1
# export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true
# add this if mlflow raise the error:
#   mlflow.exceptions.MlflowException: The configured tracking uri scheme: 'file' is invalid for u se with the proxy mlflow-artifact scheme. The allowed tracking schemes are: {'http', 'https'}
# export MLFLOW_TRACKING_URI=http://localhost:5000

MODEL=llama2-7b
BASE_MODEL=meta-llama/Llama-2-7b-hf
DATASET=llm_train.jsonl
TRAIN_METHOD=qlora
NUM_GPUS=2
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
EXPERIMENT_NAME="Instruction-tuning LM"
echo "Training $BASE_MODEL using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# Lora training
accelerate launch \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    finetune.py \
    --model_name_or_path ${BASE_MODEL} \
    --use_qlora \
    --use_lora \
    --use_flash_attn \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --use_slow_tokenizer \
    --train_file data/${DATASET} \
    --max_seq_length 2048 \
    --gradient_checkpointing \
    --preprocessing_num_workers 20 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --output_dir output/model/${MODEL}_${DATASET}_${TRAIN_METHOD} \
    --with_tracking \
    --report_to mlflow \
    --experiment_name "${EXPERIMENT_NAME}" \
    --run_name "instruction-tuning ${MODEL} on ${DATASET} data with ${TRAIN_METHOD}" \
    --logging_steps 1 &&

python merge_lora.py \
    --base_model_name_or_path ${BASE_MODEL} \
    --lora_model_name_or_path output/model/${MODEL}_${DATASET}_${TRAIN_METHOD}/ \
    --output_dir output/model/${MODEL}_${DATASET}_${TRAIN_METHOD}_merged/ \
    --qlora \
    --save_tokenizer
