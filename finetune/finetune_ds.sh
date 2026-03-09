#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2,3
GPUS_PER_NODE=3
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6012

MODEL="/root/Documents/code_2/FM9G4B-V/model"
DATA="/root/Documents/code_2/FM9G4B-V/data/train/fully_merged_training_data_2.json"
EVAL_DATA="/root/Documents/code_2/FM9G4B-V/finetune/data.json"

MODEL_MAX_Length=2048 # if conduct multi-images sft, please set MODEL_MAX_Length=4096

#RESUME_CHECKPOINT="/root/Documents/code_2/FM9G4B-V/output/hra_merge_data/checkpoint-8000"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

    #--use_lora \
    # --use_boft\
    #--use_llama_adapter\
    #--use_bone \
torchrun $DISTRIBUTED_ARGS finetune.py  \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --eval_data_path $EVAL_DATA \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 true \
    --bf16_full_eval true \
    --fp16 false \
    --fp16_full_eval false \
    --do_train \
    --do_eval \
    --use_hra \
    --tune_vision true \
    --tune_llm false \
    --model_max_length $MODEL_MAX_Length \
    --max_slice_nums 9 \
    --num_train_epochs 1 \
    --eval_steps 8000 \
    --output_dir /root/Documents/code_2/FM9G4B-V/output/hra_merge_data_2 \
    --logging_dir /root/Documents/code_2/FM9G4B-V/output/hra_merge_data_2\
    --logging_strategy "steps" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed ds_config_zero2.json \
    --report_to "tensorboard" \
   # --resume_from_checkpoint $RESUME_CHECKPOINT \
