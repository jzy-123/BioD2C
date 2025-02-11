unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1
export PATH=/usr/local/cuda/bin:$PATH
export DS_SKIP_CUDA_CHECK=1
deepspeed train.py \
    --train_stage 1 \
    --use_queue True \
    --ckp path to checkpoints \
    --Train_data_path './Data/BioVGQ/train.jsonl' \
    --Eval_data_path './Data/BioVGQ/val.jsonl' \
    --output_dir ./Results\
    --run_name vqa \
    --num_train_epochs 5 \
    --lora_rank 8 \
    --per_device_train_batch_size 8\
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 1500 \
    --save_strategy "steps" \
    --save_steps 1500 \
    --load_best_model_at_end True \
    --learning_rate 2e-5 \
    --weight_decay 1e-3 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --deepspeed ./ds_config/ds.json  
