unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=6,7
export PATH=/usr/local/cuda/bin:$PATH
export DS_SKIP_CUDA_CHECK=1
deepspeed  train_stage1.py \
    --use_queue False \
    --train_stage 1 \
    --Train_data_path '.Data/PMC-600K/train.json' \
    --Eval_data_path '.Data/PMC-600K/test.json' \
    --output_dir ./Results \
    --run_name Alignment_training_lora8 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --save_strategy "steps" \
    --save_steps 500 \
    --load_best_model_at_end True \
    --learning_rate 5e-5 \
    --weight_decay 1e-3 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --deepspeed ./ds_config/ds.json  