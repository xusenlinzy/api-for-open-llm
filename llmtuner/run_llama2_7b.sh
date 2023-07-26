output_model=checkpoints/llama2/sft-7b-chat
cp ./run_llama2_7b.sh ${output_model}
deepspeed --include localhost:0 --master_port 29510 train_llama2.py \
    --tokenizer_name ./openbuddy_tokenizer \
    --model_name_or_path ./checkpoints/llama2-7b-chat \
    --train_file ./data/dummy_data.jsonl \
    --per_device_train_batch_size 1 \
    --do_train \
    --use_fast_tokenizer false \
    --output_dir ${output_model} \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 10 \
    --warmup_steps 400 \
    --load_in_bits 4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --preprocessing_num_workers 10 \
    --save_steps 200 \
    --save_total_limit 20 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 4096 \
    --report_to tensorboard \
    --overwrite_output_dir \
    --deepspeed ds_config_zero2.json \
    --ignore_data_skip true \
    --bf16 \
    --gradient_checkpointing \
    --bf16_full_eval \
    --ddp_timeout 18000000 \
    | tee -a ${output_model}/train.log
