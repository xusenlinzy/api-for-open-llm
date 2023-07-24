output_model=checkpoints/llama2/sft_7b_chat
if [ ! -d ${output_model} ];then
    mkdir ${output_model}
fi
cp ./finetune.sh ${output_model}
deepspeed --include localhost:0 --master_port 29510 train.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --train_file ./data/train_sft.csv \
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
