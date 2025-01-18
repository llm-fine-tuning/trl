CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 \
   sft_vlm_qwen2_vl.py \
   --model_name_or_path "Qwen/Qwen2-VL-7B-Instruct" \
   --dataset_test_ratio 0.2 \
   --per_device_train_batch_size 1 \
   --gradient_accumulation_steps 8 \
   --num_train_epochs 3 \
   --learning_rate 1e-4 \
   --evaluation_strategy "steps" \
   --eval_steps 50 \
   --save_checkpoint_steps 50 \
   --output_dir "output_dir" \
   --bf16 \
   --torch_dtype bfloat16 \
   --gradient_checkpointing \
   --lora_r 16 \
   --lora_alpha 32 \
   --lora_target_modules q_proj k_proj v_proj o_proj \
   --deepspeed deepspeed_zero3.json