'''
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
'''

import torch
from datasets import load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration
from trl import (
    ModelConfig,          # 모델 설정을 관리하는 클래스
    ScriptArguments,      # 스크립트 실행 인자를 관리하는 클래스
    SFTConfig,           # Supervised Fine-Tuning 설정을 관리하는 클래스
    SFTTrainer,          # Supervised Fine-Tuning을 수행하는 트레이너
    TrlParser,           # TRL(Transformer Reinforcement Learning) 관련 인자 파서
    get_kbit_device_map, # 8비트/4비트 양자화를 위한 디바이스 매핑
    get_peft_config,     # PEFT(Parameter-Efficient Fine-Tuning) 설정
    get_quantization_config, # 모델 양자화 설정
)

if __name__ == "__main__":
    # 1. 커맨드라인 인자 파싱 및 설정
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    
    # 그래디언트 체크포인팅 설정: 메모리 효율을 위해 중간 활성화 값을 저장하지 않음
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    # 데이터셋에서 사용하지 않는 컬럼 유지 (전처리를 위해 필요할 수 있음)
    training_args.remove_unused_columns = False
    # 데이터셋 준비 과정 스킵 (사용자 정의 전처리를 사용할 경우)
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    ################
    # 2. 모델, 토크나이저, 프로세서 초기화
    ################
    # torch_dtype 설정 (float16, bfloat16 등)
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    
    # 양자화 설정 가져오기 (8비트/4비트 등)
    quantization_config = get_quantization_config(model_config)
    
    # 모델 로드를 위한 기본 설정
    model_kwargs = dict(
        revision=model_config.model_revision,              # 모델 리비전(버전)
        attn_implementation=model_config.attn_implementation,  # 어텐션 구현 방식
        torch_dtype=torch_dtype,                          # 텐서 데이터 타입
        device_map=get_kbit_device_map() if quantization_config is not None else None,  # 디바이스 매핑
        quantization_config=quantization_config,          # 양자화 설정
    )

    # 이미지-텍스트 처리를 위한 프로세서 로드
    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path, 
        trust_remote_code=model_config.trust_remote_code
    )
    
    # Vision2Seq 모델 로드
    model = AutoModelForVision2Seq.from_pretrained(
        model_config.model_name_or_path, 
        trust_remote_code=model_config.trust_remote_code, 
        **model_kwargs
    )

    ################
    # 3. 데이터 콜레이터(전처리기) 정의
    ################
    def collate_fn(examples):
        # 텍스트와 이미지를 챗 형식으로 변환
        texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        images = [example["images"] for example in examples]
        
        # LLaVA 1.5 모델은 단일 이미지만 지원
        if isinstance(model, LlavaForConditionalGeneration):
            images = [image[0] for image in images]
            
        # 텍스트 토큰화 및 이미지 처리
        batch = processor(
            text=texts, 
            images=images, 
            return_tensors="pt",  # PyTorch 텐서로 반환
            padding=True          # 배치 내 시퀀스 길이 통일
        )
        
        # 레이블 생성 (입력 ID를 복사)
        labels = batch["input_ids"].clone()
        # 패딩 토큰은 손실 계산에서 제외 (-100)
        labels[labels == processor.tokenizer.pad_token_id] = -100
        
        # 이미지 토큰도 손실 계산에서 제외
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        labels[labels == image_token_id] = -100
        
        batch["labels"] = labels
        return batch

    ################
    # 4. 데이터셋 로드
    ################
    dataset = load_dataset(script_args.dataset_name)

    ################
    # 5. 학습 설정 및 실행
    ################
    trainer = SFTTrainer(
        model=model,                     # 학습할 모델
        args=training_args,              # 학습 관련 설정
        data_collator=collate_fn,        # 데이터 전처리 함수
        train_dataset=dataset[script_args.dataset_train_split],  # 학습 데이터
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,  # 평가 데이터
        processing_class=processor.tokenizer,  # 토크나이저
        peft_config=get_peft_config(model_config),  # PEFT 설정 (LoRA 등)
    )
    
    # 학습 실행
    trainer.train()
    
    # 모델 저장
    trainer.save_model(training_args.output_dir)
    
    # Hugging Face Hub에 모델 업로드 (설정된 경우)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        # 메인 프로세스에서만 프로세서 업로드
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)