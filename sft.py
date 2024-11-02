from datasets import load_dataset
from transformers import AutoTokenizer
from trl import (
    ModelConfig,          # 모델 설정을 관리하는 클래스
    ScriptArguments,      # 스크립트 실행 인자를 관리하는 클래스
    SFTConfig,           # Supervised Fine-Tuning 설정 클래스
    SFTTrainer,          # Supervised Fine-Tuning 트레이너
    TrlParser,           # 커맨드라인 인자 파서
    get_kbit_device_map, # 8비트/4비트 양자화를 위한 디바이스 매핑
    get_peft_config,     # PEFT(Parameter-Efficient Fine-Tuning) 설정
    get_quantization_config, # 모델 양자화 설정
)

if __name__ == "__main__":
    # 1. 커맨드라인 인자 파싱 및 설정
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    ################
    # 2. 모델 초기화 설정 및 토크나이저
    ################
    # 양자화 설정 가져오기 (8비트/4비트 등)
    quantization_config = get_quantization_config(model_config)
    
    # 모델 로드를 위한 기본 설정
    model_kwargs = dict(
        revision=model_config.model_revision,              # 모델 리비전(버전)
        trust_remote_code=model_config.trust_remote_code, # 원격 코드 신뢰 여부
        attn_implementation=model_config.attn_implementation,  # 어텐션 구현 방식
        torch_dtype=model_config.torch_dtype,             # 텐서 데이터 타입
        # 그래디언트 체크포인팅 사용 시 캐시 비활성화
        use_cache=False if training_args.gradient_checkpointing else True,
        # 양자화 사용 시 디바이스 매핑 설정
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,          # 양자화 설정
    )
    
    # 학습 인자에 모델 초기화 설정 추가
    training_args.model_init_kwargs = model_kwargs
    
    # 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True  # 빠른 토크나이저 사용
    )
    # pad_token이 없는 경우 eos_token을 pad_token으로 사용
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # 3. 데이터셋 로드
    ################
    dataset = load_dataset(script_args.dataset_name)

    ################
    # 4. 학습 설정 및 실행
    ################
    trainer = SFTTrainer(
        # 모델은 경로만 지정 (SFTTrainer가 내부적으로 로드)
        model=model_config.model_name_or_path,
        args=training_args,              # 학습 관련 설정
        train_dataset=dataset[script_args.dataset_train_split],  # 학습 데이터
        # 평가 데이터 (eval_strategy가 'no'가 아닐 때만)
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,      # 토크나이저
        peft_config=get_peft_config(model_config),  # PEFT 설정 (LoRA 등)
    )
    
    # 학습 실행
    trainer.train()
    
    # 모델 저장
    trainer.save_model(training_args.output_dir)
    
    # Hugging Face Hub에 모델 업로드 (설정된 경우)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)