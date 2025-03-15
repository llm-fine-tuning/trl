import argparse

from datasets import load_dataset
from transformers import AutoTokenizer

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

# 메인 함수 정의
def main(script_args, training_args, model_args):
    ################
    # Model init kwargs & Tokenizer
    ################
    # 모델 초기화에 필요한 설정 및 키워드 인자 생성
    quantization_config = get_quantization_config(model_args)  # 모델 양자화 설정 가져오기
    model_kwargs = dict(
        revision=model_args.model_revision,  # 모델 리비전 설정
        trust_remote_code=model_args.trust_remote_code,  # 신뢰할 수 있는 코드 사용 여부
        attn_implementation=model_args.attn_implementation,  # 어텐션 구현 방식
        torch_dtype=model_args.torch_dtype,  # PyTorch 데이터 타입 설정
        use_cache=False if training_args.gradient_checkpointing else True,  # 그래디언트 체크포인팅 사용 여부에 따른 캐시 설정
        device_map=get_kbit_device_map() if quantization_config is not None else None,  # 디바이스 매핑 설정
        quantization_config=quantization_config,  # 양자화 설정
    )
    training_args.model_init_kwargs = model_kwargs  # 학습 인자에 모델 초기화 키워드 인자 추가

    # 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,  # 모델 경로 지정
        trust_remote_code=model_args.trust_remote_code,  # 신뢰할 수 있는 코드 사용 여부
        use_fast=True  # 빠른 토크나이저 사용
    )
    tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰을 종료 토큰으로 설정

    ################
    # Dataset
    ################
    # 데이터셋 로드
    dataset = load_dataset(
        script_args.dataset_name,  # 데이터셋 이름
        name=script_args.dataset_config  # 데이터셋 설정
    )

    ################
    # Training
    ################
    # SFTTrainer 초기화 및 학습 수행
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,  # 모델 경로 지정
        args=training_args,  # 학습 관련 설정
        train_dataset=dataset[script_args.dataset_train_split],  # 학습 데이터셋 지정
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,  # 평가 데이터셋 지정 (평가 전략에 따라 조건부로 지정)
        processing_class=tokenizer,  # 데이터 전처리에 사용할 토크나이저
        peft_config=get_peft_config(model_args),  # LoRA 설정 로드
    )

    trainer.train()  # 학습 시작

    # 모델 저장 및 Hugging Face Hub에 업로드
    trainer.save_model(training_args.output_dir)  # 학습된 모델 저장
    if training_args.push_to_hub:  # Hub에 업로드 여부 확인
        trainer.push_to_hub(dataset_name=script_args.dataset_name)  # 데이터셋 이름과 함께 업로드

# 명령줄 인자 파서를 생성하는 함수
def make_parser(subparsers: argparse._SubParsersAction = None):
    # 필요한 데이터 클래스 타입 정의
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig)
    if subparsers is not None:  # 서브 파서가 제공된 경우
        parser = subparsers.add_parser(
            "sft",  # SFT 서브 명령 추가
            help="Run the SFT training script",  # 도움말 메시지
            dataclass_types=dataclass_types  # 데이터 클래스 타입 연결
        )
    else:  # 서브 파서가 없는 경우
        parser = TrlParser(dataclass_types)  # 기본 TrlParser 생성
    return parser  # 생성된 파서 반환

# 스크립트의 메인 실행 부분
if __name__ == "__main__":
    parser = make_parser()  # 파서 생성
    script_args, training_args, model_args = parser.parse_args_and_config()  # 명령줄 인자와 설정 파싱
    main(script_args, training_args, model_args)  # 메인 함수 호출