'''
# 단일 GPU 실행
python sft_vlm_qwen2_vl.py \
    --model_name_or_path "Qwen/Qwen2-VL-7B-Instruct" \
    --dataset_test_ratio 0.2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --save_checkpoint_steps 50 \
    --output_dir "output_dir" \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --target_modules "q_proj,v_proj,k_proj,o_proj"

# 다중 GPU 실행 (4 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file=deepspeed_zero3.yaml \
    sft_vlm_qwen2_vl.py \
    --model_name_or_path "Qwen/Qwen2-VL-7B-Instruct" \
    --dataset_test_ratio 0.2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --save_checkpoint_steps 50 \
    --output_dir "output_dir" \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --target_modules "q_proj,v_proj,k_proj,o_proj"
'''

# PyTorch와 관련된 모듈을 임포트합니다.
import torch
import logging
from typing import Dict, List, Optional, Union

# 데이터셋을 로드하는 데 사용되는 Hugging Face datasets 모듈을 임포트합니다.
from datasets import load_dataset, Dataset

# 모델과 프로세서를 로드하는 데 사용되는 Hugging Face transformers 모듈을 임포트합니다.
from transformers import (
   AutoModelForVision2Seq, 
   AutoProcessor, 
   LlavaForConditionalGeneration,
   Qwen2VLProcessor
)

# Transformer Reinforcement Learning(TRL) 관련 모듈을 임포트합니다.
from trl import (
   ModelConfig,               # 모델 설정을 관리하는 클래스
   ScriptArguments,           # 스크립트 실행 인자를 관리하는 클래스
   SFTConfig,                 # Supervised Fine-Tuning 설정을 관리하는 클래스
   SFTTrainer,                # Supervised Fine-Tuning을 수행하는 트레이너
   TrlParser,                 # TRL 관련 인자를 처리하는 파서
   get_kbit_device_map,       # 8비트/4비트 양자화를 위한 디바이스 매핑
   get_peft_config,           # PEFT(Parameter-Efficient Fine-Tuning) 설정
   get_quantization_config,   # 모델 양자화 설정
)

# 이미지 처리 유틸리티를 임포트합니다.
from qwen_vl_utils import process_vision_info

if __name__ == "__main__":
   # 1. 커맨드라인 인자 파싱 및 설정
   parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))

   # 추가: 테스트 비율 파라미터 정의
   parser.add_argument("--dataset_test_ratio", type=float, default=0.2, help="Test dataset ratio (train ratio is 1 - test ratio)")
   parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
   parser.add_argument("--save_checkpoint_steps", type=int, default=500, help="Number of steps between saving checkpoints")
   parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training")
   
   # 추가: LoRA 관련 파라미터
   parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha value")
   parser.add_argument("--lora_r", type=int, default=8, help="LoRA r value")
   parser.add_argument("--target_modules", type=str, default="q_proj,v_proj,k_proj,o_proj", help="Target modules for LoRA tuning, separated by commas")

   # 추가: 기타 설정 관련 파라미터
   parser.add_argument("--skip_prepare_dataset", type=bool, default=True, help="Skip dataset preparation step")
   parser.add_argument("--remove_unused_columns", type=bool, default=False, help="Remove unused columns in the dataset")
   parser.add_argument("--evaluation_strategy", type=str, default="steps", help="The evaluation strategy to adopt during training")
   parser.add_argument("--eval_steps", type=int, default=500, help="Number of update steps between two evaluations")

   # 인자 파싱
   script_args, training_args, model_config = parser.parse_args_and_config()
   
   # 그래디언트 체크포인팅 설정: 메모리 효율을 위해 중간 활성화 값을 저장하지 않음
   training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
   # 데이터셋에서 사용하지 않는 컬럼 유지 (전처리를 위해 필요할 수 있음)
   training_args.remove_unused_columns = False
   # 데이터셋 준비 과정 스킵 (사용자 정의 전처리를 사용할 경우)
   training_args.dataset_kwargs = {"skip_prepare_dataset": True}

   # 테스트 비율 유효성 검증
   if not 0 < script_args.dataset_test_ratio < 1:
       raise ValueError("Test ratio must be a float between 0 and 1.")

   # 학습 비율 자동 계산
   dataset_train_ratio = 1.0 - script_args.dataset_test_ratio

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
   # 3. 데이터셋 로드 및 분리
   ################
   prompt = """질문: {korean_question}
   선택지: {korean_choices}
   힌트: {korean_hint}"""

   system_message = "주어진 이미지와 질문을 바탕으로 답변하세요.\n이때 정답은 선택지 중 1개를 선택해야하며 힌트가 주어질 수 있습니다. 가장 적절한 답을 1개 선택하세요."

   def format_data(sample):
       return {"messages": [
                   {
                       "role": "system", 
                       "content": [{"type": "text", "text": system_message}], 
                   },
                   {
                       "role": "user",  
                       "content": [
                           {
                               "type": "text",
                               "text": prompt.format(
                                   korean_question=sample["korean_question"], 
                                   korean_choices=sample["korean_choices"], 
                                   korean_hint=sample["korean_hint"]
                               ),
                           },{
                               "type": "image", 
                               "image": sample["image"] if sample["image"] is not None else "", 
                           }
                       ],
                   },
                   {
                       "role": "assistant", 
                       "content": [
                           {
                               "type": "text", 
                               "text": sample["answer_str"]
                           }
                       ], 
                   },
               ],
           }

   try:
       dataset = load_dataset("daje/Ko-ScienceQA", split="train")
       dataset = dataset.filter(lambda example: example["image"] is not None)
       dataset = dataset.map(format_data)

       train_size = int(len(dataset) * dataset_train_ratio)
       test_size = len(dataset) - train_size
       train_dataset, test_dataset = dataset.train_test_split(
           train_size=train_size, test_size=test_size, shuffle=True
       ).values()
   except Exception as e:
       logger.error(f"데이터셋 로드 중 오류 발생: {e}")
       raise

   ################
   # 4. 데이터 콜레이터 정의
   ################
   def collate_fn(examples):
       # 각 예제에서 텍스트와 이미지를 추출하고, 텍스트는 채팅 템플릿을 적용
       texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
       image_inputs = [process_vision_info(example["messages"])[0] for example in examples]

       # 텍스트를 토크나이징하고 이미지를 처리하여 일괄 처리(batch) 형태로 변환
       batch = processor(
           text=texts, 
           images=image_inputs, 
           return_tensors="pt",  # PyTorch 텐서로 반환
           padding=True          # 배치 내 시퀀스 길이 통일
       )

       # labels로 사용할 input_ids 복사본 생성 후, 패딩 토큰을 -100으로 설정하여 손실 계산 시 무시하도록 함
       labels = batch["input_ids"].clone()
       labels[labels == processor.tokenizer.pad_token_id] = -100  # 패딩 토큰 손실 계산 제외

       # 특정 이미지 토큰 인덱스는 손실 계산에서 무시 (모델에 따라 다름)
       if isinstance(processor, Qwen2VLProcessor):  
           # Qwen2VL 모델의 이미지 토큰 인덱스
           image_tokens = [151652, 151653, 151655]
       else:
           # 다른 모델에서 이미지 토큰 ID를 얻어 손실 계산에서 제외
           image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
       
       # 손실 계산 시 이미지 토큰 인덱스를 무시하도록 설정
       for image_token_id in image_tokens:
           labels[labels == image_token_id] = -100
       
       # 배치에 labels 추가 (손실 계산 시 사용)
       batch["labels"] = labels

       return batch

   ################
   # 5. 학습 설정 및 실행
   ################
   trainer = SFTTrainer(
       model=model,                           # 학습할 모델
       args=training_args,                    # 학습 관련 설정
       data_collator=collate_fn,              # 데이터 전처리 함수
       train_dataset=train_dataset,           # 학습 데이터셋
       eval_dataset=test_dataset if training_args.evaluation_strategy != "no" else None,  # 평가 데이터셋
       tokenizer=processor.tokenizer,         # 텍스트 처리용 토크나이저
       peft_config=get_peft_config(
           lora_alpha=script_args.lora_alpha,      # LoRA 알파 값
           lora_r=script_args.lora_r,             # LoRA r 값
           target_modules=script_args.target_modules.split(","),  # 타겟 모듈
       ),
       num_train_epochs=script_args.num_train_epochs,   # 에포크 수
       save_steps=script_args.save_checkpoint_steps,    # 체크포인트 저장 주기
       learning_rate=script_args.learning_rate,         # 러닝 레이트
       evaluation_strategy=script_args.evaluation_strategy,  # 평가 전략
       eval_steps=script_args.eval_steps,              # 평가 스텝 간격
   )

   # 학습 시작
   trainer.train()
   
   # 모델 저장
   trainer.save_model(training_args.output_dir)

   # Hugging Face Hub에 모델 업로드 (선택적)
   if training_args.push_to_hub:
       trainer.push_to_hub(dataset_name=script_args.dataset_name)
       if trainer.accelerator.is_main_process:
           processor.push_to_hub(training_args.hub_model_id)

   # GPU 메모리 정리
   torch.cuda.empty_cache()