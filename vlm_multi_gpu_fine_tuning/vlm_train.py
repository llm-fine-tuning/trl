# pip install torch==2.4.0 transformers==4.45.1 datasets==3.0.1 accelerate==0.34.2 trl==0.11.1 peft==0.13.0 deepspeed qwen-vl-utils

import os
import torch
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoModelForVision2Seq, AutoProcessor  # Qwen2-VL 모델용 (다른 멀티모달 모델 사용 시 모델 ID와 프로세서 변경)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from qwen_vl_utils import process_vision_info

# 만약 max_seq_length가 전역 변수라면 정의 (예시: 1024)
max_seq_length = 1024

# 1. 허깅페이스 허브에서 데이터셋 로드 및 이미지가 있는 데이터만 필터링
dataset = load_dataset("daje/Ko-SciecneQA", split="train")
dataset = dataset.filter(lambda example: example["image"] is not None)

# 2. 텍스트 입력 템플릿 및 시스템 프롬프트 정의
prompt = """질문: {korean_question}
선택지: {korean_choices}
힌트: {korean_hint}"""

system_message = "주어진 이미지와 질문을 바탕으로 답변하세요.\n이때 정답은 선택지 중 1개를 선택해야하며 힌트가 주어질 수 있습니다. 가장 적절한 답을 1개 선택하세요."

# 3. 데이터셋을 OpenAI 메시지 형식으로 변환하는 함수
def format_data(sample):
    return {
        "messages": [
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
                    },
                    {
                        "type": "image",
                        "image": sample["image"] if sample["image"] is not None else "",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample["answer_str"]}
                ],
            },
        ]
    }

# 4. 데이터셋 전체를 OpenAI 메시지 형식으로 변환
dataset = [format_data(sample) for sample in dataset]
print('데이터의 개수:', len(dataset))

# 5. train/test 분할 (예시: 90% train, 10% test)
train_dataset_list = dataset[:int(len(dataset) * 0.9)]
test_dataset_list = dataset[int(len(dataset) * 0.9):]
print('학습 데이터의 개수:', len(train_dataset_list))
print('테스트 데이터의 개수:', len(test_dataset_list))

# 6. 로컬에 저장된 데이터셋이 있으면 불러오고, 없으면 새로 변환 후 저장
if os.path.exists("train_dataset"):
    train_dataset = load_from_disk("train_dataset")
    print("로컬 train_dataset 로드 완료.")
else:
    train_dataset = Dataset.from_list(train_dataset_list)
    train_dataset.save_to_disk("train_dataset")
    print("train_dataset을 로컬에 저장함.")

if os.path.exists("test_dataset"):
    test_dataset = load_from_disk("test_dataset")
    print("로컬 test_dataset 로드 완료.")
else:
    test_dataset = Dataset.from_list(test_dataset_list)
    test_dataset.save_to_disk("test_dataset")
    print("test_dataset을 로컬에 저장함.")

# 7. 모델과 프로세서 로드 (모델 ID를 원하는 멀티모달 모델로 변경 가능)
model_id = "Qwen/Qwen2-VL-7B-Instruct"
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    # device_map="auto",  # DeepSpeed/Accelerate가 분산 배치를 직접 관리하므로 주석 처리
    torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained(model_id)

# 8. 단일 예시 확인 (예: 첫 번째 데이터의 템플릿 적용 결과)
example_text = processor.apply_chat_template(train_dataset[0]["messages"], tokenize=False)
print("템플릿 적용 결과:")
print(example_text)

# 9. PEFT LoRA 설정
peft_config = LoraConfig(
    lora_alpha=128,
    lora_dropout=0.05,
    r=256,
    bias="none",
    target_modules=[
        "q_proj",    # Query 투영 레이어
        "up_proj",   # FFN 상향 투영 레이어
        "o_proj",    # Output 투영 레이어
        "k_proj",    # Key 투영 레이어
        "down_proj", # FFN 하향 투영 레이어
        "gate_proj", # FFN 게이트 투영 레이어
        "v_proj"     # Value 투영 레이어
    ],
    task_type="CAUSAL_LM",
)

# 10. SFTConfig 학습 설정 (DeepSpeed 설정 파일 지정)
args = SFTConfig(
    output_dir="output_dir",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=5,
    save_strategy="epoch",
    learning_rate=1e-4,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns=False,
    deepspeed="ds_config.json"  # DeepSpeed 설정 파일 지정
)

# 11. 데이터 collator 함수 (텍스트와 이미지 쌍을 인코딩)
def collate_fn(examples):
    # processor를 이용하여 채팅 템플릿 적용
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    image_inputs = [process_vision_info(example["messages"])[0] for example in examples]

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    # labels는 input_ids 복사본으로, 패딩 토큰은 -100으로 변경
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # 이미지 토큰 인덱스 무시 (모델별로 조정 필요)
    if hasattr(processor, "image_token"):
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    else:
        # Qwen2VL 모델의 예시 이미지 토큰 ID (예시 값, 모델마다 다를 수 있음)
        image_tokens = [151652, 151653, 151655]
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100

    batch["labels"] = labels
    return batch

# 12. SFTTrainer 생성 (모든 인자는 기존 코드 그대로)
trainer = SFTTrainer(
    model=model,
    args=args,
    max_seq_length=max_seq_length,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    peft_config=peft_config,
    tokenizer=processor.tokenizer,
)

# 13. 학습 시작 및 모델 저장
def main():
    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
