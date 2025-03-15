# pip install torch==2.4.0 transformers==4.45.1 datasets==3.0.1 accelerate==0.34.2 trl==0.11.1 peft==0.13.0 deepspeed
# train.py

import os
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, LoraConfig

# 만약 max_seq_length가 전역 변수라면 정의
max_seq_length = 1024  # 환경에 맞게 수정

# 1. 허깅페이스 허브에서 데이터셋 로드
dataset = load_dataset("iamjoon/klue-mrc-ko-rag-dataset", split="train")

# 2. system_message 정의
system_message = """당신은 검색 결과를 바탕으로 질문에 답변해야 합니다.

다음의 지시사항을 따르십시오.
1. 질문과 검색 결과를 바탕으로 답변하십시오.
2. 검색 결과에 없는 내용을 답변하려고 하지 마십시오.
3. 질문에 대한 답이 검색 결과에 없다면 검색 결과에는 "해당 질문~에 대한 내용이 없습니다." 라고 답변하십시오.
4. 답변할 때 특정 문서를 참고하여 문장 또는 문단을 작성했다면 뒤에 출처는 이중 리스트로 해당 문서 번호를 남기십시오. 예를 들어서 특정 문장이나 문단을 1번 문서에서 인용했다면 뒤에 [[ref1]]이라고 기재하십시오.
5. 예를 들어서 특정 문장이나 문단을 1번 문서와 5번 문서에서 동시에 인용했다면 뒤에 [[ref1]], [[ref5]]이라고 기재하십시오.
6. 최대한 다수의 문서를 인용하여 답변하십시오.

검색 결과:
-----
{search_result}"""

# 3. 원본 데이터의 type별 분포 출력 
print("원본 데이터의 type 분포:")
for type_name in set(dataset['type']):
    print(f"{type_name}: {dataset['type'].count(type_name)}")

# 4. train/test 분할 비율 설정 (0.5면 5:5로 분할)
test_ratio = 0.8

train_data = []
test_data = []

# 5. type별로 순회하면서 train/test 데이터 분할
for type_name in set(dataset['type']):
    curr_type_data = [i for i in range(len(dataset)) if dataset[i]['type'] == type_name]
    test_size = int(len(curr_type_data) * test_ratio)
    test_data.extend(curr_type_data[:test_size])
    train_data.extend(curr_type_data[test_size:])

# 6. OpenAI format으로 데이터 변환을 위한 함수 
def format_data(sample):
    search_result = "\n-----\n".join([f"문서{idx + 1}: {result}" for idx, result in enumerate(sample["search_result"])])
    return {
        "messages": [
            {
                "role": "system",
                "content": system_message.format(search_result=search_result),
            },
            {
                "role": "user",
                "content": sample["question"],
            },
            {
                "role": "assistant",
                "content": sample["answer"]
            },
        ],
    }

# 7. 분할된 데이터를 OpenAI format으로 변환
train_dataset = [format_data(dataset[i]) for i in train_data]
test_dataset = [format_data(dataset[i]) for i in test_data]

print(f"\n전체 데이터 분할 결과: Train {len(train_dataset)}개, Test {len(test_dataset)}개")

print("\n학습 데이터의 type 분포:")
for type_name in set(dataset['type']):
    count = sum(1 for i in train_data if dataset[i]['type'] == type_name)
    print(f"{type_name}: {count}")

print("\n테스트 데이터의 type 분포:")
for type_name in set(dataset['type']):
    count = sum(1 for i in test_data if dataset[i]['type'] == type_name)
    print(f"{type_name}: {count}")

# 리스트 형태에서 다시 Dataset 객체로 변경
print("변경 전 train_dataset 타입:", type(train_dataset))
print("변경 전 test_dataset 타입:", type(test_dataset))
train_dataset = Dataset.from_list(train_dataset)
test_dataset = Dataset.from_list(test_dataset)
print("변경 후 train_dataset 타입:", type(train_dataset))
print("변경 후 test_dataset 타입:", type(test_dataset))

# 허깅페이스 모델 ID
model_id = "Qwen/Qwen2-7B-Instruct" 

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # device_map="auto", DeepSpeed로 멀티 GPU 학습을 할 때는 Accelerate나 DeepSpeed가 직접 모델의 분산 배치를 관리하도록 해야하므로 해당 코드를 주석 처리할 것.
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 템플릿 적용 예시
text = tokenizer.apply_chat_template(
    train_dataset[0]["messages"], tokenize=False, add_generation_prompt=False
)
print("템플릿 적용 결과:", text)

peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

args = SFTConfig(
    output_dir="qwen2-7b-rag-ko",           # 저장될 디렉토리 및 저장소 ID
    num_train_epochs=3,                      # 에포크 수
    per_device_train_batch_size=2,           # GPU 당 배치 크기
    gradient_accumulation_steps=2,           # 그래디언트 누적 스텝
    gradient_checkpointing=True,             # 메모리 절약 체크포인팅
    optim="adamw_torch_fused",               # 최적화기
    logging_steps=10,                        # 로그 주기
    save_strategy="steps",                   # 저장 전략
    save_steps=50,                           # 저장 주기
    bf16=True,                               # bfloat16 사용
    learning_rate=1e-4,                      # 학습률
    max_grad_norm=0.3,                       # 그래디언트 클리핑
    warmup_ratio=0.03,                       # 워밍업 비율
    lr_scheduler_type="constant",            # 학습률 스케줄러 타입
    push_to_hub=False,                       # 허브 업로드 여부
    remove_unused_columns=False,
    dataset_kwargs={"skip_prepare_dataset": True},
    report_to=None,
    deepspeed="ds_config.json"               # DeepSpeed 설정 파일 지정
)

def collate_fn(batch):
    new_batch = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }
    
    for example in batch:
        clean_messages = []
        for message in example["messages"]:
            clean_messages.append({
                "role": message["role"],
                "content": message["content"]
            })
        text = tokenizer.apply_chat_template(
            clean_messages,
            tokenize=False,
            add_generation_prompt=False
        ).strip()
        
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None,
        )
        
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        labels = [-100] * len(input_ids)
        
        im_start = "<|im_start|>"
        im_end = "<|im_end|>"
        assistant = "assistant"
        
        im_start_tokens = tokenizer.encode(im_start, add_special_tokens=False)
        im_end_tokens = tokenizer.encode(im_end, add_special_tokens=False)
        assistant_tokens = tokenizer.encode(assistant, add_special_tokens=False)
        
        i = 0
        while i < len(input_ids):
            if (i + len(im_start_tokens) <= len(input_ids) and 
                input_ids[i:i+len(im_start_tokens)] == im_start_tokens):
                assistant_pos = i + len(im_start_tokens)
                if (assistant_pos + len(assistant_tokens) <= len(input_ids) and 
                    input_ids[assistant_pos:assistant_pos+len(assistant_tokens)] == assistant_tokens):
                    current_pos = assistant_pos + len(assistant_tokens)
                    while current_pos < len(input_ids):
                        if (current_pos + len(im_end_tokens) <= len(input_ids) and 
                            input_ids[current_pos:current_pos+len(im_end_tokens)] == im_end_tokens):
                            for j in range(len(im_end_tokens)):
                                labels[current_pos + j] = input_ids[current_pos + j]
                            break
                        labels[current_pos] = input_ids[current_pos]
                        current_pos += 1
                    i = current_pos
            i += 1
        
        new_batch["input_ids"].append(input_ids)
        new_batch["attention_mask"].append(attention_mask)
        new_batch["labels"].append(labels)
    
    max_length = max(len(ids) for ids in new_batch["input_ids"])
    for i in range(len(new_batch["input_ids"])):
        padding_length = max_length - len(new_batch["input_ids"][i])
        new_batch["input_ids"][i].extend([tokenizer.pad_token_id] * padding_length)
        new_batch["attention_mask"][i].extend([0] * padding_length)
        new_batch["labels"][i].extend([-100] * padding_length)
    
    for k, v in new_batch.items():
        new_batch[k] = torch.tensor(v)
    
    return new_batch

def main():
    trainer = SFTTrainer(
        model=model,
        args=args,
        max_seq_length=max_seq_length,  # 최대 시퀀스 길이 설정
        train_dataset=train_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
    )
    
    # 학습 시작 (모델은 자동으로 허브 및 output_dir에 저장됨)
    trainer.train()
    # 모델 저장 (최종 모델)
    trainer.save_model()

if __name__ == "__main__":
    main()
