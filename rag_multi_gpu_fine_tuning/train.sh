#!/bin/bash
# train.sh
# DeepSpeed로 2개의 GPU를 사용해 train.py 실행, 로그는 train.log로 저장
deepspeed --num_gpus=2 rag_train.py