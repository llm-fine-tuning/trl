#!/bin/bash
# train.sh
# DeepSpeed로 4개의 GPU를 사용해 train.py 실행, 로그는 train.log로 저장

deepspeed --num_gpus=4 train.py > train.log 2>&1
