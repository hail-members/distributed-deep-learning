#!/bin/bash
CUDA_VISIBLE_DEVICES=1,2
deepspeed --num_gpus=2 cifar10_deepspeed.py --epochs=2 --deepspeed_config bf16.json
deepspeed --num_gpus=2 cifar10_deepspeed.py --epochs=2 --deepspeed_config fp32.json