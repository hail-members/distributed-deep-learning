#!/bin/bash

deepspeed --num_gpus=2 train.py --deepspeed_config=ds_config.json -p 2 --steps=200