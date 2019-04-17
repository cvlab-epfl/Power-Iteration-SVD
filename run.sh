#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py --norm='batchnorm' --batch_size=32
CUDA_VISIBLE_DEVICES=2 python main.py --norm='groupnorm' --batch_size=32
CUDA_VISIBLE_DEVICES=0 python main.py --norm='instancenorm' --batch_size=32
CUDA_VISIBLE_DEVICES=1 python main.py --norm='layernorm' --batch_size=32

CUDA_VISIBLE_DEVICES=2 python main.py --norm='mygroupnorm' --batch_size=32