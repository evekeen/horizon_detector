#!/bin/bash

# Train the improved horizon detector model with optimal parameters
python train.py \
  --model light \
  --scheduler warmup_cosine \
  --lr 0.0005 \
  --epochs 30 \
  --early-stopping 5 \
  --batch-size 64