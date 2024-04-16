#!/usr/bin/env bash

set -eu

# python train.py  --gpu "0" --batch-size 32 --log-interval 50
python test.py --gpu "7"
