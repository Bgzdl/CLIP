#!/bin/bash

model='LoRA'
path='./data' # '/root/autodl-tmp/patch' in autodl
epoches='30'
batch_size='32'
lr='0.0001'
temperature='0.01'
decayRate='0.8'
python_script="./train/train.py"

python "$python_script" "$model" "$path" "$epoches" "$batch_size" "$lr" "$temperature" "$decayRate"

shutdown -h now