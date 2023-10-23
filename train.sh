#!/bin/bash

epoches='5'
batch_size='32'
lr='0.00001'
temperature='0.01'
decayRate='0.8'
python_script="./train/train.py"

python "$python_script" "$epoches" "$batch_size" "$lr" "$temperature" "$decayRate"