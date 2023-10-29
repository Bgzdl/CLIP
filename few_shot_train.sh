#!/bin/bash

model='LoRA'
epoches='5'
batch_size='32'
lr='0.0001'
temperature='0.01'
decayRate='0.8'
shot_nums=('1' '2' '4' '8' '16')

python_script="./train/few_shot_train.py"

# 循环遍历参数数组
for shot_num in "${shot_nums[@]}"; do
    # 调用Python脚本并传入参数
    python "$python_script" "$model" "$epoches" "$batch_size" "$lr" "$temperature" "$decayRate" "$shot_num"
done

shutdown -h now