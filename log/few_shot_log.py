import logging
import os
from train.few_shot_train import shot_num

save_path = f"../log/few_shot/{shot_num}"
if not os.path.exists(save_path):
    os.makedirs(save_path)
train_logger = logging.getLogger('train')
train_logger.setLevel(logging.INFO)
train_path = os.path.join(save_path, 'train_loss.txt')
train_logger.addHandler(logging.FileHandler(train_path, mode='w'))  # 将日志输出到txt文件

predict_logger = logging.getLogger('predict')
predict_logger.setLevel(logging.INFO)
predict_path = os.path.join(save_path, 'predict.txt')
predict_logger.addHandler(logging.FileHandler(predict_path, mode='w'))  # 将日志输出到txt文件

running_logger = logging.getLogger('running')
running_logger.setLevel(logging.INFO)
running_path = os.path.join(save_path, 'result.txt')
running_logger.addHandler(logging.FileHandler(running_path, mode='w'))  # 将日志输出到txt文件
