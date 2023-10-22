import logging
# 日志文件
train_logger = logging.getLogger('train')
train_logger.setLevel(logging.INFO)
train_logger.addHandler(logging.FileHandler('./all/train_loss.txt', mode='w'))  # 将日志输出到txt文件

predict_logger = logging.getLogger('predict')
predict_logger.setLevel(logging.INFO)
predict_logger.addHandler(logging.FileHandler('./all/predict.txt', mode='w'))  # 将日志输出到txt文件

running_logger = logging.getLogger('running')
running_logger.setLevel(logging.INFO)
running_logger.addHandler(logging.FileHandler('./all/result.txt', mode='w'))  # 将日志输出到txt文件
