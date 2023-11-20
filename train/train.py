import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import sys
current_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_path)
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
import clip
from clip.LoRA import LoRA_CLIP, embedMethod
from clip.Adapter import Adapter_CLIP
from clip.Prompt_LoRA import VPT_LoRA_CLIP
from function import train, evaluate, save_model, evaluate_1
from dataset.dataset import Patch, Patch_Dataset
from loss.InfoNCE import InfoNCE_loss
from parse.parser import parser

# 设备准备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 模型超参数
args = parser.parse_args()
Optimization = args.model
path = args.data_path
epoches = args.epoches
batch_size = args.batch_size
temperature = args.temperature
lr = args.learning_rate
decayRate = args.decayRate

# 结果
best_acc = 0.0
best_epoch = 0

# 模型准备
model_name = 'ViT-L/14'  # ['ViT-B/16', 'ViT-L/14']
_, transform = clip.load(model_name)
print(model_name)
embed = embedMethod.clip
if Optimization == 'Adapter':
    model = Adapter_CLIP(embed, model_name)
elif Optimization == 'LoRA':
    model = LoRA_CLIP(embed, model_name)
elif Optimization == 'Prompt_LoRA':
    model = VPT_LoRA_CLIP(embed, model_name, 1)
else:
    raise Exception("unknown model name ")
print('model is ', Optimization)
model.to(device)
'''
for name, param in model.named_parameters():
    print(name, param.requires_grad)
'''
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# loss
infoNCE_loss = InfoNCE_loss(temperature).cuda()


# 数据集
'''
print('preparing dataset')
dataset = Patch(path, label_type=False, transform=transform, load=False)
count_0, count_1, count_2 = dataset.Count_the_number_of_various_tags()
print('Quantity of various categories is', count_0, count_1, count_2)
train_dataset, val_dataset, test_dataset = dataset.split()
'''
train_dataset = Patch_Dataset(path, 'train', load=False, transform=transform, seed=0)
val_dataset = Patch_Dataset(path, 'val', load=False, transform=transform, seed=0)
group_names, group_labels = val_dataset.get_ground_true()
count_0, count_1, count_2 = train_dataset.Count_the_number_of_various_tags()
print('Quantity of various categories in train is', count_0, count_1, count_2)
print('Number of Group in train dataset is', train_dataset.get_group_length())
count_0, count_1, count_2 = val_dataset.Count_the_number_of_various_tags()
print('Quantity of various categories in val is', count_0, count_1, count_2)
print('Number of Group in val dataset is', val_dataset.get_group_length())

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
print('finish')

# 优化器
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

# 日志文件
save_path = "./log/all"
save_path = os.path.join(save_path, Optimization)
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

# 训练过程
for epoch in range(epoches):
    torch.cuda.empty_cache()
    loss_function = infoNCE_loss
    with torch.autocast("cuda"):
        model.train()
        train_loss = train(model, train_dataloader, loss_function, optimizer, model.embed, epoch, train_logger)
        print(f"Train Epoch {epoch + 1}/{30}, Average Loss: {train_loss:.4f}")
        scheduler.step()
        model.eval()
        with torch.no_grad():
            acc, single_acc = evaluate_1(model, group_names,group_labels, val_dataloader, embed, epoch,  predict_logger)
            print(f"Validation Epoch {epoch + 1}/{epoches}, Accuracy: {acc:.8f}, Single Accuracy:{single_acc:.8f}")
        running_logger.info(f"Epoch: {epoch + 1}, Running Loss: {train_loss:.8f}, acc: {acc:.8f}, Single Accuracy:{single_acc:.8f}")

    # 如果验证正确率高于当前最佳正确率，则保存模型参数
    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
        save_path = os.path.join('./model', 'all')
        save_model(model, save_path, best_acc)
        print(f"Best model saved at epoch {epoch + 1}, acc: {best_acc:.8f}")
