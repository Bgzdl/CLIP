import os.path
import torch
import torch.optim as optim
import torch.nn as nn
from ..parser.few_shot_parser import parser
import clip
from clip.LoRA import LoRA_CLIP, embedMethod
from clip.Adapter import Adapter_CLIP
from loss.InfoNCE import InfoNCE_loss
from dataset.few_shot_dataset import Few_shot_train, Few_shot_val
from torch.utils.data import DataLoader
from function import train, evaluate, save_model

# 设备准备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 模型超参数
args = parser.parse_args()
epoches = args.epoches
batch_size = args.batch_size
temperature = args.temperature
lr = args.learning_rate
decayRate = args.decayRate

# few shot num
shot_num = args.shot_num

# 结果
best_acc = 0.0
best_epoch = 0

# 模型准备
model_name = 'ViT-L/14'  # ['ViT-B/16', 'ViT-L/14']
_, transform = clip.load(model_name)
print(model_name)
Optimization = 'LoRA'  # model_name = ['Adapter', 'LoRA']
embed = embedMethod.clip
if Optimization == 'Adapter':
    model = Adapter_CLIP(embed, model_name)
elif Optimization == 'LoRA':
    model = LoRA_CLIP(embed, model_name)
else:
    raise Exception("unknown model name ")
print('model is ', Optimization)
model.to(device)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# loss
infonce_loss = InfoNCE_loss(temperature).cuda()

# 数据集
print('preparing dataset')
train_dataset = Few_shot_train('../data', transform, load=True, shot_num=shot_num)
val_dataset = Few_shot_val('../data', transform, load=False, shot_num=shot_num)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
print('finish')

# 优化器
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

# 日志
from ..log.few_shot_log import train_logger, predict_logger, running_logger


for epoch in range(epoches):
    torch.cuda.empty_cache()
    with torch.autocast("cuda"):
        model.train()
        train_loss = train(model, train_dataloader, infonce_loss, optimizer, model.embed, epoch, train_logger)
        print(f"Train Epoch {epoch + 1}/{epoches}, Average Loss: {train_loss:.4f}")
        scheduler.step()
        model.eval()
        with torch.no_grad():
            acc = evaluate(model, val_dataloader, model.embed, epoch, predict_logger)
            print(f"Validation Epoch {epoch + 1}/{epoches}, Accuracy: {acc:.8f}")
        running_logger.info(f"Epoch: {epoch + 1}, Running Loss: {train_loss:.8f}, acc: {acc:.8f}")

    # 如果验证正确率高于当前最佳正确率，则保存模型参数
    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
        save_path = os.path.join('model', 'few_shot', str(shot_num))
        save_model(model, save_path, best_acc)
        print(f"Best model saved at epoch {epoch + 1}, acc: {best_acc:.8f}")
