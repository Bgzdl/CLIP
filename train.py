import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import clip
from clip.model import Adapter_CLIP
from dataset.dataset import Patch
from torchvision import transforms
from torch.utils.data import DataLoader


class InfoNCE_loss(nn.Module):
    def __init__(self, W_i, W_t, t):
        super(InfoNCE_loss, self).__init__()
        self.W_i = W_i
        self.W_t = W_t
        self.t = t

    def forward(self, predict, target):
        # joint multimodal embedding [n, d_e]
        predict_projection = torch.tensor(torch.dot(predict, self.W_i))
        predict_e = predict_projection / torch.norm(predict_projection, p=2, dim=1, keepdim=True)
        target_projection = torch.tensor(torch.dot(target, self.W_t))
        target_e = target_projection / torch.norm(target_projection, p=2, dim=1, keepdim=True)

        # scaled pairwise cosine similarities [n, n]
        logits = np.dot(predict_e, target_e.T) * np.exp(self.t)

        # symmetric loss function
        labels = np.arange(len(predict))
        criterion = nn.CrossEntropyLoss()
        loss_i = criterion(logits, labels, axis=0)
        loss_t = criterion(logits, labels, axis=1)
        loss = (loss_i + loss_t) / 2
        return loss


# 模型准备
model, transform = clip.load('ViT-B/16')
print(transform)
# model = Adapter_CLIP(model)
model.to('cuda')
infonce_loss = InfoNCE_loss(model.visual.proj, model.text_projection, 0.5)
infonce_loss = infonce_loss.cuda()
# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 数据集
print('preparing dataset')
dataset = Patch('data', False, transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
print('finish')
epoches = 30
for epoch in range(epoches):
    print(epoch)
    running_loss = 0.0
    for dictionary in dataloader:
        I, T = dictionary['data'], dictionary['target']
        I = torch.tensor(np.stack(I)).cuda()
        T = clip.tokenize([desc for desc in T]).cuda()
        I_f = model.encode_image(I)
        T_f = model.encode_text(T)
        optimizer.zero_grad()
        loss = infonce_loss(I_f, T_f)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(epoch)
