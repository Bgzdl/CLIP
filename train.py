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
        # scaled pairwise cosine similarities [n, n]
        logits = torch.mm(predict, target.T) * np.exp(self.t)

        # symmetric loss function
        labels = torch.tensor(np.arange(len(predict))).to('cuda')
        criterion = nn.CrossEntropyLoss()
        loss_i = criterion(logits, labels)
        loss_t = criterion(logits.T, labels)
        loss = (loss_i + loss_t) / 2
        return loss


# train
def train(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for dictionary in train_dataloader:
        optimizer.zero_grad()
        I, T = dictionary['data'], dictionary['target']
        I = torch.tensor(np.stack(I)).cuda()
        T = clip.tokenize([desc for desc in T]).cuda()
        I_f = model.encode_image(I)
        T_f = model.encode_text(T)
        loss = criterion(I_f, T_f)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


# acc calculate
def get_max_indices(matrix):
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a NumPy array")

    max_indices = []  # 存储每一行最大数字的下标

    for row in matrix:
        max_index = np.argmax(row)  # 获取当前行最大数字的下标
        max_indices.append(max_index)

    return np.array(max_indices)


# evaluate
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for dictionary in dataloader:
            I = dictionary['data']
            I = torch.tensor(np.stack(I)).cuda()
            label = dictionary['label']
            T = ['Well differentiated tubular adenocarcinoma',
                 'Moderately differentiated tubular adenocarcinoma',
                 'Poorly differentiated adenocarcinoma']
            T = clip.tokenize([desc for desc in T]).cuda()
            image_features = model.encode_image(I).float()
            text_features = model.encode_text(T).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
            predict = get_max_indices(similarity.T)
            total += len(predict)
            predict, label = np.array(predict), np.array(label)
            comparision = predict == label
            correct += np.sum(comparision)
    return correct / total


# 模型准备
model, transform = clip.load('ViT-B/16')
print(transform)
model = Adapter_CLIP(model)
model.to('cuda')
temperature = 0.01
infonce_loss = InfoNCE_loss(model.visual.proj, model.text_projection, temperature)
infonce_loss = infonce_loss.cuda()
print('temperature is ', temperature)
# 数据集
print('preparing dataset')
dataset = Patch('data', True, transform, True) # '/root/autodl-tmp/patch' in autodl
count_0, count_1, count_2 = dataset.Count_the_number_of_various_tags()
print('Quantity of various categories is', count_0, count_1, count_2)
train_dataset, val_dataset, test_dataset = dataset.split()
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True)
print('finish')
# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.0001)
epoches = 30

for epoch in range(epoches):
    torch.cuda.empty_cache()
    print(epoch)
    train_loss = train(model, train_dataloader, infonce_loss, optimizer)
    print('train loss is ', train_loss)
    acc = evaluate(model, val_dataloader)
    print('acc is ', acc)
