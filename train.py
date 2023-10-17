import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import clip
from clip.LoRA import LoRA_CLIP, embedMethod
# tokenizer
from biobert.biobert import bert
from clip.Adapter import Adapter_CLIP
from dataset.dataset import Patch
from torch.utils.data import DataLoader
from tqdm import tqdm


class InfoNCE_loss(nn.Module):
    def __init__(self, t):
        super(InfoNCE_loss, self).__init__()
        self.t = t

    def forward(self, similarity, labels):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(similarity, labels)
        return loss


# train
def train(model, dataloader, criterion, optimizer, embed, epoch):
    running_loss = 0.0
    T = ['Well differentiated tubular adenocarcinoma',
         'Moderately differentiated tubular adenocarcinoma',
         'Poorly differentiated adenocarcinoma']
    if embed == embedMethod.clip:
        T = clip.tokenize([desc for desc in T]).cuda()
    elif embed == embedMethod.bio_bert:
        T = bert.tokenize([desc for desc in T]).cuda()
    else:
        raise Exception("Val Token Error")
    text_features = model.encode_text(T).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)
    for dictionary in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{30}"):
        optimizer.zero_grad()
        I, labels = dictionary['data'], dictionary['label']
        I = torch.tensor(np.stack(I)).cuda()
        image_features = model.encode_image(I).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
        loss = criterion(similarity, labels)
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
def evaluate(model, dataloader, embed: embedMethod):
    correct = 0
    total = 0
    T = ['Well differentiated tubular adenocarcinoma',
         'Moderately differentiated tubular adenocarcinoma',
         'Poorly differentiated adenocarcinoma']
    if embed == embedMethod.clip:
        T = clip.tokenize([desc for desc in T]).cuda()
    elif embed == embedMethod.bio_bert:
        T = bert.tokenize([desc for desc in T]).cuda()
    else:
        raise Exception("Val Token Error")
    text_features = model.encode_text(T).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)
    for dictionary in tqdm(dataloader):
        I = dictionary['data']
        I = torch.tensor(np.stack(I)).cuda()
        label = dictionary['label']
        image_features = model.encode_image(I).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
        predict = get_max_indices(similarity.T)
        total += len(predict)
        predict, label = np.array(predict), np.array(label)
        comparision = predict == label
        correct += np.sum(comparision)

    return correct / total


# 模型准备
model_name = 'ViT-L/14'  # ['ViT-B/16', 'ViT-L/14']
_, transform = clip.load(model_name)
print(model_name)
Optimization = 'Adapter'  # model_name = ['Adapter', 'LoRA']
embed = embedMethod.bio_bert
if Optimization == 'Adapter':
    model = Adapter_CLIP(embed, model_name)
elif Optimization == 'LoRA':
    model = LoRA_CLIP(embed, model_name)
else:
    raise Exception("unknown model name ")
print('model is ', Optimization)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# 超参设置
temperature = 0.01
infonce_loss = InfoNCE_loss(temperature)
infonce_loss = infonce_loss.cuda()

# 数据集
print('preparing dataset')
dataset = Patch('data', True, transform, load=False)  # '/root/autodl-tmp/patch' in autodl
count_0, count_1, count_2 = dataset.Count_the_number_of_various_tags()
print('Quantity of various categories is', count_0, count_1, count_2)
train_dataset, val_dataset, test_dataset = dataset.split()

train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=True, num_workers=8)
print('finish')

# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.0001)
decayRate = 0.96
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
epoches = 30

for epoch in range(epoches):
    torch.cuda.empty_cache()
    with torch.autocast("cuda"):
        model.train()
        train_loss = train(model, train_dataloader, infonce_loss, optimizer, model.embed, epoch)
        print(f"Train Epoch {epoch + 1}/{30}, Average Loss: {train_loss:.4f}")
        scheduler.step()
        model.eval()
        with torch.no_grad():
            acc = evaluate(model, val_dataloader, model.embed)
            print(f"Validation Epoch {epoch + 1}/{30}, Accuracy: {acc:.4f}")
