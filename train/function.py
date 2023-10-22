import torch
import numpy as np
from clip.LoRA import embedMethod
import clip
from tqdm import tqdm
from biobert.biobert import bert
import os


# train
def train(model, dataloader, criterion, optimizer, embed, epoch, train_logger):
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
    for i, dictionary in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{30}")):
        optimizer.zero_grad()
        I, labels = dictionary['data'], dictionary['label']
        labels = labels.cuda()
        I = torch.tensor(np.stack(I)).cuda()
        image_features = model.encode_image(I).float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = torch.mm(text_features, image_features.T)
        loss = criterion(similarity, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_logger.info(f"Epoch: {epoch + 1}, batch: {i + 1}, Train Loss: {loss.item():.8f}")

    return running_loss / len(dataloader)


# evaluate
def get_max_indices(matrix):
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a NumPy array")

    max_indices = []  # 存储每一行最大数字的下标

    for row in matrix:
        max_index = np.argmax(row)  # 获取当前行最大数字的下标
        max_indices.append(max_index)

    return np.array(max_indices)


def evaluate(model, dataloader, embed: embedMethod, epoch, predict_logger):
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
    for i, dictionary in enumerate(tqdm(dataloader)):
        I = dictionary['data']
        I = torch.tensor(np.stack(I)).cuda()
        label = dictionary['label']
        image_features = model.encode_image(I).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
        predict = get_max_indices(similarity.T)
        total += len(predict)
        predict, label = np.array(predict), np.array(label)
        predict_logger.info(f"Epoch: {epoch + 1}, batch: {i + 1},  Predict: {predict}")
        comparision = predict == label
        correct += np.sum(comparision)

    return correct / total


# save model
def save_model(model, path, acc):
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = os.path.join(path, 'best model.pth')
    with open(save_path, "wb") as f:
        torch.save(model.state_dict(), f)
    acc_file = os.path.join(path, 'acc.txt')
    with open(acc_file, "w") as f:
        f.write(f"Accuracy is {acc}")
