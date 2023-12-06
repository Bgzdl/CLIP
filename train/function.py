import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import clip
from clip.LoRA import embedMethod
from biobert.biobert import bert
from loss.InfoNCE import ContrastiveLoss


# train
def train(model, dataloader, criterion, optimizer, embed, epoch, train_logger):
    running_loss = 0.0
    for i, dictionary in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{30}")):
        optimizer.zero_grad()
        I, T, label = dictionary['data'], dictionary['target'], dictionary['label']
        I = torch.tensor(np.stack(I)).cuda()
        if embed == embedMethod.clip:
            T = clip.tokenize([desc for desc in T]).cuda()
        elif embed == embedMethod.bio_bert:
            T = bert.tokenize([desc for desc in T]).cuda()
        else:
            raise Exception("Val Token Error")
        logits_per_image, _ = model(I, T)
        if isinstance(criterion, ContrastiveLoss):
            T_mask, F_mask = get_mask(label, dictionary['target'])
            loss = criterion(logits_per_image, T_mask, F_mask)
        else:
            loss = criterion(logits_per_image, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_logger.info(f"Epoch: {epoch + 1}, batch: {i + 1}, Train Loss: {loss.item():.8f}")

    return running_loss / len(dataloader)


def evaluate(model, dataloader, cirterion, embed: embedMethod, epoch, predict_logger):
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
    for i, dictionary in enumerate(tqdm(dataloader)):
        I = dictionary['data']
        I = torch.tensor(np.stack(I)).cuda()
        label = dictionary['label']
        similarity, predict = model.predict(I, T)
        total += len(predict)
        predict, label = np.array(predict), np.array(label)
        predict_logger.info(f"Epoch: {epoch + 1}, batch: {i + 1},  Predict: {predict}")
        comparision = predict == label
        correct += np.sum(comparision)

    return correct / total


def predict(model, dataloader, embed: embedMethod, epoch, predict_logger):
    result = dict()
    total = 0
    correct = 0
    T = ['Well differentiated tubular adenocarcinoma',
         'Moderately differentiated tubular adenocarcinoma',
         'Poorly differentiated adenocarcinoma']
    if embed == embedMethod.clip:
        T = clip.tokenize([desc for desc in T]).cuda()
    elif embed == embedMethod.bio_bert:
        T = bert.tokenize([desc for desc in T]).cuda()
    else:
        raise Exception("Val Token Error")
    for i, dictionary in enumerate(tqdm(dataloader)):
        I, labels = dictionary['data'], dictionary['label']
        group_names = dictionary['group_name']
        I = torch.tensor(np.stack(I)).cuda()
        similarity, max_index = model.predict(I, T)
        similarity = torch.tensor(similarity)
        predict_logger.info(f"Epoch: {epoch + 1}, batch: {i + 1},  Predict: {max_index}")
        total += len(max_index)
        max_index, labels = np.array(max_index), np.array(labels)
        comparision = max_index == labels
        correct += np.sum(comparision)
        for j, row in enumerate(similarity.T):
            normalized_row = F.softmax(row, dim=0)
            if group_names[j] in result.keys():
                result[group_names[j]] += normalized_row
            else:
                result[group_names[j]] = normalized_row
    for key, value in result.items():
        _, index = value.max(0)
        result[key] = index
    return result, correct / total


def evaluate_1(model, group_names, group_labels, dataloader, embed: embedMethod, epoch, predict_loger):
    result, single_acc = predict(model, dataloader, embed, epoch, predict_loger)
    correct = 0
    for group_name, group_label in zip(group_names, group_labels):
        correct += result[group_name] == group_label
    acc = correct / len(group_labels)
    return acc, single_acc


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


# get True false negative mask
def get_T_mask(T: list):
    text_to_int = {}
    current_int = 0
    N = len(T)
    # 生成映射
    for text in T:
        if text not in text_to_int:
            text_to_int[text] = current_int
            current_int += 1
    int_array = torch.tensor([text_to_int[text] for text in T])
    T_mask = torch.zeros((N, N)).bool()
    for i in range(N):
        for j in range(i + 1, N):
            if int_array[i] == int_array[j]:
                T_mask[i, j] = True
                T_mask[j, i] = True

    return T_mask


# get True and Fake false negative mask
def get_mask(label: list, T: list):
    length = len(label)
    # negative mask
    mask = np.zeros((length, length), dtype=bool)
    for i in range(length):
        for j in range(length):
            if i == j:
                mask[i, j] = 0
            else:
                if label[i] == label[j]:
                    pass
                else:
                    mask[i, j] = 1
    mask = torch.from_numpy(mask)
    mask = ~mask
    T_mask = get_T_mask(T)
    F_mask = (mask.float() - T_mask.float() - torch.eye(length)).bool()
    return T_mask, F_mask
