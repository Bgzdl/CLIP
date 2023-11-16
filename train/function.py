import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import clip
from clip.LoRA import embedMethod
from biobert.biobert import bert


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
        group_names = dictionary['group_name']
        I = torch.tensor(np.stack(I)).cuda()
        similarity, max_index = model.predict(I, T)
        similarity = torch.tensor(similarity)
        predict_logger.info(f"Epoch: {epoch + 1}, batch: {i + 1},  Predict: {max_index}")
        for j, row in enumerate(similarity.T):
            normalized_row = F.softmax(row, dim=0)
            if group_names[j] in result.keys():
                result[group_names[j]] += normalized_row
            else:
                result[group_names[j]] = normalized_row
    for key, value in result.items():
        _, index = value.max(0)
        result[key] = index
    return result


def evaluate_1(model, group_names, group_labels, dataloader, embed: embedMethod, epoch, predict_loger):
    result = predict(model, dataloader, embed, epoch, predict_loger)
    correct = 0
    for group_name, group_label in zip(group_names, group_labels):
        correct += result[group_name] == group_label
    acc = correct / len(group_labels)
    return acc


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
