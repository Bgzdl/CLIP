import torch
import numpy as np
from tqdm import tqdm
import os
import clip
from clip.LoRA import embedMethod
from biobert.biobert import bert


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
