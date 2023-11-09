import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, t):
        super(CrossEntropyLoss, self).__init__()
        self.t = t

    def forward(self, similarity, labels):
        criterion = nn.CrossEntropyLoss()
        similarity = similarity * np.exp(self.t)
        similarity = similarity + 1e-6
        loss = criterion(similarity.T, labels)
        return loss


def get_probability_matrix(labels: np.array):
    length = len(labels)
    probability = np.zeros((length, length))
    for i in range(length):
        n = np.count_nonzero(labels == labels[i]) - 1
        for j in range(length):
            if i == j:
                probability[i, j] = 0.7
            else:
                if labels[j] == labels[i]:
                    probability[i, j] = 0.3 / n
                else:
                    probability[i, j] = 0
    return torch.tensor(probability)


def get_weight_matrix(labels: np.array):
    length = len(labels)
    weight = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            if i == j:
                weight[i, j] = 1
            else:
                if labels[i] == labels[j]:
                    pass
                else:
                    weight[i, j] = 1
    return torch.tensor(weight)


class Probability_Loss(nn.Module):
    def __init__(self, t):
        super(Probability_Loss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.t = t

    def forward(self, logits_per_image, labels):
        logits_per_image = logits_per_image * np.exp(self.t)
        m = nn.LogSoftmax(dim=1)
        logits_per_image = m(logits_per_image)
        probability = get_probability_matrix(labels).to(logits_per_image.device)
        loss_i = self.criterion(logits_per_image, probability)
        return loss_i


class InfoNCE_Loss(nn.Module):
    def __init__(self, t):
        super(InfoNCE_Loss, self).__init__()
        self.t = t

    def forward(self, logits_per_image, labels):
        logits_per_image = logits_per_image * np.exp(self.t)
        weight = get_weight_matrix(labels)
        loss_i = F.cross_entropy(logits_per_image, labels, weight=weight)
        return loss_i
