import numpy as np
import torch
import torch.nn as nn


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


class InfoNCE_Loss(nn.Module):
    def __init__(self, t):
        super(InfoNCE_Loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.t = t

    def forward(self, logits_per_image):
        logits_per_image = logits_per_image * np.exp(self.t)
        labels = torch.arange(logits_per_image.shape[0]).to(logits_per_image.device)
        loss_i = self.criterion(logits_per_image, labels)
        return loss_i
