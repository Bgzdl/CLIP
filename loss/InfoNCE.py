import numpy as np
import torch.nn as nn


class InfoNCE_loss(nn.Module):
    def __init__(self, t):
        super(InfoNCE_loss, self).__init__()
        self.t = t

    def forward(self, similarity, labels):
        criterion = nn.CrossEntropyLoss()
        similarity = similarity * np.exp(self.t)
        similarity = similarity + 1e-6
        loss = criterion(similarity.T, labels)
        return loss
