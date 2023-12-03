import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax


class CrossEntropyLoss(nn.Module):
    def __init__(self, t, weight: bool = False):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.weight = weight
        self.t = t

    def forward(self, similarity, labels):
        similarity = similarity * np.exp(self.t)
        if torch.isnan(similarity).any():
            raise Exception("similarity has nan!!!")
        if self.weight:
            mask = get_weight(similarity.clone().detach().cpu(), 0.5)
            mask = torch.tensor(mask).to(similarity.device)
            similarity = mask * similarity
        labels = labels.to(similarity.device)
        loss = self.criterion(similarity.T, labels)
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


def get_mask(labels: np.array):
    length = len(labels)
    mask = np.zeros((length, length), dtype=bool)
    for i in range(length):
        for j in range(length):
            if i == j:
                mask[i, j] = 0
            else:
                if labels[i] == labels[j]:
                    pass
                else:
                    mask[i, j] = 1
    return torch.from_numpy(mask)


def get_weight(similarity_matrix, delta):
    # 假定n是样本的数量，similarity_matrix是一个n*n的相似度矩阵
    # 其中对角线元素similarity_matrix[i,i]表示第i个样本与自身的相似度，也就是正样本的相似度
    n = similarity_matrix.shape[0]  # 示例，实际中n应该是相似度矩阵的维度
    epsilon = 1e-9  # 防止除以0

    # 创建mask矩阵，初始化为I
    mask = np.eye(n)

    if np.isnan(similarity_matrix).any():
        raise Exception("similarity_matrix has nan!!!")
    # 对每一行，除了对角线上的正样本，计算其他负样本的权重
    for i in range(n):
        # 提取第i个样本的所有负样本相似度

        negative_similarities = np.concatenate((similarity_matrix[i, :i], similarity_matrix[i, i + 1:]))
        # 计算权重
        if np.isnan(negative_similarities).any():
            raise Exception("negative_similarities has nan!!!")
        weights = 1 / (negative_similarities + epsilon)
        # 应用缩放因子
        weights *= delta
        # 归一化权重
        weights /= np.sum(weights)
        # 将权重分配到mask矩阵的相应位置
        mask[i, :i] = weights[:i]
        mask[i, i + 1:] = weights[i:]

    return mask  # 返回最终的mask矩阵


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


class maskedInfoNCE_Loss(nn.Module):
    def __init__(self, t):
        super(maskedInfoNCE_Loss, self).__init__()
        self.t = t

    def forward(self, logits_per_image, labels):
        logits_per_image = logits_per_image * np.exp(self.t)
        logits_per_image = torch.softmax(logits_per_image, dim=1)
        mask = get_mask(labels).to(logits_per_image.device)
        diagonal_loss = torch.log(logits_per_image.diagonal())
        masked_logits = logits_per_image.masked_fill(~mask, 0)
        non_diagonal_loss = torch.log(1 - masked_logits)
        loss = -diagonal_loss.sum() - (mask * non_diagonal_loss).sum()
        N = logits_per_image.size(0)
        loss = loss / N
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1.0, weight=1.0):
        super(ContrastiveLoss, self).__init__()
        self.weight = weight
        self.temperature = torch.tensor(temperature)

    def forward(self, similarity_matrix, labels):
        """
        Calculate custom contrastive loss with false negatives attraction.

        Args:
            similarity_matrix: A matrix of shape (N, N) where N is the number of samples.
                               similarity_matrix[i][j] represents sim(zi, zj).
            false_negative_masks: A boolean tensor of shape (N, N) where false_negative_masks[i][j]
                                  is True if zj is a false negative of zi.

        Returns:
            A scalar loss value.
        """
        false_negative_masks = ContrastiveLoss.get_false_negative_mask(labels).to(similarity_matrix.device)
        N = similarity_matrix.size(0)
        temp_exp = np.exp(self.temperature)
        f = nn.LogSoftmax(dim=1)
        modified_sim = f(similarity_matrix * temp_exp)

        diagonal_loss = torch.sum(modified_sim * torch.eye(modified_sim.shape))

        fn_loss = torch.sum(modified_sim * false_negative_masks)

        total_loss = (diagonal_loss + self.weight * fn_loss)/N

        return total_loss

    @staticmethod
    def get_false_negative_mask(labels):
        mask = get_mask(labels)
        false_negative_mask = ~mask-np.eye(len(mask))
        return torch.tensor(false_negative_mask).bool()
