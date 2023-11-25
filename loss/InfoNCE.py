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
        self.softmax = nn.Softmax(dim=1)
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
    def __init__(self, temperature=1.0):
        super(ContrastiveLoss, self).__init__()
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
        false_negative_masks = ContrastiveLoss.get_false_negative_mask(labels).bool().to(similarity_matrix.device)
        N = similarity_matrix.size(0)
        temp_exp = torch.exp(self.temperature)
        modified_sim = similarity_matrix * temp_exp

        # Create a mask to zero-out self-similarities along the diagonal
        diagonal_mask = torch.eye(N, device=similarity_matrix.device).bool()

        # Calculate the softmax denominator for normal and false negatives
        exp_sim = torch.exp(modified_sim)
        exp_sim.masked_fill_(diagonal_mask, 0)

        softmax_denominator = exp_sim.sum(dim=1, keepdim=True)

        # Calculate the log probabilities for normal samples
        log_probs_normal = modified_sim - torch.log(softmax_denominator)

        # Calculate the log probabilities for false negatives
        fn_softmax_denominator = exp_sim.clone()
        fn_softmax_denominator.masked_fill_(false_negative_masks, 0)
        fn_softmax_denominator = fn_softmax_denominator.sum(dim=1, keepdim=True)
        log_probs_fn = modified_sim - torch.log(fn_softmax_denominator)

        # Calculate the loss for normal and false negatives
        loss_normal = -log_probs_normal.diagonal().mean()
        fn_loss = -torch.sum(log_probs_fn * false_negative_masks.float(), dim=1)
        fn_count = false_negative_masks.float().sum(dim=1) + 1
        loss_false_negatives = torch.sum(fn_loss / fn_count) / N

        # Combine the losses
        total_loss = (loss_normal + loss_false_negatives) / 2

        return total_loss

    @staticmethod
    def get_false_negative_mask(labels):
        mask = get_mask(labels).float()  # 将布尔张量转换为浮点张量
        identity_matrix = torch.eye(len(labels), device=mask.device)  # 创建单位矩阵
        false_negative_mask = identity_matrix - mask  # 执行减法操作
        return false_negative_mask
