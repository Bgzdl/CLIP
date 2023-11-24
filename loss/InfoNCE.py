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
        self.temperature = temperature

    def forward(self, similarity_matrix, labels):
        """
        Calculate contrastive loss with false negatives attraction.

        Args:
            similarity_matrix: A matrix of shape (N, N) where N is the number of samples.
                               similarity_matrix[i][j] represents sim(zi, zj).
            false_negative_masks: A boolean tensor of shape (N, N) where false_negative_masks[i][j]
                                  is True if zj is a false negative of zi.

        Returns:
            A scalar loss value.
        """
        false_negative_masks = ContrastiveLoss.get_false_negative_mask(labels)
        false_negative_masks = 0.5 * false_negative_masks
        N = similarity_matrix.size(0)
        exp_sim = torch.exp(similarity_matrix / self.temperature)

        # Create a mask to zero-out self-similarities along the diagonal
        diagonal_mask = torch.eye(N, device=similarity_matrix.device).bool()

        # Set diagonal and true negatives to a very small value before applying the mask
        exp_sim.masked_fill_(diagonal_mask, 0)
        exp_sim.masked_fill_(false_negative_masks, 0)

        # Calculate the log probabilities
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # Calculate the loss for each anchor excluding false negatives
        loss_normal = -log_prob.diagonal().mean()

        # Now calculate the loss where false negatives are treated as positives
        fn_exp_sim = torch.exp(similarity_matrix / self.temperature)  # Recalculate without zeroing out false negatives
        fn_log_prob = similarity_matrix - torch.log(fn_exp_sim.sum(dim=1, keepdim=True))

        # Apply the mask for false negatives and calculate the loss
        fn_loss = -torch.sum(fn_log_prob * false_negative_masks.float(), dim=1)

        # Average the loss across all anchors and normalize by the count of false negatives + 1
        fn_count = false_negative_masks.float().sum(dim=1) + 1  # +1 to avoid division by zero
        loss_false_negatives = torch.sum(fn_loss / fn_count) / N

        # Combine the normal loss and the false negative loss
        total_loss = (loss_normal + loss_false_negatives) / 2

        return total_loss

    @staticmethod
    def get_false_negative_mask(labels):
        return torch.tensor(~get_mask(labels) - np.eye(len(labels)))
