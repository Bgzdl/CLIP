import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import clip
from .embedMethod import embedMethod
from biobert.biobert import bert_token_embedding


def get_max_indices(matrix):
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a NumPy array")

    max_indices = []  # 存储每一行最大数字的下标

    for row in matrix:
        max_index = np.argmax(row)  # 获取当前行最大数字的下标
        max_indices.append(max_index)

    return np.array(max_indices)


class Resnet_CLIP(nn.Module):
    def __init__(self, embed: embedMethod, model_name):
        super().__init__()
        self.embed = embed
        self.origin_model, _ = clip.load(model_name)
        self.name = model_name
        for param in self.origin_model.parameters():
            param.requires_grad = False
        self.visual = models.resnet152(pretrained=True)
        if self.name == 'ViT-B/16':
            self.visual.fc = torch.nn.Linear(self.visual.fc.in_features, 512)
        elif self.name == 'ViT-L/14':
            self.visual.fc = torch.nn.Linear(self.visual.fc.in_features, 768)
        self.Biobert = bert_token_embedding(self.name)
        for param in self.Biobert.parameters():
            param.requires_grad = False

    def encode_image(self, image):
        image_feature = self.visual(image)
        return image_feature

    def encode_text(self, text):
        if self.embed == embedMethod.clip:
            x = self.origin_model.encode_text(text)
            return x
        elif self.embed == embedMethod.bio_bert:
            x = self.Biobert(text)  # [batch_size, n_ctx, d_model]
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
            return x
        else:
            raise Exception('Embedding Error')

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logits_per_image = image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

    @torch.no_grad()
    def predict(self, image, text):
        image_features = self.encode_image(image).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features = self.encode_text(text).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
        predict = get_max_indices(similarity.T)
        return similarity, predict
