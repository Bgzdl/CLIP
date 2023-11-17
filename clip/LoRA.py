import numpy as np
import torch
import torch.nn as nn
import clip
from .embedMethod import embedMethod
import math
from biobert.biobert import bert_token_embedding

def get_max_indices(matrix):
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a NumPy array")

    max_indices = []  # 存储每一行最大数字的下标

    for row in matrix:
        max_index = np.argmax(row)  # 获取当前行最大数字的下标
        max_indices.append(max_index)

    return np.array(max_indices)

class LoRALayer:
    def __init__(
            self,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class LoRA(nn.Module, LoRALayer):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 8,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = True,
    ):
        nn.Module.__init__(self)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        self.in_features = in_features
        self.out_features = out_features
        self.scaling = self.lora_alpha / self.r
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)
        return result


class LoraResidualAttentionBlock(nn.Module):
    def __init__(self, origin_model: nn.Module, d_model: int):
        super().__init__()
        self.origin_model = origin_model
        for param in self.origin_model.parameters():
            param.requires_grad = False
        self.LoRA = LoRA(d_model, d_model, 8)

    def forward(self, x: torch.Tensor):
        lora_features = self.LoRA(x)
        x = x + self.origin_model.attention(self.origin_model.ln_1(x))
        x = x + self.origin_model.mlp(self.origin_model.ln_2(x))
        x = lora_features + x
        return x


class LoRA_CLIP(nn.Module):
    def __init__(self, embed: embedMethod, model_name):
        super().__init__()
        self.name = model_name
        self.origin_model, _ = clip.load(model_name)
        for param in self.origin_model.parameters():
            param.requires_grad = False
        new_model = []
        for block in self.origin_model.visual.transformer.resblocks:
            new_model.append(LoraResidualAttentionBlock(block, block.d_model))
        new_model = nn.Sequential(*new_model)
        self.origin_model.visual.transformer.resblocks = new_model
        self.embed = embed
        self.project = nn.Linear(512, 768)
        self.Biobert = bert_token_embedding(self.name)
        for param in self.Biobert.parameters():
            param.requires_grad = False

    def encode_image(self, image):
        image_feature = self.origin_model.visual(image.type(self.origin_model.dtype))
        if self.name == 'ViT-B/16':
            image_feature = self.project(image_feature)
        return image_feature

    def encode_text(self, text):
        if self.embed == embedMethod.clip:
            x = self.origin_model.encode_text(text)
            return x
        elif self.embed == embedMethod.bio_bert:
            x = self.Biobert(text)  # [batch_size, n_ctx, d_model]
            return x
        else:
            raise Exception('Embedding Error')

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.origin_model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
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
