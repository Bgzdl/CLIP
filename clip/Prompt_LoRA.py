import torch
import torch.nn as nn
from .LoRA import LoRA_CLIP
from .embedMethod import embedMethod


class VPT_LoRA_CLIP(nn.Module):
    def __init__(self, embed: embedMethod, model_name: str, prompt_length: int):
        super(VPT_LoRA_CLIP, self).__init__()
        self.pretrained_model = LoRA_CLIP(embed, model_name)
        self.embed = embed
        self.prompt_length = prompt_length
        self.prompt_vector = nn.Parameter(torch.randn(prompt_length, 768))

    def encode_image(self, image):
        image_feature = self.pretrained_model.encode_image(image)
        image_feature = self.prompt_vector + image_feature
        return image_feature

    def encode_text(self, text):
        text_feature = self.pretrained_model.encode_text(text)
        return text_feature

    def forward(self, image, text):
        return self.pretrained_model(image, text)
