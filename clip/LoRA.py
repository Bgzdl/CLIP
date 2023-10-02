import torch.nn as nn
from .model import CLIP


class LoRA(nn.Module):
    def __init__(self, input_dim, output_dim, r):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.r = r
        self.A = nn.Linear(r, output_dim)
        self.B = nn.Linear(input_dim, r)
        nn.init.normal_(self.A.weight, mean=0, std=0.01)
        nn.init.constant_(self.A.bias, val=0)
        nn.init.zeros_(self.B.weight)
        nn.init.zeros_(self.B.bias)

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        return self.A(self.B(x))


class LoRA_CLIP(CLIP):
    def __init__(self, model: CLIP):
        super().__init__(model.embed_dim,
                         # visual
                         model.image_resolution,
                         model.vision_layers,
                         model.vision_width,
                         model.vision_patch_size,
                         # text
                         model.context_length,
                         model.vocab_size,
                         model.transformer_width,
                         model.transformer_heads,
                         model.transformer_layers,
                         )
        self.origin_model = model
        for param in model.parameters():
            param.requires_grad = False
        self.LoRA = LoRA(224 * 224 * 3, 512, 16)

    def encode_image(self, image):
        image_feature = self.origin_model.encode_image(image)
        feature = self.LoRA(image)
        return image_feature + feature
