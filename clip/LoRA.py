import torch
import torch.nn as nn
import clip
from .model import CLIP
from .embedMethod import embedMethod
from biobert.biobert import bert_token_embedding


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
    def __init__(self, model: CLIP, embed: embedMethod):
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

        for param in super().parameters():
            param.requires_grad = False
        self.LoRA = LoRA(224 * 224 * 3, 512, 16)
        self.embed = embed
        self.Biobert = bert_token_embedding()

    def encode_image(self, image):
        image_feature = super().encode_image(image)
        feature = self.LoRA(image)
        return image_feature + feature

    def encode_text(self, text):
        if self.embed == embedMethod.clip:
            x = super().encode_text(text)
            return x
        elif self.embed == embedMethod.bio_bert:
            x = self.Biobert(text)  # [batch_size, n_ctx, d_model]
            x = x + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
            return x
        else:
            raise Exception('Embedding Error')
