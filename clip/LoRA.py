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


class LoRA_CLIP(nn.Module):
    def __init__(self, model: CLIP, embed: embedMethod, model_name):
        super().__init__()
        self.name = model_name
        self.origin_model = model
        for param in self.origin_model.parameters():
            param.requires_grad = False
        if self.name == 'ViT-L/14':
            self.LoRA = LoRA(224 * 224 * 3, 768, 16)
        elif self.name == 'ViT-B/16':
            self.LoRa = LoRA(224*224*3, 512, 16)
        self.embed = embed
        self.Biobert = bert_token_embedding(self.name)

    def encode_image(self, image):
        image_feature = self.origin_model.encode_image(image)
        feature = self.LoRA(image)
        return image_feature + feature

    def encode_text(self, text):
        if self.embed == embedMethod.clip:
            x = self.origin_model.encode_text(text)
            return x
        elif self.embed == embedMethod.bio_bert:
            x = self.Biobert(text)  # [batch_size, n_ctx, d_model]
            x = x + self.origin_model.positional_embedding.type(self.origin_model.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.origin_model.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.origin_model.ln_final(x).type(self.origin_model.dtype)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.origin_model.text_projection
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
