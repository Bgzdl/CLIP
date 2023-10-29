import torch
import torch.nn as nn
from .LoRA import LoRA_CLIP
from .embedMethod import embedMethod


class VPT_LoRA_CLIP(nn.Module):
    def __init__(self, embed, model_name, prompt_length: int):
        super(VPT_LoRA_CLIP, self).__init__()
        self.pretrained_model = LoRA_CLIP(embed, model_name)
        self.prompt_length = prompt_length
        self.prompt_vector = nn.Parameter(torch.randn(prompt_length, 768))

    def encode_image(self, image):
        image_feature = self.pretrained_model.encode_image(image)
        image_feature = self.prompt_vector + image_feature
        return image_feature

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
