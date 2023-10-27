import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from .embedMethod import embedMethod
from .model import CLIP, ResidualAttentionBlock, LayerNorm
from biobert.biobert import bert_token_embedding


class FeedforwardAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super(FeedforwardAdapter, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # First linear transformation
        net = self.fc1(x)
        net = F.gelu(net)  # Apply GELU activation
        # Second linear transformation
        net = self.fc2(net)
        # Residual connection
        net = x + net
        return net


class AdapterResidualAttentionBlock(nn.Module):
    """
    Parallel Adapter
    """
    def __init__(self, origin_model: nn.Module):
        super().__init__()
        self.pretrained_model = origin_model
        d_model = origin_model.d_model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        self.adapter_layer = FeedforwardAdapter(d_model)

    def forward(self, x: torch.Tensor):
        x = x + self.pretrained_model.attention(self.pretrained_model.ln_1(x))
        x = self.pretrained_model.ln_2(x)
        adapter_x = self.adapter_layer(x)
        x = self.pretrained_model.mlp(x) + adapter_x
        return x


class Adapter_CLIP(nn.Module):
    def __init__(self, embed: embedMethod, model_name: str):
        super().__init__()
        self.name = model_name
        self.origin_model, _ = clip.load(model_name)
        for param in self.origin_model.parameters():
            param.requires_grad = False
        # Add adapter to vision transformer
        new_visual_model = []
        for block in self.origin_model.visual.transformer.resblocks:
            new_visual_model.append(AdapterResidualAttentionBlock(block))
        new_visual_model = nn.Sequential(*new_visual_model)
        self.origin_model.visual.transformer.resblocks = new_visual_model
        # Embedding method
        self.embed = embed
        self.Biobert = bert_token_embedding(self.name)
        for param in self.Biobert.parameters():
            param.requires_grad = False

    def encode_image(self, image):
        image_feature = self.origin_model.visual(image.type(self.origin_model.dtype))
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
