import torch
import torch.nn as nn
import torch.nn.functional as F
from embedMethod import embedMethod
from .model import CLIP, ResidualAttentionBlock, LayerNorm


class FeedforwardAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, init_scale=1e-3):
        super(FeedforwardAdapter, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

        # Initialize weights and biases
        self.fc1.weight.data.normal_(0, init_scale)
        self.fc2.weight.data.normal_(0, init_scale)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)

    def forward(self, x):
        # First linear transformation
        net = self.fc1(x)
        net = F.gelu(net)  # Apply GELU activation

        # Second linear transformation
        net = self.fc2(net)

        # Residual connection
        net += x

        return net


class AdapterResidualAttentionBlock(nn.Module):
    def __init__(self, origin_model: ResidualAttentionBlock):
        super().__init__()
        self.pretrained_model = origin_model
        d_model = origin_model.d_model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        self.attn_mask = origin_model.attn_mask
        self.adapter_layer_1 = FeedforwardAdapter(d_model)
        for param in self.adapter_layer_1.parameters():
            param.requires_grad = True
        self.adapter_layer_2 = FeedforwardAdapter(d_model)
        for param in self.adapter_layer_2.parameters():
            param.requires_grad = True
        self.ln_3 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.pretrained_model.attention(self.pretrained_model.ln_1(x))
        x = self.adapter_layer_1(x)
        x = x + self.pretrained_model.mlp(self.pretrained_model.ln_2(x))
        x = self.adapter_layer_2(x)
        x = self.ln_3(x)
        return x


class Adapter_CLIP(CLIP):
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
        for param in model.parameters():
            param.requires_grad = False
        new_model = nn.Sequential()
        for block in self.visual.transformer.resblocks:
            new_model.add_module('AdapterResidualAttentionBlock', AdapterResidualAttentionBlock(block))
        self.visual.transformer.resblocks = new_model
        self.embed = embed

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
