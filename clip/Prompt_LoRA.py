import torch
import torch.nn as nn
from .LoRA import LoRA_CLIP, get_max_indices
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
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.pretrained_model.origin_model.logit_scale.exp()
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
