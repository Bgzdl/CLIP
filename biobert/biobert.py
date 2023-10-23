import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertForNextSentencePrediction, BertConfig, BertForMaskedLM


class bert_token:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('./biobert-base-cased-v1.2')

    def tokenize(self, text: list):
        outputs = self.tokenizer(text, padding='max_length', max_length=77, return_tensors='pt')
        T = outputs['input_ids']
        return T


bert = bert_token()


class bert_token_embedding(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.name = model_name
        model_config = BertConfig.from_pretrained('./biobert-base-cased-v1.2')
        model_config.output_hidden_states = True
        model_config.output_attentions = True
        self.bert_model = BertModel.from_pretrained('./biobert-base-cased-v1.2', config=model_config)
        for param in self.bert_model.parameters():
            param.require_grad = False
        self.adaptive_layer = nn.Linear(768, 512)

    def forward(self, text):
        attention_mask = torch.ones(text.shape, dtype=torch.long).cuda()
        outputs = self.bert_model(text, attention_mask=attention_mask)
        x = outputs[0]
        if self.name == 'ViT-B/16':
            x = self.adaptive_layer(x)
        elif self.name == 'ViT-L/14':
            pass
        else:
            raise Exception('model name error')
        return x
