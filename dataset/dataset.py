import torch
import os
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from PIL import Image

label_dict = {'Well differentiated tubular adenocarcinoma': 0,
              'Moderately differentiated tubular adenocarcinoma': 1,
              'Poorly differentiated adenocarcinoma, non-solid type': 2,
              'Poorly differentiated adenocarcinoma, solid type': 2}


class Patch(Dataset):
    def __init__(self, path, label_type: bool, transform=None):
        self.data_information = pd.read_csv(os.path.join(path, 'captions.csv'))
        self.label_type = label_type
        self.length = 0
        self.data = []
        self.label = []
        self.path = path
        self.transform = transform
        self.read()

    def read(self):
        # 读取数据并存在list里
        for i, [_, d] in enumerate(self.data_information.iterrows()):
            if i > 50:
                continue
            text = d['subtype']
            if text in label_dict.keys():
                if ',' in text:
                    text = text.split(',')[0]
                imgs_path = os.path.join(self.path, 'new_dataset', d['id'])
                for root, dirs, files in os.walk(imgs_path):
                    for file in files:
                        img_path = os.path.join(root, file)
                        image = Image.open(img_path).convert("RGB")
                        if self.transform is not None:
                            image = self.transform(image)
                        self.data.append(image)
                        if self.label_type:
                            self.label.append(text)
                        else:
                            self.label.append(d['text'])

    def __getitem__(self, idx):
        return {'data': self.data[idx], 'target': self.label[idx]}

    def __len__(self):
        return len(self.data)
