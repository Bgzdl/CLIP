import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

label_dict = {'Well differentiated tubular adenocarcinoma': 0,
              'Moderately differentiated tubular adenocarcinoma': 1,
              'Poorly differentiated adenocarcinoma, non-solid type': 2,
              'Poorly differentiated adenocarcinoma, solid type': 2}


class sub_Patch(Dataset):
    def __init__(self, data, target, label, load=False, transform=None):
        self.data = data
        self.target = target
        self.label = label
        self.load = load
        self.transform = transform

    def load_img(self,img_path):
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        if self.load:
            return {'data': self.data[idx], 'target': self.target[idx], 'label': self.label[idx]}
        else:
            return {'data': self.load_img(self.data[idx]), 'target': self.target[idx], 'label': self.label[idx]}

    def __len__(self):
        return len(self.data)


class Patch(Dataset):
    def __init__(self, path, label_type: bool, transform=None, load=False):
        self.data_information = pd.read_csv(os.path.join(path, 'captions.csv'))
        self.label_type = label_type
        self.length = 0
        self.data = []
        self.target = []
        self.label = []
        self.path = path
        self.transform = transform
        self.load = load
        if self.load:
            self.preprocess()
        else:
            self.load_img_path()

    def load_img(self, img_path):
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

    def load_img_path(self):
        # 读取图片路径并存在list里
        for i, [_, d] in enumerate(self.data_information.iterrows()):
            text = d['subtype']
            if text in label_dict.keys():
                if ',' in text:
                    text = text.split(',')[0]
                imgs_path = os.path.join(self.path, 'new_dataset', d['id'])
                for root, dirs, files in os.walk(imgs_path):
                    for file in files:
                        img_path = os.path.join(root, file)
                        self.data.append(img_path)
                        self.label.append(label_dict[d['subtype']])
                        if self.label_type:
                            self.target.append(text)
                        else:
                            self.target.append(d['text'])

    def preprocess(self):
        # 读取数据并存在list里
        for i, [_, d] in enumerate(self.data_information.iterrows()):
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
                        self.label.append(label_dict[d['subtype']])
                        if self.label_type:
                            self.target.append(text)
                        else:
                            self.target.append(d['text'])

    def __getitem__(self, idx):
        if self.load:
            return {'data': self.data[idx], 'target': self.target[idx], 'label': self.label[idx]}
        else:
            return {'data': self.load_img(self.data[idx]), 'target': self.target[idx], 'label': self.label[idx]}

    def __len__(self):
        return len(self.data)

    def split(self):
        length = len(self.data)
        train = sub_Patch(self.data[:int(length * 0.2)], self.target[:int(length * 0.2)],
                          self.label[:int(length * 0.2)], self.load, self.transform)
        val = sub_Patch(self.data[int(length * 0.2):int(length * 0.98)],
                        self.target[int(length * 0.2):int(length * 0.98)],
                        self.label[int(length * 0.2):int(length * 0.98)], self.load, self.transform)
        test = sub_Patch(self.data[int(length * 0.98):], self.target[int(length * 0.98):],
                         self.label[int(length * 0.98):], self.load, self.transform)
        return [train, val, test]

    def Count_the_number_of_various_tags(self):
        count_0 = self.label.count(0)
        count_1 = self.label.count(1)
        count_2 = self.label.count(2)
        return [count_0, count_1, count_2]
