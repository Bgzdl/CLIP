import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

label_dict = ['Well differentiated tubular adenocarcinoma',
              'Moderately differentiated tubular adenocarcinoma',
              'Poorly differentiated adenocarcinoma']


class Few_shot_Dataset(Dataset):
    def __init__(self, path, dataset_type, transform=None, load=False, shot_num=0):
        self.length = 0
        self.data = []
        self.target = []
        self.label = []
        self.path = path
        self.dataset_type = dataset_type
        self.shot_num = shot_num
        self.transform = transform
        self.data_information = self.get_data_split_path()
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

    def get_data_split_path(self):
        if self.dataset_type == 'train':
            if self.shot_num == 0:
                information_path = os.path.join(self.path, 'split', 'train_all_0.2.csv')
            elif self.shot_num == 1:
                information_path = os.path.join(self.path, 'split', 'train_1_0.2.csv')
            elif self.shot_num == 2:
                information_path = os.path.join(self.path, 'split', 'train_2_0.2.csv')
            elif self.shot_num == 4:
                information_path = os.path.join(self.path, 'split', 'train_4_0.2.csv')
            elif self.shot_num == 8:
                information_path = os.path.join(self.path, 'split', 'train_8_0.2.csv')
            elif self.shot_num == 16:
                information_path = os.path.join(self.path, 'split', 'train_4_0.2.csv')
            else:
                raise Exception("Shot number should be 1, 2, 4, 8, 16")
        elif self.dataset_type == 'val':
            information_path = os.path.join(self.path, 'split', 'val_0.8.csv')
        else:
            raise Exception("Type should be train or val")
        return pd.read_csv(information_path)

    def load_img_path(self):
        for _, row in self.data_information.iterrows():
            image_path = row[0]
            group_name, _ = image_path.split('_')
            image_path = os.path.join(self.path, 'new_dataset', group_name, image_path)
            label = int(row[1])
            target = row[2]
            self.data.append(image_path)
            self.label.append(label)
            self.target.append(target)

    def preprocess(self):
        for _, row in self.data_information.iterrows():
            image_path = row[0]
            group_name, _ = image_path.split('_')
            image_path = os.path.join(self.path, 'new_dataset', group_name, image_path)
            image = Image.open(image_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            label = int(row[1])
            target = row[2]
            self.data.append(image)
            self.label.append(label)
            self.target.append(target)

    def __getitem__(self, idx):
        if self.load:
            return {'data': self.data[idx], 'target': self.target[idx], 'label': self.label[idx]}
        else:
            return {'data': self.load_img(self.data[idx]), 'target': self.target[idx], 'label': self.label[idx]}

    def __len__(self):
        return len(self.data)

    def Count_the_number_of_various_tags(self):
        count_0 = self.label.count(0)
        count_1 = self.label.count(1)
        count_2 = self.label.count(2)
        return [count_0, count_1, count_2]
