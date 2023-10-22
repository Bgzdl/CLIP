import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

label_dict = {'Well differentiated tubular adenocarcinoma': 0,
              'Moderately differentiated tubular adenocarcinoma': 1,
              'Poorly differentiated adenocarcinoma, non-solid type': 2,
              'Poorly differentiated adenocarcinoma, solid type': 2}


class Few_shot_train(Dataset):
    def __init__(self, path, transform=None, load=False, shot_num: int = 1):
        self.data_information = pd.read_csv(os.path.join(path, 'captions.csv'))
        self.shot_num = shot_num
        self.data = []
        self.target = []
        self.label = []
        self.path = path
        self.transform = transform
        self.load = load
        self.groups = self.Count_number_of_statistical_group()
        if self.load:
            self.preprocess(self.groups)
        else:
            self.load_img_path(self.groups)

    def load_img(self, img_path):
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

    def load_img_path(self, groups):
        group_0, group_1, group_2 = groups
        for group_name in group_0:
            text = 'Well differentiated tubular adenocarcinoma'
            imgs_path = os.path.join(self.path, 'new_dataset', group_name)
            for root, dirs, files in os.walk(imgs_path):
                for file in files:
                    img_path = os.path.join(root, file)
                    self.data.append(img_path)
                    self.label.append(0)
                    self.target.append(text)
        for group_name in group_1:
            text = 'Moderately differentiated tubular adenocarcinoma'
            imgs_path = os.path.join(self.path, 'new_dataset', group_name)
            for root, dirs, files in os.walk(imgs_path):
                for file in files:
                    img_path = os.path.join(root, file)
                    self.data.append(img_path)
                    self.label.append(1)
                    self.target.append(text)
        for group_name in group_2:
            text = 'Poorly differentiated adenocarcinoma'
            imgs_path = os.path.join(self.path, 'new_dataset', group_name)
            for root, dirs, files in os.walk(imgs_path):
                for file in files:
                    img_path = os.path.join(root, file)
                    self.data.append(img_path)
                    self.label.append(2)
                    self.target.append(text)

    def preprocess(self, groups):
        group_0, group_1, group_2 = groups
        for group_name in group_0:
            text = 'Well differentiated tubular adenocarcinoma'
            imgs_path = os.path.join(self.path, 'new_dataset', group_name)
            for root, dirs, files in os.walk(imgs_path):
                for file in files:
                    img_path = os.path.join(root, file)
                    image = Image.open(img_path).convert("RGB")
                    if self.transform is not None:
                        image = self.transform(image)
                    self.data.append(image)
                    self.label.append(0)
                    self.target.append(text)
        for group_name in group_1:
            text = 'Moderately differentiated tubular adenocarcinoma'
            imgs_path = os.path.join(self.path, 'new_dataset', group_name)
            for root, dirs, files in os.walk(imgs_path):
                for file in files:
                    img_path = os.path.join(root, file)
                    image = Image.open(img_path).convert("RGB")
                    if self.transform is not None:
                        image = self.transform(image)
                    self.data.append(image)
                    self.label.append(1)
                    self.target.append(text)
        for group_name in group_2:
            text = 'Poorly differentiated adenocarcinoma'
            imgs_path = os.path.join(self.path, 'new_dataset', group_name)
            for root, dirs, files in os.walk(imgs_path):
                for file in files:
                    img_path = os.path.join(root, file)
                    image = Image.open(img_path).convert("RGB")
                    if self.transform is not None:
                        image = self.transform(image)
                    self.data.append(image)
                    self.label.append(2)
                    self.target.append(text)

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

    def Count_number_of_statistical_group(self):
        def get_keys_with_max_n_chars(dictionary, n):
            result = []
            for key, value in dictionary.items():
                result.append((key, value))
            result.sort(key=lambda x: x[1], reverse=True)
            return [key for key, value in result[:n]]

        group_0_dict = dict()
        group_1_dict = dict()
        group_2_dict = dict()
        for i, [_, d] in enumerate(self.data_information.iterrows()):
            text = d['subtype']
            if text in label_dict.keys():
                imgs_path = os.path.join(self.path, 'new_dataset', d['id'])
                file_count = len(
                    [name for name in os.listdir(imgs_path) if os.path.isfile(os.path.join(imgs_path, name))])
                if label_dict[d['subtype']] == 0:
                    group_0_dict[d['id']] = file_count
                elif label_dict[d['subtype']] == 1:
                    group_1_dict[d['id']] = file_count
                else:
                    group_2_dict[d['id']] = file_count

        group_0 = get_keys_with_max_n_chars(group_0_dict, self.shot_num)
        group_1 = get_keys_with_max_n_chars(group_1_dict, self.shot_num)
        group_2 = get_keys_with_max_n_chars(group_2_dict, self.shot_num)
        return group_0, group_1, group_2


class Few_shot_val(Dataset):
    def __init__(self, path, transform=None, load=False, shot_num: int = 1):
        self.data_information = pd.read_csv(os.path.join(path, 'captions.csv'))
        self.shot_num = shot_num
        self.data = []
        self.target = []
        self.label = []
        self.path = path
        self.transform = transform
        self.load = load
        groups = self.Count_number_of_statistical_group()
        groups = groups[0] + groups[1] + groups[2]
        if self.load:
            self.preprocess(groups)
        else:
            self.load_img_path(groups)

    def load_img(self, img_path):
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

    def Count_number_of_statistical_group(self):
        def get_keys_with_max_n_chars(dictionary, n):
            result = []
            for key, value in dictionary.items():
                result.append((key, value))
            result.sort(key=lambda x: x[1], reverse=True)
            return [key for key, value in result[:n]]

        group_0_dict = dict()
        group_1_dict = dict()
        group_2_dict = dict()
        for i, [_, d] in enumerate(self.data_information.iterrows()):
            text = d['subtype']
            if text in label_dict.keys():
                imgs_path = os.path.join(self.path, 'new_dataset', d['id'])
                file_count = len(
                    [name for name in os.listdir(imgs_path) if os.path.isfile(os.path.join(imgs_path, name))])
                if label_dict[d['subtype']] == 0:
                    group_0_dict[d['id']] = file_count
                elif label_dict[d['subtype']] == 1:
                    group_1_dict[d['id']] = file_count
                else:
                    group_2_dict[d['id']] = file_count

        group_0 = get_keys_with_max_n_chars(group_0_dict, self.shot_num)
        group_1 = get_keys_with_max_n_chars(group_1_dict, self.shot_num)
        group_2 = get_keys_with_max_n_chars(group_2_dict, self.shot_num)
        return group_0, group_1, group_2

    def load_img_path(self, groups):
        # 读取图片路径并存在list里
        for i, [_, d] in enumerate(self.data_information.iterrows()):
            text = d['subtype']
            if text in label_dict.keys():
                if ',' in text:
                    text = text.split(',')[0]
                if d['id'] in groups:
                    continue
                imgs_path = os.path.join(self.path, 'new_dataset', d['id'])
                for root, dirs, files in os.walk(imgs_path):
                    for file in files:
                        img_path = os.path.join(root, file)
                        self.data.append(img_path)
                        self.label.append(label_dict[d['subtype']])
                        self.target.append(text)

    def preprocess(self, groups):
        # 读取数据并存在list里
        for i, [_, d] in enumerate(self.data_information.iterrows()):
            text = d['subtype']
            if text in label_dict.keys():
                if ',' in text:
                    text = text.split(',')[0]
                if d['id'] in groups:
                    continue
                imgs_path = os.path.join(self.path, 'new_dataset', d['id'])
                for root, dirs, files in os.walk(imgs_path):
                    for file in files:
                        img_path = os.path.join(root, file)
                        image = Image.open(img_path).convert("RGB")
                        if self.transform is not None:
                            image = self.transform(image)
                        self.data.append(image)
                        self.label.append(label_dict[d['subtype']])
                        self.target.append(text)

    def __getitem__(self, idx):
        if self.load:
            return {'data': self.data[idx], 'target': self.target[idx], 'label': self.label[idx]}
        else:
            return {'data': self.load_img(self.data[idx]), 'target': self.target[idx], 'label': self.label[idx]}

    def __len__(self):
        return len(self.data)
