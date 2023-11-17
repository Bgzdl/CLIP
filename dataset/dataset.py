import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import random

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

    def load_img(self, img_path):
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
        # shuffle
        indices = list(range(len(self.data)))
        random.shuffle(indices)
        self.data = [self.data[i] for i in indices]
        self.target = [self.target[i] for i in indices]
        self.label = [self.label[i] for i in indices]
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


class Patch_Dataset(Dataset):
    def __init__(self, path, mode='train', load=False, transform=None, seed=0):
        self.path = path
        self.load = load
        self.mode = mode
        self.seed = seed
        self.transform = transform
        group_names, group_labels, group_targets = self.get_split()
        self.data, self.group_names, self.labels, self.targets, self.num_of_groups = self.load_data(group_names, group_labels, group_targets)
        indices = list(range(len(group_names)))
        random.shuffle(indices)
        self.data = [self.data[i] for i in indices]
        self.group_names = [self.group_names[i] for i in indices]
        self.targets = [self.targets[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.load:
            return {'data': self.data[idx], 'target': self.targets[idx],
                    'label': self.labels[idx], 'group_name': self.group_names[idx]}
        else:
            return {'data': self.load_img(self.data[idx]), 'target': self.targets[idx],
                    'label': self.labels[idx], 'group_name': self.group_names[idx]}

    def load_img(self, img_path):
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

    def get_split(self):
        random.seed(self.seed)
        group_names = []
        group_labels = []
        group_targets = []
        data_information = pd.read_csv(os.path.join(self.path, 'captions.csv'))
        for _, row in data_information.iterrows():
            text = row['subtype']
            if text in label_dict.keys():
                if ',' in text:
                    text = text.split(',')[0]
                group_names.append(row['id'])
                group_labels.append(label_dict[row['subtype']])
                group_targets.append(row['text'])
        indices = list(range(len(group_names)))
        random.shuffle(indices)
        group_names = [group_names[i] for i in indices]
        group_labels = [group_labels[i] for i in indices]
        group_targets = [group_targets[i] for i in indices]
        if self.mode == 'train':
            return group_names[:int(len(group_names) * 0.2)], group_labels[:int(len(group_names) * 0.2)], \
                   group_targets[:int(len(group_names) * 0.2)]
        elif self.mode == 'val':
            return group_names[int(len(group_names) * 0.2):], group_labels[int(len(group_names) * 0.2):], \
                   group_targets[int(len(group_names) * 0.2):]
        else:
            raise Exception('mode must be train or val')

    def load_data(self, group_names, group_labels, group_targets):
        data = []
        data_group_name = []
        num_of_group = dict()
        labels = []
        targets = []
        for group_name, group_label, group_target in zip(group_names, group_labels, group_targets):
            imgs_path = os.path.join(self.path, 'new_dataset', group_name)
            for img_name in os.listdir(imgs_path):
                img_path = os.path.join(imgs_path, img_name)
                if os.path.isfile(img_path):
                    if self.load:
                        image = self.load_img(img_path)
                        data.append(image)
                    else:
                        data.append(img_path)
                    data_group_name.append(group_name)
                    labels.append(group_label)
                    targets.append(group_target)
            num_of_group[group_name] = len(os.listdir(imgs_path))
        return data, data_group_name, labels, targets, num_of_group

    def get_ground_true(self):
        group_names, group_labels, group_targets = self.get_split()
        return group_names, group_labels

    def Count_the_number_of_various_tags(self):
        count_0 = self.labels.count(0)
        count_1 = self.labels.count(1)
        count_2 = self.labels.count(2)
        return [count_0, count_1, count_2]

    def get_group_length(self):
        return self.group_length

    def get_num_of_group(self):
        return self.num_of_groups
