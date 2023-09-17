import pandas as pd
import os
import shutil


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath='', label=0, domain=-1, classname=''):
        assert isinstance(impath, str)
        assert isinstance(label, int)
        assert isinstance(domain, int)
        assert isinstance(classname, str)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


def read(dataset_dir):
    data = pd.read_csv(os.path.join(dataset_dir, 'captions.csv'))
    label_dict = {'Well differentiated tubular adenocarcinoma': 0,
                  'Moderately differentiated tubular adenocarcinoma': 1,
                  'Poorly differentiated adenocarcinoma, non-solid type': 2,
                  'Poorly differentiated adenocarcinoma, solid type': 2}
    train = []
    val = []
    test = []
    # 读取训练集
    for i, [_, d] in enumerate(data.iterrows()):
        text = d['subtype']
        if text in label_dict.keys():
            label = label_dict[text]
            for img_name in os.listdir(os.path.join(dataset_dir, 'patches_captions')):
                if d['id'] in img_name:
                    image_dir = os.path.join(dataset_dir, 'patches_captions', img_name)
                    if ',' in text:
                        text = text.split(',')[0]
                    if i < 800:
                        train.append(Datum(impath=image_dir, label=label, classname=text))
                    elif 900 > i >= 800:
                        val.append(Datum(impath=image_dir, label=label, classname=text))
                    else:
                        test.append(Datum(impath=image_dir, label=label, classname=text))

    print(train[0]._impath, train[0]._label, train[0]._classname)
    print(len(train), len(val), len(test))

def rebuild_dataset(dataset_dir):
    source_dir = os.path.join(dataset_dir, 'patches_captions')
    target = os.path.join(dataset_dir, 'new_dataset')
    for img_name in os.listdir(source_dir):
        img_class = img_name.split('_')[0]
        target_class = os.path.join(target, img_class)
        if not os.path.exists(target_class):
            os.makedirs(target_class)
        shutil.copy(os.path.join(source_dir, img_name), target_class)


rebuild_dataset('data')