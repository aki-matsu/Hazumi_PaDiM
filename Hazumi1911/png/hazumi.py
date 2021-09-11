import os
# import tarfile
from PIL import Image
from tqdm import tqdm
# import urllib.request

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


# URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
#CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
#               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
#               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

CLASS_NAMES = ["1911F2001", "1911F2002", "1911F3001", "1911F3002", "1911F3003",
            "1911F4001", "1911F4002", "1911F4003", "1911F5001", "1911F5002",
            "1911F6001", "1911F6002", "1911F6003", "1911F7002", "1911M2001",
            "1911M2002", "1911M2003", "1911M4001", "1911M4002", "1911M5001",
            "1911M5002", "1911M6001", "1911M6002", "1911M6003", "1911M7001"]#, "1911M7002"]

class HazumiDataset(Dataset):
    def __init__(self, dataset_path='D:/dataset/mvtec_anomaly_detection', class_name='1911F2001', is_train=True,
                 resize=256, cropsize=224):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        # self.mvtec_folder_path = os.path.join(root_path, 'mvtec_anomaly_detection')

        # download dataset if not exist
        # self.download()

        # load dataset
        # x: 画像データ, y: ラベル, mask: 正解マスク
        self.x, self.y = self.load_dataset_folder()

        # set transforms
        self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        return x, y

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'validation'
        x, y = [], []

        #img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        hazumi_insert_dir = 'experiment/splited_test_label_0_seed_1'
        img_dir = os.path.join(self.dataset_path, self.class_name, hazumi_insert_dir, phase)

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'label_1':
                y.extend([0] * len(img_fpath_list))
                
            else:
                y.extend([1] * len(img_fpath_list))
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y)
