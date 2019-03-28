import torch
import cv2
import os
import numpy as np
import pickle
import lmdb
import hashlib
import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from torch.utils.data.dataset import Dataset

from IPython import embed


class test_dataset(Dataset):
    def __init__(self, video_path):
        self.image_names = glob.glob(video_path + '/*/*')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = self.image_names[idx]
        img = cv2.imread(image_path)
        # half_img = cv2.resize(img, None, fx=1 / 2, fy=1 / 2)
        # half_img_path = image_path.replace('whole','half')
        # dir_path = os.path.dirname(half_img_path)
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        # cv2.imwrite(half_img_path,half_img)
        return img


if __name__ == '__main__':
    # ------------------------------------------------------------------------
    whole_ssd_dataset = test_dataset('/dataset_ssd/test_speed/wholeimg')
    for i in tqdm(range(len(whole_ssd_dataset))):
        img = whole_ssd_dataset[i]

    whole_ssd_loader = DataLoader(whole_ssd_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=2,
                                  drop_last=True)
    data_iter = iter(whole_ssd_loader)
    for i in tqdm(range(len(whole_ssd_loader))):
        data = next(data_iter)

    # ------------------------------------------------------------------------

    half_ssd_dataset = test_dataset('/dataset_ssd/test_speed/halfimg')
    for i in tqdm(range(len(half_ssd_dataset))):
        img = half_ssd_dataset[i]

    half_ssd_loader = DataLoader(half_ssd_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=2,
                                 drop_last=True)
    data_iter = iter(half_ssd_loader)
    for i in tqdm(range(len(half_ssd_loader))):
        data = next(data_iter)

    # ------------------------------------------------------------------------

    whole_hdd_dataset = test_dataset('/home/zrq/test_speed/wholeimg')
    for i in tqdm(range(len(whole_hdd_dataset))):
        img = whole_hdd_dataset[i]

    whole_hdd_loader = DataLoader(whole_hdd_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=2,
                                  drop_last=True)
    data_iter = iter(whole_hdd_loader)
    for i in tqdm(range(len(whole_hdd_loader))):
        data = next(data_iter)
    # ------------------------------------------------------------------------
