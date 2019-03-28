import lmdb
import cv2
import numpy as np
import os
import hashlib
import functools

from glob import glob
from fire import Fire
from tqdm import tqdm
from multiprocessing import Pool
from IPython import embed
import multiprocessing as mp
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
    def __init__(self, video_path, db):
        self.image_names = glob.glob(video_path + '/*/*')
        self.txn = db.begin(write=False)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = self.image_names[idx]

        key = hashlib.md5(image_path.encode()).digest()
        img_buffer = self.txn.get(key)
        img_buffer = np.frombuffer(img_buffer, np.uint8)
        img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
        # img = cv2.imread(image_path)
        return img


if __name__ == '__main__':
    embed()
    output_dir = '/dataset_ssd/test_speed/whole.lmdb'

    image_names = glob.glob('/dataset_ssd/test_speed/wholeimg' + '/*/*')
    db = lmdb.open(output_dir, map_size=int(5e9))

    for image_name in tqdm(image_names):
        img = cv2.imread(image_name)
        _, img_encode = cv2.imencode('.jpg', img)
        img_encode = img_encode.tobytes()

        with db.begin(write=True) as txn:
            txn.put(hashlib.md5(image_name.encode()).digest(), img_encode)
    # ------------------------------------------------------------------------
    db = lmdb.open(output_dir, readonly=True, map_size=int(5e9))

    whole_ssd_dataset = test_dataset('/dataset_ssd/test_speed/wholeimg', db)
    for i in tqdm(range(len(whole_ssd_dataset))):
        img = whole_ssd_dataset[i]

    whole_ssd_loader = DataLoader(whole_ssd_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=2,
                                  drop_last=True)
    data_iter = iter(whole_ssd_loader)
    for i in tqdm(range(len(whole_ssd_loader))):
        data = next(data_iter)
    # ------------------------------------------------------------------------

    output_dir = '/dataset_ssd/test_speed/half.lmdb'

    image_names = glob.glob('/dataset_ssd/test_speed/halfimg' + '/*/*')
    db = lmdb.open(output_dir, map_size=int(5e9))

    for image_name in tqdm(image_names):
        img = cv2.imread(image_name)
        _, img_encode = cv2.imencode('.jpg', img)
        img_encode = img_encode.tobytes()

        with db.begin(write=True) as txn:
            txn.put(hashlib.md5(image_name.encode()).digest(), img_encode)
    # ------------------------------------------------------------------------
    db = lmdb.open(output_dir, readonly=True, map_size=int(5e9))

    half_ssd_dataset = test_dataset('/dataset_ssd/test_speed/halfimg', db)
    for i in tqdm(range(len(half_ssd_dataset))):
        img = half_ssd_dataset[i]

    half_ssd_loader = DataLoader(half_ssd_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=2,
                                  drop_last=True)
    data_iter = iter(half_ssd_loader)
    for i in tqdm(range(len(half_ssd_loader))):
        data = next(data_iter)