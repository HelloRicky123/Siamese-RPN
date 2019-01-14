import numpy as np
import time
import cv2
import glob
import os
import sys

sys.path.append(os.getcwd())
from siamfc import SiamRPNTracker, config
from tqdm import tqdm
from IPython import embed


def run_SiamRPN(seq_path, model_path, init_box):
    x, y, w, h = init_box
    tracker = SiamRPNTracker(model_path)
    res = []
    frames = [seq_path + '/img/' + x for x in np.sort(os.listdir(seq_path + '/img'))]
    frames = [x for x in frames if '.jpg' in x]
    title = seq_path.split('/')[-1]
    if title == 'David':
        frames = frames[299:]
    elif title == 'Football1':
        frames = frames[0:74]
    elif title == 'Freeman3':
        frames = frames[0:460]
    elif title == 'Freeman4':
        frames = frames[0:283]
    elif title == 'Diving':
        frames = frames[:215]
    elif title == 'Tiger1':
        frames = frames[5:]
    # starting tracking
    tic = time.clock()
    for idx, frame in tqdm(enumerate(frames), total=len(frames)):
        frame = cv2.imread(frame)
        # frame = cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB)
        if idx == 0:
            tracker.init(frame, init_box)
            bbox = (x + w / 2 - 1 / 2, y + h / 2 - 1 / 2, w, h)
            bbox = np.array(bbox).astype(np.float64)
        else:
            bbox, score = tracker.update(frame)  # x,y,w,h
            bbox = np.array(bbox)
        res.append(list((bbox[0] - bbox[2] / 2 + 1 / 2, bbox[1] - bbox[3] / 2 + 1 / 2, bbox[2], bbox[3])))
    duration = time.clock() - tic
    result = {}
    result['res'] = res
    result['type'] = 'rect'
    result['fps'] = round(len(frames) / duration, 3)
    return result
