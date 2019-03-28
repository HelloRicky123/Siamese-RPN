from IPython import embed

import numpy as np
import pickle
import os
import cv2
import functools
import xml.etree.ElementTree as ET
import sys
import multiprocessing as mp

from multiprocessing import Pool
from fire import Fire
from tqdm import tqdm
from glob import glob

sys.path.append(os.getcwd())
from net.config import config
from lib.utils import get_instance_image, add_box_img

if __name__ == '__main__':
    with open('/mnt/diska1/YT-BB/stdxml/yt_bb_detection_train.csv', 'r') as f:
        std_xml = f.readlines()
    std_xml_dict = {}
    for line in tqdm(std_xml):
        video_name, frame, class_id, class_name, track_id, present, xmin_scale, xmax_scale, ymin_scale, ymax_scale = \
            line.split(',')
        if video_name not in std_xml_dict.keys():
            std_xml_dict[video_name] = {}
        if frame not in std_xml_dict[video_name].keys():
            std_xml_dict[video_name][frame] = {}
        if class_id not in std_xml_dict[video_name][frame].keys():
            std_xml_dict[video_name][frame][class_id] = {}
        if track_id not in std_xml_dict[video_name][frame][class_id].keys():
            std_xml_dict[video_name][frame][class_id][track_id] = [class_name, present, xmin_scale, xmax_scale,
                                                                   ymin_scale, ymax_scale]
        else:
            embed()
    embed()
    for key in tqdm(std_xml_dict.keys()):
        with open('/dataset_ssd/std_xml_ytb/' + key + '.pkl', 'wb') as f:
            pickle.dump(std_xml_dict[key], f)
