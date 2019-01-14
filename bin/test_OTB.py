import argparse
import os
import glob
import numpy as np
import re
import json
import matplotlib
import setproctitle
import functools
import multiprocessing as mp
import matplotlib.pyplot as plt

from run_SiamRPN import run_SiamRPN
from tqdm import tqdm
from IPython import embed
from multiprocessing import Pool


def embeded_numbers(s):
    re_digits = re.compile(r'(\d+)')
    pieces = re_digits.split(s)
    return int(pieces[1])


def embeded_numbers_results(s):
    re_digits = re.compile(r'(\d+)')
    pieces = re_digits.split(s)
    return int(pieces[-2])


def cal_iou(box1, box2):
    r"""

    :param box1: x1,y1,w,h
    :param box2: x1,y1,w,h
    :return: iou
    """
    x11 = box1[0]
    y11 = box1[1]
    x21 = box1[0] + box1[2] - 1
    y21 = box1[1] + box1[3] - 1
    area_1 = (x21 - x11 + 1) * (y21 - y11 + 1)

    x12 = box2[0]
    y12 = box2[1]
    x22 = box2[0] + box2[2] - 1
    y22 = box2[1] + box2[3] - 1
    area_2 = (x22 - x12 + 1) * (y22 - y12 + 1)

    x_left = max(x11, x12)
    x_right = min(x21, x22)
    y_top = max(y11, y12)
    y_down = min(y21, y22)

    inter_area = max(x_right - x_left + 1, 0) * max(y_down - y_top + 1, 0)
    iou = inter_area / (area_1 + area_2 - inter_area)
    return iou


def cal_success(iou):
    success_all = []
    overlap_thresholds = np.arange(0, 1.05, 0.05)
    for overlap_threshold in overlap_thresholds:
        success = sum(np.array(iou) > overlap_threshold) / len(iou)
        success_all.append(success)
    return np.array(success_all)


# def worker(video_paths, model_path):
#     results_ = {}
#     for video_path in tqdm(video_paths, total=len(video_paths)):
#         groundtruth_path = video_path + '/groundtruth_rect.txt'
#         assert os.path.isfile(groundtruth_path), 'groundtruth of ' + video_path + ' doesn\'t exist'
#         with open(groundtruth_path, 'r') as f:
#             boxes = f.readlines()
#         if ',' in boxes[0]:
#             boxes = [list(map(int, box.split(','))) for box in boxes]
#         else:
#             boxes = [list(map(int, box.split())) for box in boxes]
#         result = run_SiamFC(video_path, model_path, boxes[0])
#         results_box_video = result['res']
#         results_[os.path.abspath(model_path)][video_path.split('/')[-1]] = results_box_video
#         return results_


if __name__ == '__main__':
    program_name = os.getcwd().split('/')[-1]
    setproctitle.setproctitle('zrq test ' + program_name)
    parser = argparse.ArgumentParser(description='Test some models on OTB2015 or OTB2013')
    parser.add_argument('--model_paths', '-ms', dest='model_paths', nargs='+',
                        help='the path of models or the path of a model or folder')
    parser.add_argument('--videos', '-v', dest='videos')  # choices=['tb50', 'tb100', 'cvpr2013']
    parser.add_argument('--save_name', '-n', dest='save_name', default='result.json')
    args = parser.parse_args()

    # ------------ prepare data  -----------
    data_path = '/dataset_ssd/OTB/data/'
    if '50' in args.videos:
        direct_file = data_path + 'tb_50.txt'
    elif '100' in args.videos:
        direct_file = data_path + 'tb_100.txt'
    elif '13' in args.videos:
        direct_file = data_path + 'cvpr13.txt'
    else:
        raise ValueError('videos setting wrong')
    with open(direct_file, 'r') as f:
        direct_lines = f.readlines()
    video_names = np.sort([x.split('\t')[0] for x in direct_lines])
    video_paths = [data_path + x for x in video_names]

    # ------------ prepare models  -----------
    input_paths = [os.path.abspath(x) for x in args.model_paths]
    model_paths = []
    for input_path in input_paths:
        if os.path.isdir(input_path):
            input_path = os.path.abspath(input_path)
            model_path = sorted([x for x in os.listdir(input_path) if 'pth' in x], key=embeded_numbers)
            model_path = [input_path + '/' + x for x in model_path]
            model_paths.extend(model_path)
        elif os.path.isfile(input_path):
            model_path = os.path.abspath(input_path)
            model_paths.append(model_path)
        else:
            raise ValueError('model_path setting wrong')

    # ------------ starting validation  -----------
    results = {}
    for model_path in tqdm(model_paths, total=len(model_paths)):
        results[os.path.abspath(model_path)] = {}
        for video_path in tqdm(video_paths, total=len(video_paths)):
            # video_path = video_paths[-10]
            groundtruth_path = video_path + '/groundtruth_rect.txt'
            assert os.path.isfile(groundtruth_path), 'groundtruth of ' + video_path + ' doesn\'t exist'
            with open(groundtruth_path, 'r') as f:
                boxes = f.readlines()
            if ',' in boxes[0]:
                boxes = [list(map(int, box.split(','))) for box in boxes]
            else:
                boxes = [list(map(int, box.split())) for box in boxes]
            boxes = [np.array(box) - [1, 1, 0, 0] for box in boxes]
            result = run_SiamRPN(video_path, model_path, boxes[0])
            result_boxes = [np.array(box) + [1, 1, 0, 0] for box in result['res']]
            results[os.path.abspath(model_path)][video_path.split('/')[-1]] = [box.tolist() for box in result_boxes]

    # with Pool(processes=mp.cpu_count()) as pool:
    #     for ret in tqdm(pool.imap_unordered(
    #             functools.partial(worker, video_paths), model_paths), total=len(model_paths)):
    #         results.update(ret)
    json.dump(results, open(args.save_name, 'w'))

    # ------------ starting evaluation  -----------
    data_path = '/dataset_ssd/OTB/data/'
    results_eval = {}
    for model in sorted(list(results.keys()), ):
        results_eval[model] = {}
        success_all_video = []
        for video in results[model].keys():
            result_boxes = results[model][video]
            with open(data_path + video + '/groundtruth_rect.txt', 'r') as f:
                result_boxes_gt = f.readlines()
            if ',' in result_boxes_gt[0]:
                result_boxes_gt = [list(map(int, box.split(','))) for box in result_boxes_gt]
            else:
                result_boxes_gt = [list(map(int, box.split())) for box in result_boxes_gt]
            result_boxes_gt = [np.array(box) for box in result_boxes_gt]
            iou = list(map(cal_iou, result_boxes, result_boxes_gt))
            success = cal_success(iou)
            auc = np.mean(success)
            success_all_video.append(success)
            results_eval[model][video] = auc
        results_eval[model]['all_video'] = np.mean(success_all_video)
        print(model.split('/')[-1] + ' : ', np.mean(success_all_video))
    json.dump(results_eval, open('eval_' + args.save_name, 'w'))
