import cv2
import numpy as np
import argparse
import os
import json

from IPython import embed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='show images with groundtruth and detected box')
    parser.add_argument('--detected_box', '-dt', dest='dt_path', default='./result.json')
    parser.add_argument('--show_videos', '-s', dest='show_videos', default=None)
    args = parser.parse_args()

    with open(args.dt_path, 'r') as f:
        dt_file = json.load(f)

    data_path = '/dataset_ssd/OTB/data/'
    for model in dt_file.keys():
        videos = np.array(list(dt_file[model].keys()))
        if args.show_videos:
            ind = list(map(int, args.show_videos.split(',')))
            ind = np.array(ind)
            videos = videos[ind]

        for video in dt_file[model].keys():
            # ------ prepare frames ------------
            video_path = data_path + video
            frames = [video_path + '/img/' + x for x in np.sort(os.listdir(video_path + '/img'))]
            frames = [x for x in frames if '.jpg' in x]
            if video == 'David':
                frames = frames[299:]
            elif video == 'Football1':
                frames = frames[0:74]
            elif video == 'Freeman3':
                frames = frames[0:460]
            elif video == 'Freeman4':
                frames = frames[0:283]
            # ------ prepare dtbox ------------
            dt_boxes = dt_file[model][video]
            # ------ prepare gtbox ------------
            gt_path = video_path + '/groundtruth_rect.txt'
            with open(gt_path, 'r') as f:
                gt_boxes = f.readlines()
            if ',' in gt_boxes[0]:
                gt_boxes = [list(map(int, box.split(','))) for box in gt_boxes]
            else:
                gt_boxes = [list(map(int, box.split())) for box in gt_boxes]

            for i, frame in enumerate(frames):
                img = cv2.imread(frame)
                img = cv2.rectangle(img, (int(dt_boxes[i][0]), int(dt_boxes[i][1])),
                                    (
                                        int(dt_boxes[i][0] + dt_boxes[i][2] - 1),
                                        int(dt_boxes[i][1] + dt_boxes[i][3] - 1)),
                                    color=(0, 255, 0))
                img = cv2.rectangle(img, (int(gt_boxes[i][0]), int(gt_boxes[i][1])),
                                    (
                                        int(gt_boxes[i][0] + gt_boxes[i][2] - 1),
                                        int(gt_boxes[i][1] + gt_boxes[i][3] - 1)),
                                    color=(0, 0, 255))
                cv2.imshow(video, img)
                cv2.waitKey(30)
            # cv2.destroyAllWindows()
