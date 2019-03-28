import glob
import os
import setproctitle

from fire import Fire
from run_SiamRPN import run_SiamRPN
from tqdm import tqdm

from IPython import embed


def main(model_path):
    kitti_root_path = '/mnt/datasets/kitti/tracking/training'
    video_path = kitti_root_path + '/image_02'
    videos = sorted([x for x in glob.glob(video_path + '/*') if os.path.isdir(x)])
    for video in videos:
        gt_file = video.replace('image', 'label') + '.txt'
        with open(gt_file, 'r') as f:
            gt = f.readlines()
        track_ids = sorted(list(set([x.split()[1] for x in gt])), key=int)
        if '-1' in track_ids:
            track_ids.remove('-1')
        for track_id in track_ids:
            frames = [x for x in gt if x.split()[1] == track_id]
            first_frame = video + '/' + frames[0][0].zfill(6) + '.png'


if __name__ == '__main__':
    program_name = os.getcwd().split('/')[-1]
    setproctitle.setproctitle('zrq test kitti ' + program_name)
    Fire(main)
