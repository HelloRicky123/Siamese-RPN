import pickle
import glob
import os
from tqdm import tqdm

from IPython import embed

if __name__ == '__main__':
    data_dir = '/dataset_ssd/ytb_vid_rpn/'
    meta_data_path = data_dir + 'meta_data.pkl'
    new_meta_data_path = data_dir + 'new_meta_data.pkl'

    meta_data = pickle.load(open(meta_data_path, 'rb'))
    meta_data = {x[0]: x[1] for x in meta_data}
    new_meta_data = {}
    for video_name in tqdm(meta_data.keys()):
        new_meta_data[video_name] = {}

        trajs = meta_data[video_name]
        for trkid in trajs.keys():
            new_meta_data[video_name][trkid] = {}
            for exemplar_idx in list(range(len(trajs[trkid]))):
                exemplar_name = glob.glob(
                    os.path.join(data_dir, video_name, trajs[trkid][exemplar_idx] + ".{:02d}.x*.jpg".format(trkid)))[0]
                new_meta_data[video_name][trkid][trajs[trkid][exemplar_idx]] = exemplar_name
    pickle.dump(new_meta_data, open(new_meta_data_path, 'wb'))
