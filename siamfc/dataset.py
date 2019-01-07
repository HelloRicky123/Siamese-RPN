import torch
import cv2
import os
import numpy as np
import pickle
import lmdb
import hashlib
import glob
import xml.etree.ElementTree as ET

from torch.utils.data.dataset import Dataset
from .generate_anchors import generate_anchors
from .config import config
from .utils import box_transform, compute_iou, add_box_img

from IPython import embed


class ImagnetVIDDataset(Dataset):
    def __init__(self, db, video_names, data_dir, z_transforms, x_transforms, training=True):
        self.video_names = video_names
        self.data_dir = data_dir
        self.z_transforms = z_transforms
        self.x_transforms = x_transforms
        meta_data_path = os.path.join(data_dir, 'meta_data.pkl')
        self.meta_data = pickle.load(open(meta_data_path, 'rb'))
        self.meta_data = {x[0]: x[1] for x in self.meta_data}
        # filter traj len less than 2
        for key in self.meta_data.keys():
            trajs = self.meta_data[key]
            for trkid in list(trajs.keys()):
                if len(trajs[trkid]) < 2:
                    del trajs[trkid]

        self.txn = db.begin(write=False)
        self.num = len(self.video_names) if config.num_per_epoch is None or not training \
            else config.num_per_epoch

        # data augmentation
        self.max_stretch = config.scale_resize
        self.max_translate = config.max_translate
        self.random_crop_size = config.instance_size
        self.training = training

        valid_scope = 2 * config.valid_scope + 1
        self.anchors = generate_anchors(config.total_stride, config.anchor_base_size, config.anchor_scales,
                                        config.anchor_ratios,
                                        valid_scope)

    def imread(self, path):
        key = hashlib.md5(path.encode()).digest()
        img_buffer = self.txn.get(key)
        img_buffer = np.frombuffer(img_buffer, np.uint8)
        img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
        return img

    def _sample_weights(self, center, low_idx, high_idx, s_type='uniform'):
        weights = list(range(low_idx, high_idx))
        weights.remove(center)
        weights = np.array(weights)
        if s_type == 'linear':
            weights = abs(weights - center)
        elif s_type == 'sqrt':
            weights = np.sqrt(abs(weights - center))
        elif s_type == 'uniform':
            weights = np.ones_like(weights)
        return weights / sum(weights)

    def RandomStretch(self, sample, gt_w, gt_h):
        scale_h = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        scale_w = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        gt_w = gt_w * scale_w
        gt_h = gt_h * scale_h
        return cv2.resize(sample, None, fx=scale_w, fy=scale_h, interpolation=cv2.INTER_LINEAR), gt_w, gt_h

    def RandomCrop(self, sample, ):
        shape = sample.shape[:2]
        cy_o = (shape[0] - 1) // 2
        cx_o = (shape[1] - 1) // 2
        cy = np.random.randint(cy_o - self.max_translate,
                               cy_o + self.max_translate + 1)
        cx = np.random.randint(cx_o - self.max_translate,
                               cx_o + self.max_translate + 1)
        assert abs(cy - cy_o) <= self.max_translate and \
               abs(cx - cx_o) <= self.max_translate
        gt_cx = cx_o - cx
        gt_cy = cy_o - cy

        ymin = cy - self.random_crop_size // 2
        xmin = cx - self.random_crop_size // 2
        ymax = cy + self.random_crop_size // 2 + self.random_crop_size % 2
        xmax = cx + self.random_crop_size // 2 + self.random_crop_size % 2
        left = right = top = bottom = 0
        im_h, im_w = shape
        if xmin < 0:
            left = int(abs(xmin))
        if xmax > im_w:
            right = int(xmax - im_w)
        if ymin < 0:
            top = int(abs(ymin))
        if ymax > im_h:
            bottom = int(ymax - im_h)

        xmin = int(max(0, xmin))
        xmax = int(min(im_w, xmax))
        ymin = int(max(0, ymin))
        ymax = int(min(im_h, ymax))
        im_patch = sample[ymin:ymax, xmin:xmax]
        if left != 0 or right != 0 or top != 0 or bottom != 0:
            im_patch = cv2.copyMakeBorder(im_patch, top, bottom, left, right,
                                          cv2.BORDER_CONSTANT, value=0)
        return im_patch, gt_cx, gt_cy

    def compute_target(self, anchors, box):
        regression_target = box_transform(anchors, box)

        iou = compute_iou(anchors, box).flatten()
        # print(np.max(iou))
        pos_index = np.where(iou > config.pos_threshold)[0]
        neg_index = np.where(iou < config.neg_threshold)[0]
        label = np.ones_like(iou) * -1
        label[pos_index] = 1
        label[neg_index] = 0
        return regression_target, label

        # pos_index = np.random.choice(pos_index, config.num_pos)
        # neg_index = np.random.choice(neg_index, config.neg_pos)
        # max_index = np.argsort(iou.flatten())[-20:]
        # boxes = anchors[max_index]

    def __getitem__(self, idx):
        while True:
            idx = idx % len(self.video_names)
            video = self.video_names[idx]
            trajs = self.meta_data[video]
            # sample one trajs
            trkid = np.random.choice(list(trajs.keys()))
            traj = trajs[trkid]
            assert len(traj) > 1, "video_name: {}".format(video)
            # sample exemplar
            exemplar_idx = np.random.choice(list(range(len(traj))))
            # exemplar_name = os.path.join(self.data_dir, video, traj[exemplar_idx] + ".{:02d}.x*.jpg".format(trkid))
            exemplar_name = \
                glob.glob(os.path.join(self.data_dir, video, traj[exemplar_idx] + ".{:02d}.x*.jpg".format(trkid)))[0]
            exemplar_img = self.imread(exemplar_name)
            exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_BGR2RGB)
            # sample instance
            low_idx = max(0, exemplar_idx - config.frame_range)
            up_idx = min(len(traj), exemplar_idx + config.frame_range)

            # create sample weight, if the sample are far away from center
            # the probability being choosen are high
            weights = self._sample_weights(exemplar_idx, low_idx, up_idx, config.sample_type)
            instance = np.random.choice(traj[low_idx:exemplar_idx] + traj[exemplar_idx + 1:up_idx], p=weights)
            instance_name = glob.glob(os.path.join(self.data_dir, video, instance + ".{:02d}.x*.jpg".format(trkid)))[0]
            instance_img = self.imread(instance_name)
            instance_img = cv2.cvtColor(instance_img, cv2.COLOR_BGR2RGB)
            gt_w, gt_h = float(instance_name.split('_')[-2]), float(instance_name.split('_')[-1][:-4])

            if np.random.rand(1) < config.gray_ratio:
                exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_RGB2GRAY)
                exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_GRAY2RGB)
                instance_img = cv2.cvtColor(instance_img, cv2.COLOR_RGB2GRAY)
                instance_img = cv2.cvtColor(instance_img, cv2.COLOR_GRAY2RGB)

            exemplar_img = self.z_transforms(exemplar_img)
            instance_img, gt_w, gt_h = self.RandomStretch(instance_img, gt_w, gt_h)
            instance_img, gt_cx, gt_cy = self.RandomCrop(instance_img, )
            instance_img = self.x_transforms(instance_img)

            regression_target, conf_target = self.compute_target(self.anchors, np.array([gt_cx, gt_cy, gt_w, gt_h]))

            # img = instance_img.numpy().transpose(1, 2, 0)
            # pos_index = np.where(conf_target == 1)[0]
            # pos_anchor = self.anchors[pos_index]
            # frame = add_box_img(img, pos_anchor)
            # frame = add_box_img(frame, np.array([[gt_cx, gt_cy, gt_w, gt_h]]), color=(0, 255, 255))

            #
            # # debug the gt_box with original box
            # title = instance_name.split('/')[-1]
            # img = instance_img.numpy().transpose(1, 2, 0)
            # box = np.array([gt_cx, gt_cy, gt_w, gt_h])[None, :]
            # frame = add_box_img(img, box)
            # if 'train' in instance_name:
            #     img_name = '.'.join([instance_name.split('/')[-1].split('.')[0], 'JPEG'])
            #     img_path = glob.glob('/dataset_ssd/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_*/'
            #                          + video + '/' + img_name)[0]
            #     xml_path = glob.glob('/dataset_ssd/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_*/'
            #                          + video + '/' + img_name[:6] + '*')[0]
            #     tree = ET.parse(xml_path)
            #     root = tree.getroot()
            #     bboxes = []
            #     image = cv2.imread(img_path)
            #     for obj in root.iter('object'):
            #         bbox = obj.find('bndbox')
            #         bbox = list(map(int, [bbox.find('xmin').text,
            #                               bbox.find('ymin').text,
            #                               bbox.find('xmax').text,
            #                               bbox.find('ymax').text]))
            #         x_ctr = (bbox[0] + bbox[2]) / 2 - image.shape[1] / 2
            #         y_ctr = (bbox[1] + bbox[3]) / 2 - image.shape[0] / 2
            #         w = bbox[2] - bbox[0]
            #         h = bbox[3] - bbox[1]
            #         bbox = [x_ctr, y_ctr, w, h]
            #         bboxes.append(bbox)
            #     frame2 = add_box_img(image, bboxes)
            #     frame = frame[:, :, ::-1]
            #     show_img = np.hstack(
            #         [cv2.resize(frame, None, fx=frame2.shape[0] / frame.shape[0], fy=frame2.shape[0] / frame.shape[0]),
            #          frame2])
            # else:
            #     img_name = '.'.join([instance_name.split('/')[-1].split('.')[0], 'JPEG'])
            #     img_path = glob.glob('/dataset_ssd/ILSVRC2015/Data/VID/val/'
            #                          + video + '/' + img_name)[0]
            #     xml_path = glob.glob('/dataset_ssd/ILSVRC2015/Annotations/VID/val/'
            #                          + video + '/' + img_name[:6] + '*')[0]
            #     tree = ET.parse(xml_path)
            #     root = tree.getroot()
            #     bboxes = []
            #     image = cv2.imread(img_path)
            #     for obj in root.iter('object'):
            #         bbox = obj.find('bndbox')
            #         bbox = list(map(int, [bbox.find('xmin').text,
            #                               bbox.find('ymin').text,
            #                               bbox.find('xmax').text,
            #                               bbox.find('ymax').text]))
            #         x_ctr = (bbox[0] + bbox[2]) / 2 - image.shape[1] / 2
            #         y_ctr = (bbox[1] + bbox[3]) / 2 - image.shape[0] / 2
            #         w = bbox[2] - bbox[0]
            #         h = bbox[3] - bbox[1]
            #         box = [x_ctr, y_ctr, w, h]
            #         bboxes.append(box)
            #     frame2 = add_box_img(image, bboxes)
            #     frame = frame[:, :, ::-1]
            #     show_img = np.hstack(
            #         [cv2.resize(frame, None, fx=frame2.shape[0] / frame.shape[0], fy=frame2.shape[0] / frame.shape[0]),
            #          frame2])
            # embed()
            # cv2.imshow('gt_box.jpg', show_img)
            # cv2.waitKey(30)

            if len(np.where(conf_target == 1)[0]) > 0:
                break
            else:
                idx = np.random.randint(self.num)
        return exemplar_img, instance_img, regression_target, conf_target.astype(np.int64)

    def draw_img(self, img, boxes, name='1.jpg', color=(0, 255, 0)):
        # boxes (x,y,w,h)
        img = img.copy()
        img_ctx = (img.shape[1] + 1) / 2
        img_cty = (img.shape[0] + 1) / 2
        for box in boxes:
            point_1 = img_ctx - box[2] / 2 + box[0], img_cty - box[3] / 2 + box[1]
            point_2 = img_ctx + box[2] / 2 + box[0], img_cty + box[3] / 2 + box[1]
            img = cv2.rectangle(img, (int(point_1[0]), int(point_1[1])), (int(point_2[0]), int(point_2[1])),
                                color, 2)
        cv2.imwrite(name, img)

    def __len__(self):
        return self.num
