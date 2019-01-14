import numpy as np
import cv2
import torch
import torch.nn.functional as F
import time
import torchvision.transforms as transforms

from .alexnet import SiameseAlexNet
from .config import config
from .custom_transforms import ToTensor
from .utils import get_exemplar_image, get_instance_image, box_transform_inv
from .generate_anchors import generate_anchors

from IPython import embed

torch.set_num_threads(1)  # otherwise pytorch will take all cpus


class SiamRPNTracker:
    def __init__(self, model_path):
        self.model = SiameseAlexNet()
        checkpoint = torch.load(model_path)
        if 'model' in checkpoint.keys():
            self.model.load_state_dict(torch.load(model_path)['model'])
        else:
            self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.cuda()
        self.model.eval()
        self.transforms = transforms.Compose([
            ToTensor()
        ])

        valid_scope = 2 * config.valid_scope + 1
        self.anchors = generate_anchors(config.total_stride, config.anchor_base_size, config.anchor_scales,
                                        config.anchor_ratios,
                                        valid_scope)
        self.window = np.tile(np.outer(np.hanning(config.score_size), np.hanning(config.score_size))[None, :],
                              [config.anchor_num, 1, 1]).flatten()

    def _cosine_window(self, size):
        """
            get the cosine window
        """
        cos_window = np.hanning(int(size[0]))[:, np.newaxis].dot(np.hanning(int(size[1]))[np.newaxis, :])
        cos_window = cos_window.astype(np.float32)
        cos_window /= np.sum(cos_window)
        return cos_window

    def init(self, frame, bbox):
        """ initialize siamfc tracker
        Args:
            frame: an RGB image
            bbox: one-based bounding box [x, y, width, height]
        """
        self.pos = np.array(
            [bbox[0] + bbox[2] / 2 - 1 / 2, bbox[1] + bbox[3] / 2 - 1 / 2])  # center x, center y, zero based
        self.target_sz = np.array([bbox[2], bbox[3]])  # width, height
        self.bbox = np.array([bbox[0] + bbox[2] / 2 - 1 / 2, bbox[1] + bbox[3] / 2 - 1 / 2, bbox[2], bbox[3]])

        self.origin_target_sz = np.array([bbox[2], bbox[3]])
        # get exemplar img
        self.img_mean = np.mean(frame, axis=(0, 1))

        exemplar_img, _, _ = get_exemplar_image(frame, self.bbox,
                                                config.exemplar_size, config.context_amount, self.img_mean)
        # get exemplar feature
        exemplar_img = self.transforms(exemplar_img)[None, :, :, :]
        self.model.track_init(exemplar_img.cuda())

    def update(self, frame):
        """track object based on the previous frame
        Args:
            frame: an RGB image

        Returns:
            bbox: tuple of 1-based bounding box(xmin, ymin, xmax, ymax)
        """
        instance_img, _, _, scale_x = get_instance_image(frame, self.bbox, config.exemplar_size,
                                                         config.instance_size,
                                                         config.context_amount, self.img_mean)
        instance_img = self.transforms(instance_img)[None, :, :, :]
        pred_score, pred_regression = self.model.track(instance_img.cuda())

        pred_conf = pred_score.reshape(-1, 2, config.anchor_num * config.score_size * config.score_size).permute(0,
                                                                                                                 2,
                                                                                                                 1)
        pred_offset = pred_regression.reshape(-1, 4,
                                              config.anchor_num * config.score_size * config.score_size).permute(0,
                                                                                                                 2,
                                                                                                                 1)
        delta = pred_offset[0].cpu().detach().numpy()
        box_pred = box_transform_inv(self.anchors, delta)
        score_pred = F.softmax(pred_conf, dim=2)[0, :, 1].cpu().detach().numpy()

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)

        def sz_wh(wh):
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)

        s_c = change(sz(box_pred[:, 2], box_pred[:, 3]) / (sz_wh(self.target_sz * scale_x)))  # scale penalty
        r_c = change((self.target_sz[0] / self.target_sz[1]) / (box_pred[:, 2] / box_pred[:, 3]))  # ratio penalty
        penalty = np.exp(-(r_c * s_c - 1.) * config.penalty_k)
        pscore = penalty * score_pred
        pscore = pscore * (1 - config.window_influence) + self.window * config.window_influence
        best_pscore_id = np.argmax(pscore)
        target = box_pred[best_pscore_id, :] / scale_x

        lr = penalty[best_pscore_id] * score_pred[best_pscore_id] * config.lr_box

        res_x = np.clip(target[0] + self.pos[0], 0, frame.shape[1])
        res_y = np.clip(target[1] + self.pos[1], 0, frame.shape[0])

        res_w = np.clip(self.target_sz[0] * (1 - lr) + target[2] * lr, config.min_scale * self.origin_target_sz[0],
                        config.max_scale * self.origin_target_sz[0])
        res_h = np.clip(self.target_sz[1] * (1 - lr) + target[3] * lr, config.min_scale * self.origin_target_sz[1],
                        config.max_scale * self.origin_target_sz[1])

        self.pos = np.array([res_x, res_y])
        self.target_sz = np.array([res_w, res_h])
        bbox = np.array([res_x, res_y, res_w, res_h])
        self.bbox = (
            np.clip(bbox[0], 0, frame.shape[1]).astype(np.float64),
            np.clip(bbox[1], 0, frame.shape[0]).astype(np.float64),
            np.clip(bbox[2], 10, frame.shape[1]).astype(np.float64),
            np.clip(bbox[3], 10, frame.shape[0]).astype(np.float64))
        return self.bbox, score_pred[best_pscore_id]
