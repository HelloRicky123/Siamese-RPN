import torch, pdb
import torch.nn
from IPython import embed
import torch.nn.functional as F
import pdb
import random
import numpy as np


def rpn_cross_entropy(input, target):
    r"""
    :param input: (15x15x5,2)
    :param target: (15x15x5,)
    :return:
    """
    mask_ignore = target == -1
    mask_calcu = 1 - mask_ignore
    loss = F.cross_entropy(input=input[mask_calcu], target=target[mask_calcu])
    return loss


def rpn_cross_entropy_balance(input, target, num_pos, num_neg):
    r"""
    :param input: (N,1125,2)
    :param target: (15x15x5,)
    :return:
    """
    pos_index = np.random.choice(np.where(target.cpu().flatten() == 1)[0], target.shape[0] * num_pos)
    neg_index = np.random.choice(np.where(target.cpu().flatten() == 0)[0], target.shape[0] * num_neg)
    cal_index = np.append(neg_index, pos_index)

    loss = F.cross_entropy(input=input.reshape(-1, 2)[cal_index], target=target.flatten()[cal_index])
    return loss


def rpn_smoothL1(input, target, label):
    r'''
    :param input: torch.Size([1, 1125, 4])
    :param target: torch.Size([1, 1125, 4])
            label: (torch.Size([1, 1125]) pos neg or ignore
    :return:
    '''
    pos_index = np.where(label.cpu() == 1)
    loss = F.smooth_l1_loss(input[pos_index], target[pos_index])
    return loss
