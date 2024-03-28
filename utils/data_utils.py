#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.autograd.variable import Variable
import os


def readCSVasFloat(filename):
    """
    Borrowed from SRNN code. Reads a csv and returns a float matrix.
    https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34
    Args
      filename: string. Path to the csv file
    Returns
      returnArray: the read data in a float32 matrix
    """
    returnArray = []
    lines = open(filename).readlines()
    for line in lines:
        line = line.strip().split(',')
        if len(line) > 0:
            returnArray.append(np.array([np.float32(x) for x in line]))

    returnArray = np.array(returnArray)
    return returnArray

def expmap2rotmat_torch(r):
    """
    Converts expmap matrix to rotation
    batch pytorch version ported from the corresponding method above
    :param r: N*3
    :return: N*3*3
    """
    theta = torch.norm(r, 2, 1)
    r0 = torch.div(r, theta.unsqueeze(1).repeat(1, 3) + 0.0000001)
    r1 = torch.zeros_like(r0).repeat(1, 3)
    r1[:, 1] = -r0[:, 2]
    r1[:, 2] = r0[:, 1]
    r1[:, 5] = -r0[:, 0]
    r1 = r1.view(-1, 3, 3)
    r1 = r1 - r1.transpose(1, 2)
    n = r1.data.shape[0]
    R = Variable(torch.eye(3, 3).repeat(n, 1, 1)).float().cuda() + torch.mul(
        torch.sin(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3), r1) + torch.mul(
        (1 - torch.cos(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3)), torch.matmul(r1, r1))
    return R

def _some_variables_cmu():
    """
    We define some variables that are useful to run the kinematic tree
    Args
      None
    Returns
      parent: 32-long vector with parent-child relationships in the kinematic tree
      offset: 96-long vector with bone lenghts
      rotInd: 32-long list with indices into angles
      expmapInd: 32-long list with indices into expmap angles
    """

    parent = np.array([0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14, 15, 16, 17, 18, 19, 16,
                       21, 22, 23, 24, 25, 26, 24, 28, 16, 30, 31, 32, 33, 34, 35, 33, 37]) - 1

    offset = 70 * np.array(
        [0, 0, 0, 0, 0, 0, 1.65674000000000, -1.80282000000000, 0.624770000000000, 2.59720000000000, -7.13576000000000,
         0, 2.49236000000000, -6.84770000000000, 0, 0.197040000000000, -0.541360000000000, 2.14581000000000, 0, 0,
         1.11249000000000, 0, 0, 0, -1.61070000000000, -1.80282000000000, 0.624760000000000, -2.59502000000000,
         -7.12977000000000, 0, -2.46780000000000, -6.78024000000000, 0, -0.230240000000000, -0.632580000000000,
         2.13368000000000, 0, 0, 1.11569000000000, 0, 0, 0, 0.0196100000000000, 2.05450000000000, -0.141120000000000,
         0.0102100000000000, 2.06436000000000, -0.0592100000000000, 0, 0, 0, 0.00713000000000000, 1.56711000000000,
         0.149680000000000, 0.0342900000000000, 1.56041000000000, -0.100060000000000, 0.0130500000000000,
         1.62560000000000, -0.0526500000000000, 0, 0, 0, 3.54205000000000, 0.904360000000000, -0.173640000000000,
         4.86513000000000, 0, 0, 3.35554000000000, 0, 0, 0, 0, 0, 0.661170000000000, 0, 0, 0.533060000000000, 0, 0, 0,
         0, 0, 0.541200000000000, 0, 0.541200000000000, 0, 0, 0, -3.49802000000000, 0.759940000000000,
         -0.326160000000000, -5.02649000000000, 0, 0, -3.36431000000000, 0, 0, 0, 0, 0, -0.730410000000000, 0, 0,
         -0.588870000000000, 0, 0, 0, 0, 0, -0.597860000000000, 0, 0.597860000000000])
    offset = offset.reshape(-1, 3)

    rotInd = [[6, 5, 4],
              [9, 8, 7],
              [12, 11, 10],
              [15, 14, 13],
              [18, 17, 16],
              [21, 20, 19],
              [],
              [24, 23, 22],
              [27, 26, 25],
              [30, 29, 28],
              [33, 32, 31],
              [36, 35, 34],
              [],
              [39, 38, 37],
              [42, 41, 40],
              [45, 44, 43],
              [48, 47, 46],
              [51, 50, 49],
              [54, 53, 52],
              [],
              [57, 56, 55],
              [60, 59, 58],
              [63, 62, 61],
              [66, 65, 64],
              [69, 68, 67],
              [72, 71, 70],
              [],
              [75, 74, 73],
              [],
              [78, 77, 76],
              [81, 80, 79],
              [84, 83, 82],
              [87, 86, 85],
              [90, 89, 88],
              [93, 92, 91],
              [],
              [96, 95, 94],
              []]
    posInd = []
    for ii in np.arange(38):
        if ii == 0:
            posInd.append([1, 2, 3])
        else:
            posInd.append([])

    expmapInd = np.split(np.arange(4, 118) - 1, 38)

    return parent, offset, posInd, expmapInd

def fkl_torch(angles, parent, offset, rotInd, expmapInd):
    """
    pytorch version of fkl.
    convert joint angles to joint locations
    batch pytorch version of the fkl() method above
    :param angles: N*99
    :param parent:
    :param offset:
    :param rotInd:
    :param expmapInd:
    :return: N*joint_n*3
    """
    n = angles.data.shape[0]
    j_n = offset.shape[0]
    p3d = Variable(torch.from_numpy(offset)).float().cuda().unsqueeze(0).repeat(n, 1, 1)
    angles = angles[:, 3:].contiguous().view(-1, 3)
    R = expmap2rotmat_torch(angles).view(n, j_n, 3, 3)
    for i in np.arange(1, j_n):
        if parent[i] > 0:
            R[:, i, :, :] = torch.matmul(R[:, i, :, :], R[:, parent[i], :, :]).clone()
            p3d[:, i, :] = torch.matmul(p3d[0, i, :], R[:, parent[i], :, :]) + p3d[:, parent[i], :]
    return p3d

def expmap2xyz_torch_cmu(expmap):
    parent, offset, rotInd, expmapInd = _some_variables_cmu()
    xyz = fkl_torch(expmap, parent, offset, rotInd, expmapInd)
    return xyz


def normalize_data(data, data_mean, data_std, dim_to_use, actions, one_hot):
    """
    Normalize input data by removing unused dimensions, subtracting the mean and
    dividing by the standard deviation
    Args
      data: nx99 matrix with data to normalize
      data_mean: vector of mean used to normalize the data
      data_std: vector of standard deviation used to normalize the data
      dim_to_use: vector with dimensions used by the model
      actions: list of strings with the encoded actions
      one_hot: whether the data comes with one-hot encoding
    Returns
      data_out: the passed data matrix, but normalized
    """
    data_out = {}
    nactions = len(actions)

    if not one_hot:
        # No one-hot encoding... no need to do anything special
        for key in data.keys():
            data_out[key] = np.divide((data[key] - data_mean), data_std)
            data_out[key] = data_out[key][:, dim_to_use]

    else:
        # TODO hard-coding 99 dimensions for un-normalized human poses
        for key in data.keys():
            data_out[key] = np.divide((data[key][:, 0:99] - data_mean), data_std)
            data_out[key] = data_out[key][:, dim_to_use]
            data_out[key] = np.hstack((data_out[key], data[key][:, -nactions:]))

    return data_out


def normalization_stats(completeData):
    """"
    Also borrowed for SRNN code. Computes mean, stdev and dimensions to ignore.
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L33
    Args
      completeData: nx99 matrix with data to normalize
    Returns
      data_mean: vector of mean used to normalize the data
      data_std: vector of standard deviation used to normalize the data
      dimensions_to_ignore: vector with dimensions not used by the model
      dimensions_to_use: vector with dimensions used by the model
    """
    data_mean = np.mean(completeData, axis=0)
    data_std = np.std(completeData, axis=0)

    dimensions_to_ignore = []
    dimensions_to_use = []

    dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
    dimensions_to_use.extend(list(np.where(data_std >= 1e-4)[0]))

    data_std[dimensions_to_ignore] = 1.0

    return data_mean, data_std, dimensions_to_ignore, dimensions_to_use


"""all methods above are borrowed from https://github.com/una-dinosauria/human-motion-prediction"""


def define_actions_cmu(action):
    """
    Define the list of actions we are using.
    Args
      action: String with the passed action. Could be "all"
    Returns
      actions: List of strings of actions
    Raises
      ValueError if the action is not included in H3.6M
    """

    actions = ["basketball", "basketball_signal", "directing_traffic", "jumping", "running", "soccer", "walking",
               "washwindow"]
    if action in actions:
        return [action]

    if action == "all":
        return actions

    raise (ValueError, "Unrecognized action: %d" % action)


def load_data_cmu(path_to_dataset, actions, input_n, output_n, data_std=0, data_mean=0, is_test=False):
    seq_len = input_n + output_n
    nactions = len(actions)
    sampled_seq = []
    complete_seq = []
    for action_idx in np.arange(nactions):
        action = actions[action_idx]
        path = '{}/{}'.format(path_to_dataset, action)
        count = 0
        for _ in os.listdir(path):
            count = count + 1
        for examp_index in np.arange(count):
            filename = '{}/{}/{}_{}.txt'.format(path_to_dataset, action, action, examp_index + 1)
            action_sequence = readCSVasFloat(filename)
            n, d = action_sequence.shape
            even_list = range(0, n, 2)
            the_sequence = np.array(action_sequence[even_list, :])
            num_frames = len(the_sequence)
            if not is_test:
                fs = np.arange(0, num_frames - seq_len + 1)
                fs_sel = fs
                for i in np.arange(seq_len - 1):
                    fs_sel = np.vstack((fs_sel, fs + i + 1))
                fs_sel = fs_sel.transpose()
                seq_sel = the_sequence[fs_sel, :]
                if len(sampled_seq) == 0:
                    sampled_seq = seq_sel
                    complete_seq = the_sequence
                else:
                    sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
                    complete_seq = np.append(complete_seq, the_sequence, axis=0)
            else:
                source_seq_len = 50
                target_seq_len = 25
                total_frames = source_seq_len + target_seq_len
                batch_size = 8
                SEED = 1234567890
                rng = np.random.RandomState(SEED)
                for _ in range(batch_size):
                    idx = rng.randint(0, num_frames - total_frames)
                    seq_sel = the_sequence[
                              idx + (source_seq_len - input_n):(idx + source_seq_len + output_n), :]
                    seq_sel = np.expand_dims(seq_sel, axis=0)
                    if len(sampled_seq) == 0:
                        sampled_seq = seq_sel
                        complete_seq = the_sequence
                    else:
                        sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
                        complete_seq = np.append(complete_seq, the_sequence, axis=0)

    if not is_test:
        data_std = np.std(complete_seq, axis=0)
        data_mean = np.mean(complete_seq, axis=0)

    dimensions_to_ignore = []
    dimensions_to_use = []
    dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
    dimensions_to_use.extend(list(np.where(data_std >= 1e-4)[0]))
    data_std[dimensions_to_ignore] = 1.0
    data_mean[dimensions_to_ignore] = 0.0

    return sampled_seq, dimensions_to_ignore, dimensions_to_use, data_mean, data_std


def load_data_cmu_3d(path_to_dataset, actions, input_n, output_n, data_std=0, data_mean=0, is_test=False):
    seq_len = input_n + output_n
    nactions = len(actions)
    sampled_seq = []
    complete_seq = []
    for action_idx in np.arange(nactions):
        action = actions[action_idx]
        path = '{}/{}'.format(path_to_dataset, action)
        count = 0
        for _ in os.listdir(path):
            count = count + 1
        for examp_index in np.arange(count):
            filename = '{}/{}/{}_{}.txt'.format(path_to_dataset, action, action, examp_index + 1)
            action_sequence = readCSVasFloat(filename)
            n, d = action_sequence.shape
            exptmps = Variable(torch.from_numpy(action_sequence)).float().cuda()
            xyz = expmap2xyz_torch_cmu(exptmps)
            xyz = xyz.view(-1, 38 * 3)
            xyz = xyz.cpu().data.numpy()
            action_sequence = xyz

            even_list = range(0, n, 2)
            the_sequence = np.array(action_sequence[even_list, :])
            num_frames = len(the_sequence)
            if not is_test:
                fs = np.arange(0, num_frames - seq_len + 1)
                fs_sel = fs
                for i in np.arange(seq_len - 1):
                    fs_sel = np.vstack((fs_sel, fs + i + 1))
                fs_sel = fs_sel.transpose()
                seq_sel = the_sequence[fs_sel, :]
                if len(sampled_seq) == 0:
                    sampled_seq = seq_sel
                    complete_seq = the_sequence
                else:
                    sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
                    complete_seq = np.append(complete_seq, the_sequence, axis=0)
            else:
                source_seq_len = 50
                target_seq_len = 25
                total_frames = source_seq_len + target_seq_len
                batch_size = 8
                SEED = 1234567890
                rng = np.random.RandomState(SEED)
                for _ in range(batch_size):
                    idx = rng.randint(0, num_frames - total_frames)
                    seq_sel = the_sequence[
                              idx + (source_seq_len - input_n):(idx + source_seq_len + output_n), :]
                    seq_sel = np.expand_dims(seq_sel, axis=0)
                    if len(sampled_seq) == 0:
                        sampled_seq = seq_sel
                        complete_seq = the_sequence
                    else:
                        sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
                        complete_seq = np.append(complete_seq, the_sequence, axis=0)

    if not is_test:
        data_std = np.std(complete_seq, axis=0)
        data_mean = np.mean(complete_seq, axis=0)

    joint_to_ignore = np.array([0, 1, 2, 7, 8, 13, 16, 20, 29, 24, 27, 33, 36])
    dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    dimensions_to_use = np.setdiff1d(np.arange(complete_seq.shape[1]), dimensions_to_ignore)

    data_std[dimensions_to_ignore] = 1.0
    data_mean[dimensions_to_ignore] = 0.0

    return sampled_seq, dimensions_to_ignore, dimensions_to_use, data_mean, data_std
