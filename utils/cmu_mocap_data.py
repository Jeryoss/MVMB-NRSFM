import os
import sys
import numpy as np
import torch


from utils.amc_parser import parse_asf, parse_amc
from torch.utils.data import Dataset

def read_subject(subject, data_root):
    subject_dir = os.path.join(data_root, subject)

    asf_path = os.path.join(subject_dir, '{}.asf'.format(subject))
    joints = parse_asf(asf_path)
    subject_pose = []
    for amc_file in os.listdir(subject_dir):
        if amc_file.endswith('amc'):
            print(amc_file)  # printing file name of desired extension
            
            amc_path = os.path.join(subject_dir, amc_file)
            motions = parse_amc(amc_path)
            for motion in motions:
                joints['root'].set_motion(motion)
                pose = np.zeros((31,3), dtype=np.float64) 
                for j, joint in enumerate(joints.values()):
                    for i in range(3): 
                        pose[j, i] = joint.coordinate[i, 0]
                subject_pose.append(pose)
    return np.asarray(subject_pose)


class CmuMocapDataset(Dataset):
    def __init__(self, subject, data_root, transform=None):
        self.subject_pose2d = np.load(os.path.join(data_root, subject, 'points2d.npy'))
        self.subject_pose3d = np.load(os.path.join(data_root, subject, 'points3d.npy'))

    def __len__(self):
        return len(self.subject_pose2d)

    def __getitem__(self, idx):
        return self.subject_pose2d[idx], self.subject_pose3d[idx]