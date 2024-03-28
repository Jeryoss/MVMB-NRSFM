import os
from torch.utils.data import Dataset
import numpy as np

import glob

import scipy.io
import collections
import typing
import copy
import pickle

# Joints in H3.6M -- data has 32 joints,
# but only 17 that move; these are the indices.
H36M_NAMES = [''] * 32
H36M_NAMES[0] = 'Hip'
H36M_NAMES[1] = 'RHip'
H36M_NAMES[2] = 'RKnee'
H36M_NAMES[3] = 'RFoot'
H36M_NAMES[6] = 'LHip'
H36M_NAMES[7] = 'LKnee'
H36M_NAMES[8] = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'


def project_point_radial(P, R, T, f, c, k, p):
    """
    Project points from 3d to 2d using camera parameters
    including radial and tangential distortion
    Args
    P: Nx3 points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: (scalar) Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
    Returns
    Proj: Nx2 points in pixel space
    D: 1xN depth of each point in camera space
    radial: 1xN radial distortion per point
    tan: 1xN tangential distortion per point
    r2: 1xN squared radius of the projected points before distortion
    """

    # P is a matrix of 3-dimensional points
    assert len(P.shape) == 2
    assert P.shape[1] == 3

    N = P.shape[0]
    X = R.dot(P.T - T)  # rotate and translate
    XX = X[:2, :] / X[2, :]
    r2 = XX[0, :] ** 2 + XX[1, :] ** 2

    radial = 1 + np.einsum(
        'ij,ij->j', np.tile(k, (1, N)), np.array([r2, r2 ** 2, r2 ** 3]))
    tan = p[0] * XX[1, :] + p[1] * XX[0, :]

    XXX = XX * np.tile(radial + tan, (2, 1)) + \
        np.outer(np.array([p[1], p[0]]).reshape(-1), r2)

    Proj = (f * XXX) + c
    Proj = Proj.T

    D = X[2,]

    return Proj, D, radial, tan, r2

class H36CompatibleJoints(object):
    joint_names = ['spine3', 'spine4', 'spine2', 'spine', 'pelvis',
                   'neck', 'head', 'head_top', 'left_clavicle', 'left_shoulder', 'left_elbow',
                   'left_wrist', 'left_hand', 'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist',
                   'right_hand', 'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe',
                   'right_hip', 'right_knee', 'right_ankle', 'right_foot', 'right_toe']
    joint_idx = [4, 23, 24, 25, 18, 19, 20, 3, 5, 7, 6, 9, 10, 11, 14, 15, 16]

    @staticmethod
    def convert_points(raw_vector):
        return numpy.array(
            [(int(raw_vector[i * 2]), int(raw_vector[i * 2 + 1])) for i in H36CompatibleJoints.joint_idx])

    @staticmethod
    def convert_points_3d(raw_vector):
        return numpy.array([
            (float(raw_vector[i * 3]), float(raw_vector[i * 3 + 1]), float(raw_vector[i * 3 + 2])) for i in
            H36CompatibleJoints.joint_idx])


class MPII3DDatasetUtil(object):
    mm3d_chest_cameras = [
        0, 2, 4, 7, 8
    ]  # Subset of chest high, used in "Monocular 3D Human Pose Estimation in-the-wild Using Improved CNN supervision"

    @staticmethod
    def read_cameraparam(path):
        params = collections.defaultdict(dict)
        index = 0
        for line in open(path):
            key = line.split()[0].strip()
            if key == "name":
                value = line.split()[1].strip()
                index = int(value)
            if key == "intrinsic":
                values = line.split()[1:]
                values = [float(value) for value in values]
                values = np.array(values).reshape((4, 4))
                params[index]["intrinsic"] = values
            if key == "extrinsic":
                values = line.split()[1:]
                values = [float(value) for value in values]
                values = np.array(values).reshape((4, 4))
                params[index]["extrinsic"] = values
        return params


MPII3DDatum = typing.NamedTuple('MPII3DDatum', [
    ('annotation_2d', np.ndarray),
    ('annotation_3d', np.ndarray),
    ('normalized_annotation_2d', np.ndarray),
    ('normalized_annotation_3d', np.ndarray),
    ('normalize_3d_scale', float),
])

class Normalization(object):
    @staticmethod
    def normalize_3d(pose):
        xs = pose.T[0::3] - pose.T[0]
        ys = pose.T[1::3] - pose.T[1]
        ls = np.sqrt(xs[1:] ** 2 + ys[1:] ** 2)
        scale = ls.mean(axis=0)
        pose = pose.T / scale
        pose[0::3] -= pose[0].copy()
        pose[1::3] -= pose[1].copy()
        pose[2::3] -= pose[2].copy()
        return pose.T, scale

    @staticmethod
    def normalize_2d(pose):
        xs = pose.T[0::2] - pose.T[0]
        ys = pose.T[1::2] - pose.T[1]
        pose = pose.T / np.sqrt(xs[1:] ** 2 + ys[1:] ** 2).mean(axis=0)
        mu_x = pose[0].copy()
        mu_y = pose[1].copy()
        pose[0::2] -= mu_x
        pose[1::2] -= mu_y
        return pose.T

    @staticmethod
    def normalize_3d_ref(pose, ref_pose):
        xa = (pose.T[0] + ref_pose.T[0])/2
        ya = (pose.T[1] + ref_pose.T[1])/2
        
        xs = np.concatenate([pose.T[0::3], ref_pose.T[0::3]]) - xa
        ys = np.concatenate([pose.T[1::3], ref_pose.T[1::3]]) - ya
        ls = np.sqrt(xs ** 2 + ys ** 2)
        scale = ls.mean(axis=0)
        pose = np.concatenate([pose.T, ref_pose.T]) / scale
        xa /= scale
        ya /= scale
        pose[0::3] -= xa
        pose[1::3] -= ya
        pose[2::3] -= pose[2].copy()
        return pose.T, scale

    @staticmethod
    def normalize_2d_ref(pose, ref_pose):

        xa = (pose.T[0] + ref_pose.T[0])/2
        ya = (pose.T[1] + ref_pose.T[1])/2

        xs = np.concatenate([pose.T[0::2], ref_pose.T[0::2]]) - xa
        ys = np.concatenate([pose.T[1::2], ref_pose.T[1::2]]) - ya

        scale = np.sqrt(xs ** 2 + ys ** 2).mean(axis=0)

        pose = np.concatenate([pose.T, ref_pose.T]) / scale

        xa /= scale
        ya /= scale

        
        pose[0::2] -= xa
        pose[1::2] -= ya
        return pose.T
    
    
class PoseDatasetBase(Dataset):
    def _normalize_3d(self, pose):
        return Normalization.normalize_3d(pose)

    def _normalize_2d(self, pose):
        return Normalization.normalize_2d(pose)  

    def _normalize_3d_ref(self, pose, ref_pose):
        return Normalization.normalize_3d_ref(pose, ref_pose)

    def _normalize_2d_ref(self, pose, ref_pose):
        return Normalization.normalize_2d_ref(pose, ref_pose)  

class MPII3DDataset(PoseDatasetBase):
    def __init__(self, annotations_glob="/mnt/dataset/MPII_INF_3DHP/mpi_inf_3dhp/*/*/annot.mat", train=True):
        self.dataset = []
        for annotation_path in glob.glob(annotations_glob):
            print("load ", annotation_path)
            annotation = scipy.io.loadmat(annotation_path)
            for camera in MPII3DDatasetUtil.mm3d_chest_cameras:
                for frame in range(len(annotation["annot2"][camera][0])):
                    annot_2d = H36CompatibleJoints.convert_points(annotation["annot2"][camera][0][frame])
                    annot_3d = H36CompatibleJoints.convert_points_3d(annotation["annot3"][camera][0][frame])
                    annot_3d_normalized, scale = self._normalize_3d(
                        annot_3d.reshape(-1, len(H36CompatibleJoints.joint_idx) * 3))
                    self.dataset.append(MPII3DDatum(
                        annotation_2d=annot_2d,
                        annotation_3d=annot_3d,
                        normalized_annotation_2d=self._normalize_2d(
                            annot_2d.reshape(-1, len(H36CompatibleJoints.joint_idx) * 2)),
                        normalized_annotation_3d=annot_3d_normalized,
                        normalize_3d_scale=scale,
                    ))
        if train == False:  # just small subset
            self.dataset = self.dataset[:1000]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i].normalized_annotation_2d, \
               self.dataset[i].normalized_annotation_3d, \
               self.dataset[i].normalize_3d_scale
    
class H36M(PoseDatasetBase):

    def __init__(self, action='all', length=1,
                 train=True, use_sh_detection=False, subjects=None, n_cams=-1):
        if subjects is None:
            if train:
                subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
            else:
                subjects = ['S9', 'S11']

        
        if not os.path.exists('data/h36m'):
            os.mkdir('data/h36m')

        if not os.path.exists('data/h36m/points_3d.pkl'):
            print('Downloading 3D points in Human3.6M dataset.')
            os.system('wget --no-check-certificate "https://onedriv' + \
                'e.live.com/download?cid=B08D60FE71FF90FD&resid=B08' + \
                'D60FE71FF90FD%2118616&authkey=AFIfEB6VYEZnhlE" -O ' + \
                'data/h36m/points_3d.pkl')
        with open('data/h36m/points_3d.pkl', 'rb') as f:
            p3d = pickle.load(f)
        if not os.path.exists('data/h36m/cameras.pkl'):
            print('Downloading camera parameters.')
            os.system('wget --no-check-certificate "https://onedriv' + \
                'e.live.com/download?cid=B08D60FE71FF90FD&resid=B08' + \
                'D60FE71FF90FD%2118615&authkey=AEUoi3s16rBTFRA" -O ' + \
                'data/h36m/cameras.pkl')
        with open('data/h36m/cameras.pkl', 'rb') as f:
            cams = pickle.load(f)
        if use_sh_detection:
            if not os.path.exists('data/h36m/sh_detect_2d.pkl'):
                print('Downloading detected 2D points by Stacked Hourglass.')
                os.system('wget --no-check-certificate "https://onedriv' + \
                    'e.live.com/download?cid=B08D60FE71FF90FD&resid=B08' + \
                    'D60FE71FF90FD%2118619&authkey=AMBf6RPcWQgjsh0" -O ' + \
                    'data/h36m/sh_detect_2d.pkl')
            with open('data/h36m/sh_detect_2d.pkl', 'rb') as f:
                p2d_sh = pickle.load(f)

        with open('data/actions.txt') as f:
            actions_all = f.read().split('\n')[:-1]
        if action == 'all':
            actions = actions_all
        elif action in actions_all:
            actions = [action]
        else:
            raise Exception('Invalid action.')

        dim_to_use_x = np.where(np.array([x != '' for x in H36M_NAMES]))[0] * 3
        dim_to_use_y = dim_to_use_x + 1
        dim_to_use_z = dim_to_use_x + 2
        dim_to_use = np.array(
            [dim_to_use_x, dim_to_use_y, dim_to_use_z]).T.flatten()
        self.N = len(dim_to_use_x)

        p3d = copy.deepcopy(p3d)
        self.data_list = []
        for s in subjects:
            for action_name in actions:
                def search(a):
                    fs = list(filter(
                        lambda x: x.split()[0] == a, p3d[s].keys()))
                    return fs

                files = []
                files += search(action_name)
                # 'Photo' is 'TakingPhoto' in S1
                if action_name == 'Photo':
                    files += search('TakingPhoto')
                # 'WalkDog' is 'WalkingDog' in S1
                if action_name == 'WalkDog':
                    files += search('WalkingDog')
                for file_name in files:
                    p3d[s][file_name] = p3d[s][file_name][:, dim_to_use]
                    L = p3d[s][file_name].shape[0]

                    if not (s == 'S11' and action_name == 'Directions'):
                    
                        # 50Hz -> 10Hz
                        for start_pos in range(0, L - length + 1, 5):
                            info = {'subject': s,
                                    'action_name': action_name,
                                    'start_pos': start_pos,
                                    'length': length,
                                    'file_name': file_name}
                            self.data_list.append(info)
        self.p3d = p3d
        self.cams = cams
        self.n_cams = n_cams
        self.train = train
        self.use_sh_detection = use_sh_detection
        if use_sh_detection:
            self.p2d_sh = p2d_sh

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        info = self.data_list[i]
        subject = info['subject']
        start_pos = info['start_pos']
        length = info['length']
        file_name = info['file_name']

        poses_xyz = self.p3d[subject][file_name][start_pos:start_pos + length]

        #print('poses_xyz', poses_xyz.shape)
        
        if self.use_sh_detection:
            if 'TakingPhoto' in file_name:
                file_name = file_name.replace('TakingPhoto', 'Photo')
            if 'WalkingDog' in file_name:
                file_name = file_name.replace('WalkingDog', 'WalkDog')
            sh_detect_xy = self.p2d_sh[subject][file_name]
            sh_detect_xy = sh_detect_xy[cam_name][start_pos:start_pos+length]

        P = poses_xyz.reshape(-1, 3)

        proj_list, X_list, scale_list = [], [], []
        for cam_name in list(self.cams[subject].keys())[:self.n_cams]:
            params = self.cams[subject][cam_name]            

            X = params['R'].dot(P.T).T
            X = X.reshape(-1, self.N * 3)  # shape=(length, 3*n_joints)

            X, scale = self._normalize_3d(X)
            X = X.astype(np.float32)
            scale = scale.astype(np.float32)

            if self.use_sh_detection:
                sh_detect_xy = self._normalize_2d(sh_detect_xy)
                sh_detect_xy = sh_detect_xy.astype(np.float32)
                proj_list.append(sh_detect_xy)
                X_list.append(X)
                scale_list.append(scale)
            else:
                proj = project_point_radial(P, **params)[0]
                proj = proj.reshape(-1, self.N * 2)  # shape=(length, 2*n_joints)
                proj = self._normalize_2d(proj)
                proj = proj.astype(np.float32)

                proj = proj.reshape(self.N, 2)
                proj_list.append(proj)
                X_list.append(X.reshape(self.N, 3))
                scale_list.append(scale)        

        return proj_list, X_list, scale_list