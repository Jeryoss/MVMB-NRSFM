import os
import numpy as np
import glob
import json
import tqdm
import random
import scipy.io as scio

from utils.h36m import PoseDatasetBase
from utils.panoptic_utils import get_uniform_camera_order


SHELF_JOINTS_DEF = {
    'Right-Ankle': 0,
    'Right-Knee': 1,
    'Right-Hip': 2,
    'Left-Hip': 3,
    'Left-Knee': 4,
    'Left-Ankle': 5,
    'Right-Wrist': 6,
    'Right-Elbow': 7,
    'Right-Shoulder': 8,
    'Left-Shoulder': 9,
    'Left-Elbow': 10,
    'Left-Wrist': 11,
    'Bottom-Head': 12,
    'Top-Head': 13
}

LIMBS = [
    [0, 1],
    [1, 2],
    [3, 4],
    [4, 5],
    [2, 3],
    [6, 7],
    [7, 8],
    [9, 10],
    [10, 11],
    [2, 8],
    [3, 9],
    [8, 12],
    [9, 12],
    [12, 13]
]

def unfold_camera_param(camera):
    R = camera['R']
    T = camera['T']
    fx = camera['fx']
    fy = camera['fy']
    f = np.array([[fx], [fy]]).reshape(-1, 1)
    c = np.array([[camera['cx']], [camera['cy']]]).reshape(-1, 1)
    k = camera['k']
    p = camera['p']
    return R, T, f, c, k, p


def project_point_radial(x, R, T, f, c, k, p):
    """
    Args
        x: Nx3 points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        f: (scalar) Camera focal length
        c: 2x1 Camera center
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
    Returns
        ypixel.T: Nx2 points in pixel space
    """
    n = x.shape[0]
    xcam = R.dot(x.T - T)
    y = xcam[:2] / (xcam[2]+1e-5)

    r2 = np.sum(y**2, axis=0)
    radial = 1 + np.einsum('ij,ij->j', np.tile(k, (1, n)),
                           np.array([r2, r2**2, r2**3]))
    tan = p[0] * y[1] + p[1] * y[0]
    y = y * np.tile(radial + 2 * tan,
                    (2, 1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2)
    ypixel = np.multiply(f, y) + c
    return ypixel.T


def project_pose(x, camera):
    R, T, f, c, k, p = unfold_camera_param(camera)
    return project_point_radial(x, R, T, f, c, k, p)

def projectPoints(X, K, R, t, Kd):
   
    
    x = np.asarray(R@X + t)
    
    x[0:2,:] = x[0:2,:]/x[2,:]
    
    r = x[0,:]*x[0,:] + x[1,:]*x[1,:]
    
    x[0,:] = x[0,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[2]*x[0,:]*x[1,:] + Kd[3]*(r + 2*x[0,:]*x[0,:])
    x[1,:] = x[1,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[3]*x[0,:]*x[1,:] + Kd[2]*(r + 2*x[1,:]*x[1,:])

    x[0,:] = K[0,0]*x[0,:] + K[0,1]*x[1,:] + K[0,2]
    x[1,:] = K[1,0]*x[0,:] + K[1,1]*x[1,:] + K[1,2]
    
    return x

class Shelf(PoseDatasetBase):

    def __init__(self, data_path, train=True, multi_person=False):
        
        self.multi_person = multi_person
        self.dataset_root = data_path
        self.N = 14
        #self.extended_data_list = data_list[:, :, :, :2]
        self.cam_list = [0, 1, 2, 3, 4]
        self.num_views = len(self.cam_list)
        self.frame_range = list(range(0,  3200))
        '''
        if self.is_train:
            self.frame_range = list(range(0,  300)) + list(range(601,  3200))
            # self.frame_range = list(range(300, 601))
        else:
            self.frame_range = list(range(300, 601))
        '''
        # self.pred_pose2d = self._get_pred_pose2d()
        self.extended_data_list, self.samples = self._get_db()

    def _get_db(self):  

        width = 1032
        height = 776
      
        db = []
        cameras, pan_cams = self._get_cam()

        datafile = os.path.join(self.dataset_root, 'actorsGT.mat')
        data = scio.loadmat(datafile)
        actor_3d = np.array(np.array(data['actor3D'].tolist()).tolist()).squeeze()

        num_person = len(actor_3d)
        # num_frames = len(actor_3d[0])

        samples = []

        
        for i in self.frame_range:
            for k, cam in cameras.items():
                image = os.path.join("Camera" + k, "img_{:06d}.png".format(i))

                all_poses_3d = []
                all_poses_vis_3d = []
                all_poses = []
                all_projs = []
                all_poses_vis = []

                pc = 0
                for person in range(num_person):
                    pose3d = actor_3d[person][i] * 1000.0
                    if len(pose3d[0]) > 0:
                        samples.append((i, pc))
                        pc += 1
                        all_poses_3d.append(pose3d)
                        all_poses_vis_3d.append(
                            np.ones((self.N, 3)))
                        
                        pose2d = project_pose(pose3d, cam)
                        proj = projectPoints(pose3d.T, pan_cams[k]['K'], pan_cams[k]['R'], pan_cams[k]['t'], pan_cams[k]['distCoef'])
                        proj = proj[0:2, :].T
                        
                        x_check \
                            = np.bitwise_and(pose2d[:, 0] >= 0,
                                                pose2d[:, 0] <= width - 1)
                        y_check \
                            = np.bitwise_and(pose2d[:, 1] >= 0,
                                                pose2d[:, 1] <= height - 1)
                        check = np.bitwise_and(x_check, y_check)

                        joints_vis = np.ones((len(pose2d), 1))
                        joints_vis[np.logical_not(check)] = 0
                        all_poses.append(pose2d)
                        all_projs.append(proj)
                        all_poses_vis.append(
                            np.repeat(
                                np.reshape(
                                    joints_vis, (-1, 1)), 2, axis=1))

                cam['standard_T'] = np.dot(-cam['R'], cam['T'])

                db.append({
                    'image': os.path.join(self.dataset_root, image),
                    'joints_3d': all_poses_3d,
                    'joints_3d_vis': all_poses_vis_3d,
                    'joints_2d': all_poses,
                    'projs_2d': all_projs,
                    'joints_2d_vis': all_poses_vis,
                    'camera': cam,
                    # 'pred_pose2d': preds
                })

        return db, samples

    def _get_cam(self):
        cam_file = os.path.join(self.dataset_root, "calibration_shelf.json")
        with open(cam_file) as cfile:
            cameras = json.load(cfile)

        
        pan_cams = dict()
        for id, cam in cameras.items():
            for k, v in cam.items():
                cameras[id][k] = np.array(v)

            pan_cam = dict()

            
            pan_cam['R'] = cameras[id]['R']
            pan_cam['t'] = np.dot(-cameras[id]['R'], cameras[id]['T'])

            pan_cam['K'] = np.zeros((3, 3))
            pan_cam['K'][2, 2] = 1.
            pan_cam['K'][0, 0] = cameras[id]['fx']
            pan_cam['K'][1, 1] = cameras[id]['fy']
            pan_cam['K'][0, 2] = cameras[id]['cx']
            pan_cam['K'][1, 2] = cameras[id]['cy']

            pan_cam['distCoef'] = np.array([cameras[id]['k'][0], cameras[id]['k'][1], cameras[id]['p'][0], cameras[id]['p'][1], cameras[id]['k'][2]])

            pan_cams[id] = pan_cam

        return cameras, pan_cams   
    

    def __len__(self):
        if self.multi_person:
            return len(self.extended_data_list)
        else:
            return len(self.samples)

    def __getitem__(self, idx):

        norm_list = []
        orig_list = []

        X_list = []
        scale_list = []
        
        for k in range(self.num_views):

            if self.multi_person:
                _data = self.extended_data_list[self.num_views * idx + k]
                d = _data['projs_2d']

            else:
                f_idx, p_idx = self.samples[idx]

                _data = self.extended_data_list[self.num_views * f_idx + k]

                P = _data['joints_3d'][p_idx]

                
                X = _data['camera']['R'].dot(P.T).T
                X = X.reshape(-1, self.N * 3)  # shape=(length, 3*n_joints)
                X = np.asarray(X)
                X, scale = self._normalize_3d(X)
                X = X.astype(np.float32)
                scale = scale.astype(np.float32)
                X_list.append(X.reshape(self.N, 3))
                scale_list.append(scale)

                d = _data['projs_2d'][p_idx]

                orig_list.append(d)
                dn = d.reshape(-1, self.N * 2)
                dn = self._normalize_2d(dn).astype(np.float32).reshape(self.N, 2)            
                norm_list.append(dn)
            
        return norm_list, orig_list, scale_list, X_list