import os
import numpy as np
import glob
import json
import tqdm

from utils.panoptic import Panoptic

from utils.misc import ensuredir
from utils.h36m import PoseDatasetBase
from utils.panoptic_utils import get_uniform_camera_order
from utils.panoptic_utils import projectPoints

class Panoptic_extended(Panoptic):
    
    def data_handler(self, seq_name, frame_idx, b, pose_xyz):
        hd_cameras = self.seq_cameras[seq_name]
        details = self.get_projection_details(seq_name, hd_cameras, pose_xyz)
        if len(details) > 0:
            return dict(seq_name = seq_name, frame_idx=frame_idx, body_idx=b, details=details)
        else:
            return None

    def is_whole_body_visible(self, proj, valid, hd_cameras, seq_name, cam_id): 
                visibilities = []
                for ip in range(proj.shape[1]):
                    if proj[0,ip]>=0 and proj[0,ip]<hd_cameras[cam_id]['resolution'][0] and proj[1,ip]>=0 and proj[1,ip]<hd_cameras[cam_id]['resolution'][1] and valid[ip]:
                        visibilities.append(True)
                    else:
                        visibilities.append(False)

                return all(visibilities)

    def get_projection_details(self, seq_name, hd_cameras, poses_xyz):
        P = poses_xyz.reshape(-1, 4)
        valid = poses_xyz.transpose()[3,:]>0.1
        P = P[:, :3]

        camera_results = {}

        cam_id_list = range(0,30)
        for cam_id in cam_id_list:
            proj_list, orig_proj_list, X_list, scale_list = [], [], [], []
            params = hd_cameras[cam_id]
                
            X = params['R'].dot(P.T).T
            X = X.reshape(-1, self.N * 3)  # shape=(length, 3*n_joints)

            X = np.asarray(X)
            X, scale = self._normalize_3d(X)
            X = X.astype(np.float32)
            scale = scale.astype(np.float32)

            orig_proj = projectPoints(P.T, params['K'], params['R'], params['t'], params['distCoef'])
            if self.is_whole_body_visible(orig_proj, valid, hd_cameras, seq_name, cam_id):
                
                proj = orig_proj[0:2, :].T
                proj = proj.reshape(-1, self.N * 2)  # shape=(length, 2*n_joints)
                proj = self._normalize_2d(proj)
                proj = proj.astype(np.float32)

                proj = proj.reshape(self.N, 2)
                orig_proj_list.append(orig_proj)
                proj_list.append(proj)
                X_list.append(X.reshape(self.N, 3))
                scale_list.append(scale)

                camera_results[cam_id] = dict(proj = proj_list, orig_proj = orig_proj_list, X_list = X_list, scale_list = scale_list)
        return camera_results

    def __getitem__(self, i):
        _data = self.extended_data_list[i]
        seq_name = _data['seq_name']
        frame_idx = _data['frame_idx']
        body_idx = _data['body_idx']  

        proj_list, X_list, scale_list = [], [], []
        if self.cameras is None:
            cam_id_list = np.random.choice(range(len(self.seq_cameras[seq_name])), self.n_cams, replace=False)    
        else:
            cam_id_list =  self.cameras

        for cam_id in cam_id_list:
            proj_list.append(_data['details'][cam_id]['proj_list'])
            X_list.append(_data['details'][cam_id]['X_list'])
            scale_list.append(_data['details'][cam_id]['scale_list'])
            
        return proj_list, X_list, scale_list, seq_name, frame_idx, body_idx, cam_id_list