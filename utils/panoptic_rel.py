import os
import numpy as np
import glob
import json
import tqdm
import random

from utils.h36m import PoseDatasetBase
from utils.misc import ensuredir
from utils.panoptic_utils import get_uniform_camera_order
from utils.panoptic_utils import projectPoints

class Panoptic(PoseDatasetBase):

    def __init__(self, data_path, train=True, subjects=None, cameras=None, n_cams=-1, seqs=1, use_cache=True, cache_path = '../panoptic_custom_data/'):
        self.cameras = cameras
        self.cache_path = cache_path

        if subjects is None:
            if train:
                subjects = ['160224_haggling1', '160226_haggling1', '160422_haggling1', '160422_ultimatum1', '160906_band1', '160906_band2', '160906_band3', '160906_ian1', 
                '160906_ian2', '160906_ian3', '160906_ian5', '160906_pizza1', '161029_flute1', '161029_piano1', '161029_piano2', '161029_piano3', '161029_piano4', 
                '170221_haggling_b1', '170221_haggling_b2', '170221_haggling_b3', '170221_haggling_m1', '170221_haggling_m2', '170221_haggling_m3', '170224_haggling_a1', 
                '170224_haggling_a2', '170224_haggling_a3', '170224_haggling_b1', '170224_haggling_b2', '170224_haggling_b3', '170228_haggling_a1', '170228_haggling_a2', 
                '170228_haggling_a3', '170228_haggling_b1', '170228_haggling_b2', '170228_haggling_b3', '170307_dance5', '170404_haggling_a1', '170404_haggling_a2', 
                '170404_haggling_a3', '170404_haggling_b1', '170404_haggling_b2', '170404_haggling_b3', '170407_haggling_a1', '170407_haggling_a2', '170407_haggling_a3', 
                '170407_haggling_b1',
                 '170407_haggling_b2', '170407_haggling_b3', '170407_office2', '170915_office1', '171026_cello3', '171026_pose1', '171026_pose2', '171026_pose3', 
                 '171204_pose1', '171204_pose2', '171204_pose3', '171204_pose4', '171204_pose5', '171204_pose6']
                            

                
            else:
                subjects = [
                            '160226_haggling1', '160422_ultimatum1', '160906_band1', '160906_band2', '160906_band3', '160906_ian1', 
                    '160906_ian2', '160906_ian3', '160906_ian5', '160906_pizza1', '161029_flute1', '161029_piano1', '161029_piano2', '161029_piano3', '161029_piano4', 
                    '170221_haggling_b1', '170221_haggling_b2']
        
        self.extended_data_list = []

        self.seq_cameras = {}
        self.seq_names = []

        self.seq_skels = {}

        self.train = train
        self.n_cams = n_cams
        self.N = 19

        ensuredir(self.cache_path)

        npy_data_list = glob.glob(f'{self.cache_path}/*npy')

        self.get_camera_details(data_path, subjects, train=train)

        for seq_name in subjects:
            if any(seq_name in _ for _ in npy_data_list) and use_cache:
                print(f"loading '{seq_name}' from cache...")
                self.extended_data_list = np.concatenate([self.extended_data_list, np.load(f'{self.cache_path}scenario_{seq_name}.npy', allow_pickle=True)])
                self.seq_skels = np.load(f'{self.cache_path}scenario_{seq_name}_skels.npy', allow_pickle=True).item()
            else: 
                print(f"creating '{seq_name}' dataset...")
                self.seq_names.append(seq_name)
                hd_skel_json_path = data_path+seq_name+'/hdPose3d_stage1_coco19/'

                temp_data_list = []
                skel_points = {}                    
                for skels in tqdm.tqdm(os.listdir(hd_skel_json_path)):    
                    skel_json_fname = hd_skel_json_path+skels

                    if not os.path.isdir(skel_json_fname) and os.stat(skel_json_fname).st_size != 0:
                        frame_idx = int(skels[-13:-5])
                        with open(skel_json_fname) as dfile:
                            bframe = json.load(dfile)
                            bodies = []

                            # Cycle through all detected bodies
                            for b, body in enumerate(bframe['bodies']):                                
                                pose_xyz = np.array(body['joints19']).reshape((-1,4))                             
                                bodies.append(pose_xyz)
                                ref_b = random.choice(range(len(bframe['bodies'])))
                                ref_body = bframe['bodies'][ref_b]
                                ref_pose_xyz = np.array(ref_body['joints19']).reshape((-1,4))                             
                                
                                data_details = self.data_handler(seq_name, frame_idx, b, pose_xyz, ref_b, ref_pose_xyz)
                                if data_details is not None:                           
                                    temp_data_list.append(data_details)
                            skel_points[frame_idx] = bodies

                    self.seq_skels[seq_name] = skel_points
                with open(f'{self.cache_path}scenario_{seq_name}.npy', 'wb') as f:
                    np.save(f, np.asarray(temp_data_list))
                with open(f'{self.cache_path}scenario_{seq_name}_skels.npy', 'wb') as f:
                    np.save(f, self.seq_skels)
                self.extended_data_list = np.concatenate([self.extended_data_list, np.asarray(temp_data_list)])

    def data_handler(self, seq_name, frame_idx, b, pose_xyz, ref_b, ref_pose_xyz):
        return dict(seq_name = seq_name, frame_idx=frame_idx, body_idx=b, pose_xyz=pose_xyz, ref_body_idx=ref_b, ref_pose_xyz=ref_pose_xyz)

    def get_camera_details(self, data_path, subjects, train):
        camera_path = os.path.join(self.cache_path, 'scenario_{}cameras.npy'.format('train' if train else 'val'))
        camera_cache = glob.glob(camera_path)
        if len(camera_cache) == 1:
            print('loading camera details from cache')
            self.seq_cameras = np.load(camera_path, allow_pickle=True).item()
        else:
            print('generating camera details')
            for seq_name in subjects:
                hd_cameras = self.load_cameras(camera_path=data_path+seq_name+'/calibration_{0}.json'.format(seq_name))
                self.seq_cameras[seq_name] = hd_cameras
            print(f'saving camera details -- path: {self.cache_path}scenario_cameras.npy')
            with open(camera_path, 'wb') as f:
                    np.save(f, self.seq_cameras)

    def load_cameras(self, camera_path):
        # Load camera calibration parameters
        with open(camera_path) as cfile:
            calib = json.load(cfile)

        # Cameras are identified by a tuple of (panel#,node#)
        cameras = {(cam['panel'],cam['node']):cam for cam in calib['cameras']}

        # Convert data into numpy arrays for convenience
        for k,cam in cameras.items():    
            cam['K'] = np.matrix(cam['K'])
            cam['distCoef'] = np.array(cam['distCoef'])
            cam['R'] = np.matrix(cam['R'])
            cam['t'] = np.array(cam['t']).reshape((3,1))

        # Choose only HD cameras for visualization
        hd_cam_idx = zip([0] * 30,range(0,30))
        hd_cameras = [cameras[cam].copy() for cam in hd_cam_idx]
        
        return hd_cameras

    def __len__(self):
        return len(self.extended_data_list)

    def __getitem__(self, i):
        _data = self.extended_data_list[i]
        seq_name = _data['seq_name']
        frame_idx = _data['frame_idx']
        body_idx_a = _data['body_idx']  
        poses_xyz_a = _data['pose_xyz']
        body_idx_b = _data['ref_body_idx']  
        poses_xyz_b = _data['ref_pose_xyz']
        hd_cameras = self.seq_cameras[seq_name]

        proj_lists, X_lists, scale_lists = dict(a=[], b=[], r=[]), dict(a=[], b=[], r=[]), dict(a=[], b=[], r=[])
        if self.cameras is None:
            cam_id_list = np.random.choice(range(len(self.seq_cameras[seq_name])), self.n_cams, replace=False)    
        else:
            cam_id_list =  self.cameras


        P_a = poses_xyz_a.reshape(-1, 4)
        P_a = P_a[:, :3]

        P_b = poses_xyz_b.reshape(-1, 4)
        P_b = P_b[:, :3]

        for cam_id in cam_id_list:                        
            
            params = hd_cameras[cam_id]
                
            X_a = params['R'].dot(P_a.T).T
            X_a = np.asarray(X_a.reshape(-1, self.N * 3))

            X_b = params['R'].dot(P_b.T).T
            X_b= np.asarray(X_b.reshape(-1, self.N * 3))            

            X_a, scalea = self._normalize_3d(X_a)
            X_a = X_a.astype(np.float32).reshape(self.N, 3)

            X_b, scaleb = self._normalize_3d(X_b)
            X_b = X_b.astype(np.float32).reshape(self.N, 3)
            
            
            proj_a = projectPoints(P_a.T, params['K'], params['R'], params['t'], params['distCoef'])
            proj_b = projectPoints(P_b.T, params['K'], params['R'], params['t'], params['distCoef'])
            
            proj_a = proj_a[0:2, :].T
            proj_a = proj_a.reshape(-1, self.N * 2)  # shape=(length, 2*n_joints)
            proj_b = proj_b[0:2, :].T
            proj_b = proj_b.reshape(-1, self.N * 2)  # shape=(length, 2*n_joints)

            
            proj_a = self._normalize_2d(proj_a).astype(np.float32).reshape(self.N, 2)
            proj_b = self._normalize_2d(proj_b).astype(np.float32).reshape(self.N, 2)
            
            proj_lists['a'].append(proj_a)
            proj_lists['b'].append(proj_b)

            
            X_lists['a'].append(X_a)
            X_lists['b'].append(X_b)

            scale_lists['a'].append(scalea)
            scale_lists['b'].append(scaleb)
            
            
        return proj_lists, X_lists, scale_lists, seq_name, frame_idx, body_idx_a, body_idx_b, cam_id_list