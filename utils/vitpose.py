import os
import numpy as np
import glob
import json
import tqdm
import random

from utils.h36m import PoseDatasetBase
from utils.panoptic_utils import get_uniform_camera_order
from utils.panoptic_utils import projectPoints

class Vitpose(PoseDatasetBase):

    def __init__(self, subjects):
        
        self.N = 17

        data_list = []

        for data_path in subjects:

            c0 = np.load(os.path.join(data_path, 'vid_results_0.npy'), allow_pickle=True)
            c1 = np.load(os.path.join(data_path, 'vid_results_1.npy'), allow_pickle=True)
            c2 = np.load(os.path.join(data_path, 'vid_results_2.npy'), allow_pickle=True)

            linyun0 = np.load(os.path.join(data_path, 'linyun_idx_0.npy'), allow_pickle=True)
            linyun1 = np.load(os.path.join(data_path, 'linyun_idx_1.npy'), allow_pickle=True)
            linyun2 = np.load(os.path.join(data_path, 'linyun_idx_2.npy'), allow_pickle=True)

            #self.extended_data_list = np.zeros((len(c0), 3, 17, 2), dtype=np.float32)

            for i, (p0, p1, p2) in enumerate(zip(c0, c1, c2)):
                l_idx0 = linyun0.item().get(i)
                l_idx1 = linyun1.item().get(i)
                l_idx2 = linyun2.item().get(i)
                if None not in [l_idx0, l_idx1, l_idx2]:                    
                    data_list.append(np.asarray([p0[l_idx0[0]]['keypoints'], p1[l_idx1[0]]['keypoints'], p2[l_idx2[0]]['keypoints']]))
                    data_list.append(np.asarray([p0[l_idx0[1]]['keypoints'], p1[l_idx1[1]]['keypoints'], p2[l_idx2[1]]['keypoints']]))

        data_list = np.asarray(data_list)

        self.extended_data_list = data_list[:, :, :, :2]
        self.score_2d = data_list[:, :, :, 2]

        data_list[:, :, :, 2] = (data_list[:, :, :, 2]-0.5)*2
        
    

    def __len__(self):
        return len(self.extended_data_list)

    def __getitem__(self, i):
        _data = self.extended_data_list[i]
        _score = self.score_2d[i]

        norm_list = []
        orig_list = []
        score_list = []
        
        for d, s in zip(_data, _score):       

            orig_list.append(d)                 
            
            dn = d.reshape(-1, self.N * 2) 
            
            dn = self._normalize_2d(dn).astype(np.float32).reshape(self.N, 2)
            
            norm_list.append(dn)
            score_list.append(s)
            
            
        return norm_list, orig_list, score_list