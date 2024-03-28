import os
import sys
import numpy as np
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
        
        pose.T[0::2] -= pose.T[0]
        pose.T[1::2] -= pose.T[1]
        
        xs = pose.T[0::2]
        ys = pose.T[1::2]
        pose = pose.T / np.sqrt(xs[1:] ** 2 + ys[1:] ** 2).mean(axis=0)
               
        return pose.T
    
    @staticmethod    
    def denormalize_2d(pose, orig_pose):
        xs = orig_pose.T[0::2] - orig_pose.T[0]
        ys = orig_pose.T[1::2] - orig_pose.T[1]
        scale = np.sqrt(xs[1:] ** 2 + ys[1:] ** 2).mean(axis=0)
        
        pose = (pose * scale).T       
        pose[0::2] += orig_pose.T[0]
        pose[1::2] += orig_pose.T[1]
        
        
        return pose.T,scale
    
def _normalize_3d(pose):
        return Normalization.normalize_3d(pose)

def _normalize_2d(pose):
    return Normalization.normalize_2d(pose)

def _denormalize_2d(pose, orig_pose):
    return Normalization.denormalize_2d(pose, orig_pose)


