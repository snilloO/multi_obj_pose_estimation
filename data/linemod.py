import argparse
import numpy as np 
import yaml
import os
cam_K= np.array([572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0]).reshape(3,3)
path = '../dataset/Linemod_preprocessed/data/gt.yml'
single = [1,2,4,5,6,8,9,10,11,12,13,14,15]
objs = {1:'ape', 2:'benchvise', 4:'cam', 5:'can', 6:'cat', 8:'driller', 9:'duck', 10:'eggbox',11:'glue', 12:'holepuncher', 13:'iron', 14:'lamp', 15:'phone']
def calculate_projections(vertices,R,t):
    if len(R)==9:
        R = R.reshape(3,3)
    t = t.reshape(3,1)
    if vertices.shape[0]!=3:
        vertices = vertices.T
    pts = cam_K @ R @ vertices + cam_K @ t
    pts[:2,:] /=pts[2,:]
    return pts[:2,:].T
def generate_kps(path,obj_id,diameter):
    mutli ={'class':[]}
    processed = {'class':obj_id}
    ob 
    gts = yaml.load(open(os.path.join(path,'gt.yml')),Loader='FullLoad') 