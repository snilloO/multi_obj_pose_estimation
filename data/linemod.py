import argparse
import numpy as np 
import yaml
import os
from plyfile import PlyData
from progress.bar import Bar
import json
cam_K= np.array([572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0]).reshape(3,3)
path = '../../dataset/Linemod_preprocessed/'
single = [1,2,4,5,6,8,9,10,11,12,13,14,15]
objs = {1:'ape', 2:'benchvise', 4:'cam', 5:'can', 6:'cat', 8:'driller', 9:'duck', 10:'eggbox',11:'glue', 12:'holepuncher', 13:'iron', 14:'lamp', 15:'phone',0:'multi'}
def calculate_projections(vertices,R,t):
    if len(R)==9:
        R = R.reshape(3,3)
    t = t.reshape(3,1)
    if vertices.shape[0]!=3:
        vertices = vertices.T
    pts = cam_K @ R @ vertices + cam_K @ t
    pts[:2,:] /=pts[2,:]
    return pts[:2,:].T
def get_3D_corners(vertices):
    
    min_x = np.min(vertices[0,:])
    max_x = np.max(vertices[0,:])
    min_y = np.min(vertices[1,:])
    max_y = np.max(vertices[1,:])
    min_z = np.min(vertices[2,:])
    max_z = np.max(vertices[2,:])

    corners = np.array([[min_x, min_y, min_z],
                        [min_x, min_y, max_z],
                        [min_x, max_y, min_z],
                        [min_x, max_y, max_z],
                        [max_x, min_y, min_z],
                        [max_x, min_y, max_z],
                        [max_x, max_y, min_z],
                        [max_x, max_y, max_z]])

    return corners.T
def generate_annotation_single(path,obj_id,diameter):    
    obj = objs[obj_id]    
    file_name = str(obj_id)
    while len(file_name)<2:
        file_name = '0'+file_name
    processed = {'class':obj_id,'diam':diameter,'obj_name':obj,'img_path':os.path.join(path,'data',file_name,'rgb')}
    processed['depth_path']=os.path.join(path,'data',file_name,'depth')
    processed['mask_path']=os.path.join(path,'data',file_name,'mask')
    gts = yaml.load(open(os.path.join(path,'data',file_name,'gt.yml')),Loader=yaml.FullLoader) 
    obj_mesh = PlyData.read(os.path.join(path,'models',f'obj_{file_name}.ply'))
    xyz = np.stack((obj_mesh['vertex']['x'],obj_mesh['vertex']['y'],obj_mesh['vertex']['z']))
    corners = get_3D_corners(xyz)
    kps = np.concatenate((xyz.mean(1).reshape(3,1),corners),1)#objcet center of preprocessed is not at zero
    bar = Bar('Processing',max = len(gts))
    processed['trainlist']=os.path.join(path,'data',file_name,'train.txt')
    processed['testlist']=os.path.join(path,'data',file_name,'test.txt')
    processed['gt_num']=len(gts)
    for idx in gts:
        annos = gts[idx]
        keep = []
        anno ={}
        for item in annos:
            if item['obj_id'] != obj_id:
                continue
            projs = calculate_projections(kps,np.array(item['cam_R_m2c']),np.array(item['cam_t_m2c']))
            item['kps'] = list(projs.reshape(-1))
            keep.append(item)
        img_name = str(idx)
        img_name = '0'*(4-len(img_name))+img_name
        assert os.path.exists(os.path.join(path,'data',file_name,'rgb',f'{img_name}.png'))
        anno['img_name'] = img_name
        anno['gt'] = keep
        processed[idx] = anno
        bar.next()
    bar.finish()
    json.dump(processed,open(f'{obj}.json','w'))
def generate_annotation_multi(path,diameters):    
    obj = objs[2]    
    file_name = '02'    
    processed = {'img_path':os.path.join(path,'data',file_name,'rgb')}
    processed['depth_path']=os.path.join(path,'data',file_name,'depth')
    processed['mask_path']=os.path.join(path,'data',file_name,'mask')
    gts = yaml.load(open(os.path.join(path,'data',file_name,'gt.yml')),Loader=yaml.FullLoader)
    processed['trainlist']=os.path.join(path,'data',file_name,'train.txt')
    processed['testlist']=os.path.join(path,'data',file_name,'test.txt')
    kps_3d = {}
    classes = {}
    for idx in single:
        name = str(idx)
        while len(name)<2:
            name = '0'+name
        obj_mesh = PlyData.read(os.path.join(path,'models',f'obj_{name}.ply'))
        xyz = np.stack((obj_mesh['vertex']['x'],obj_mesh['vertex']['y'],obj_mesh['vertex']['z']))
        corners = get_3D_corners(xyz)
        kps_3d[idx] = np.concatenate((xyz.mean(1).reshape(3,1),corners),1)#objcet center of preprocessed is not at zero
    bar = Bar('Processing',max = len(gts))
    processed['gt_num']=len(gts)
    for idx in gts:
        annos = gts[idx]
        anno ={}
        for item in annos:
            obj_id = item['obj_id']
            if obj_id not in classes.keys():
                classes[obj_id] = diameters[obj_id]
            projs = calculate_projections(kps_3d[obj_id],np.array(item['cam_R_m2c']),np.array(item['cam_t_m2c']))
            item['kps'] = list(projs.reshape(-1))
        img_name = str(idx)
        img_name = '0'*(4-len(img_name))+img_name
        assert os.path.exists(os.path.join(path,'data',file_name,'rgb',f'{img_name}.png'))
        anno['img_name'] = img_name
        anno['gt'] = annos
        processed[idx] = anno
        bar.next()
    bar.finish()
    processed['diamters'] = classes
    processed['cls_num'] = len(classes)
    json.dump(processed,open(f'multi.json','w'))
info = yaml.load(open(os.path.join(path,'models','models_info.yml')),Loader=yaml.FullLoader)
diameters = {}
for idx in single:
    diam = info[idx]['diameter']
    generate_annotation_single(path,idx,diam)
    diameters[idx] = diam
generate_annotation_multi(path,diameters)