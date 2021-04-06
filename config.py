
import numpy as np
import random
import os
#Train Setting
objs = {1:'ape', 2:'benchvise', 4:'cam', 5:'can', 6:'cat', 8:'driller', 9:'duck', 10:'eggbox',11:'glue', 12:'holepuncher', 13:'iron', 14:'lamp', 15:'phone',0:'multi'}
#640x480
class Config:
    def __init__(self,mode='train'):
        #Path Setting
        self.path = f'../dataset/'
        self.checkpoint='../checkpoints'
        self.cls_num = 20        
        self.res = 50
        self.size = 608
        self.multiscale = 3
        self.sizes = [608]#list(range(self.size-32*self.multiscale,self.size+32*self.multiscale+1,32)) 
        self.nms_threshold = 0.5
        self.dc_threshold = 0.95
        
        self.anchors = []
        self.anchor_divide=[(6,7,8),(3,4,5),(0,1,2)]
        self.anchor_num = len(self.anchors)
        self.model_path = "models/yolov3-spp.cfg"
        
        self.bs = 8       
        self.pre_trained_path = '../network_weights'
        self.augment = False
        #train_setting
        self.lr = 0.001
        self.weight_decay=5e-4
        self.momentum = 0.9
        #lr_scheduler
        self.min_lr = 5e-5
        self.lr_factor = 0.25
        self.patience = 12
        #exp_setting
        self.save_every_k_epoch = 15
        self.val_every_k_epoch = 10
        self.adjust_lr = False
        #loss hyp
        self.obj_scale = 2
        self.noobj_scale = 5
        self.cls_scale = 1
        self.reg_scale = 1#for giou
        self.ignore_threshold = 0.5
        self.match_threshold = 0#regard as match above this threshold
        self.base_epochs = [-1]#base epochs with large learning rate,adjust lr_facter with 0.1
        if mode=='train':
            self.file=f'./data/train.json'
            self.bs = 32 # batch size
            
            #augmentation parameter
            self.augment = True
            self.flip = True
            self.rot = 25
            self.crop = 0.3
            self.trans = .3
            self.scale = 0.2
            self.valid_scale = 0.25
            
        elif mode=='val':
            self.file = f'./data/val.json'
        elif mode=='trainval':
            self.file = f'./data/trainval.json'
        elif mode=='test':
            self.file = f'./data/test.json'
        
