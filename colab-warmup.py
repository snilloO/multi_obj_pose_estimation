###libs
import os
import argparse
import torch
from torch.utils.data import DataLoader
###files
from config import Config as cfg
from dataProcessing import VOC_dataset as dataset
from models.network import NetAPI
from trainer import Trainer
import warnings
from config import cal_anchors
from tqdm import tqdm
warnings.filterwarnings('ignore')
def main(args,cfgs):
    #get data config
    for k in cfgs:
        curset = dataset(cfgs[k],mode='test')
        loader = DataLoader(curset,batch_size=args.bs,shuffle=True,pin_memory=False,collate_fn=curset.collate_fn)
        for data in tqdm(loader):
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs",type=int,default=16,help="batchsize")
    args = parser.parse_args()
    cfgs = {}
    
    #Generate config for different dataset
    cfgs['train'] = cfg()
    cfgs['trainval'] = cfg('trainval')
    cfgs['val'] = cfg('val')
    cfgs['test'] = cfg('test')
    main(args,cfgs)
    
    

    