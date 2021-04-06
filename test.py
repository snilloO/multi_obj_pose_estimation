###libs
import os
import argparse
import torch
from torch.utils.data import DataLoader
###files
from config import Config as cfg
from dataProcessing import WheatDet
from models.network import *
from trainer import Trainer
import warnings

warnings.filterwarnings('ignore')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=int, default=0, help="start from epoch?")
    parser.add_argument("--exp",type=str,default='exp',help="name of exp")
    args = parser.parse_args()
    
    

    #get data config
    config  = cfg()
    val_cfg = cfg('val')
    trainval_cfg = cfg('trainval')
    test_set = WheatDet(trainval_cfg,train=False)
    test_loader = DataLoader(test_set,batch_size=trainval_cfg.bs,shuffle=False,pin_memory=False,collate_fn=test_set.collate_fn)
    datasets = {'test':test_loader}
    config.exp_name = args.exp
    config.device = torch.device("cuda")
    torch.cuda.empty_cache()
    #network
    network = YOLO(config.res,config.int_shape,config.cls_num)
    det = Trainer(config,datasets,network,(0,0))
    det.test()