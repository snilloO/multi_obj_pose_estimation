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

warnings.filterwarnings('ignore')
def main(args,cfgs):
    #get data config
    config  = cfgs['train']
    val_cfg = cfgs['val']
    trainval_cfg = cfgs['trainval']
    test_cfg = cfgs['test']
    train_set = dataset(config)
    val_set = dataset(val_cfg,mode='val')
    trainval_set = dataset(trainval_cfg,mode='val')
    test_set = dataset(test_cfg,mode='test')
    train_bs = config.bs if args.bs is None else args.bs
    val_bs = val_cfg.bs if (args.bs is None) or (args.mode=='train') else args.bs   
    train_loader = DataLoader(train_set,batch_size=train_bs,shuffle=True,pin_memory=False,collate_fn=train_set.collate_fn)
    val_loader = DataLoader(val_set,batch_size=val_bs,shuffle=False,pin_memory=False,collate_fn=val_set.collate_fn)
    trainval_loader = DataLoader(trainval_set,batch_size=val_bs,shuffle=False,pin_memory=False,collate_fn=val_set.collate_fn)
    test_loader = DataLoader(test_set,batch_size=val_bs,shuffle=False,pin_memory=False,collate_fn=test_set.collate_fn)
    datasets = {'train':train_loader,'val':val_loader,'trainval':trainval_loader,'test':test_loader}
    config.exp_name = args.exp
    config.device = torch.device("cuda")
    torch.cuda.empty_cache()
    #for reproducity
    torch.manual_seed(2333)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #network
    if args.anchors:
        print('calculating new anchors')
        config.anchors,_ = cal_anchors(config.size)
        print(config.anchors)
    network = NetAPI(config,args.net,args.loss)
    #network_ = NetAPI(config,'yoloo',args.loss)
    torch.cuda.empty_cache()
    det = Trainer(config,datasets,network,(args.resume,args.epochs))#,network_)
    if args.mode=='val':
        #metrics = det.validate(det.start-1,mode='val')        
        #det.logger.write_metrics(det.start-1,metrics,[])
        metrics = det.validate(det.start-1,mode='train',save=True)
        print(metrics)
        #det.logger.write_metrics(det.start-1,metrics,[],mode='Trainval')
    elif args.mode=='test':
        det.test()
    else:
        det.train()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--resume", type=str, default=None, help="start from epoch?")
    parser.add_argument("--exp",type=str,default='exp',help="name of exp")
    parser.add_argument("--res",type=int,default=50,help="resnet depth")
    parser.add_argument("--mode",type=str,default='train',help="only validation")
    parser.add_argument("--loss",type=str,default='yolo',help="loss type")
    parser.add_argument("--net",type=str,default='yolo',help="network type:yolo")
    parser.add_argument("--bs",type=int,default=None,help="batchsize")
    parser.add_argument("--anchors",action='store_true')
    args = parser.parse_args()
    cfgs = {}
    
    #Generate config for different dataset
    cfgs['train'] = cfg()
    cfgs['trainval'] = cfg('trainval')
    cfgs['val'] = cfg('val')
    cfgs['test'] = cfg('test')
    main(args,cfgs)
    
    

    