import torch
import torch.nn as nn
import time
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os
import json
import random

from utils import Logger,ap_per_class
from utils import non_maximum_supression_soft as nms
from utils import cal_tp_per_item as cal_tp
tosave = ['mAP']
plot = [0.5,0.75] 
thresholds = np.around(np.arange(0.5,0.76,0.05),2)

class Trainer:
    def __init__(self,cfg,datasets,net,epoch,cmp_net=None):
        self.cfg = cfg
        if 'train' in datasets:
            self.trainset = datasets['train']
        if 'val' in datasets:
            self.valset = datasets['val']
        if 'trainval' in datasets:
            self.trainval = datasets['trainval']
        else:
            self.trainval = False
        if 'test' in datasets:
            self.testset = datasets['test']
        self.net = net

        name = cfg.exp_name
        self.name = name
        self.checkpoints = os.path.join(cfg.checkpoint,name)

        self.device = cfg.device
        self.net_ = cmp_net

        self.optimizer = optim.Adam(self.net.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
        self.lr_sheudler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min', factor=cfg.lr_factor, threshold=0.0001,patience=cfg.patience,min_lr=cfg.min_lr)
        
        if not(os.path.exists(self.checkpoints)):
            os.mkdir(self.checkpoints)
        self.predictions = os.path.join(self.checkpoints,'pred')
        if not(os.path.exists(self.predictions)):
            os.mkdir(self.predictions)

        start,total = epoch
        self.start = start        
        self.total = total
        log_dir = os.path.join(self.checkpoints,'logs')
        if not(os.path.exists(log_dir)):
            os.mkdir(log_dir)
        self.logger = Logger(log_dir)
        torch.cuda.empty_cache()
        self.save_every_k_epoch = cfg.save_every_k_epoch #-1 for not save and validate
        self.val_every_k_epoch = cfg.val_every_k_epoch
        self.upadte_grad_every_k_batch = 1

        self.best_mAP = 0
        self.best_mAP_epoch = 0
        self.movingLoss = 0
        self.bestMovingLoss = 10000
        self.bestMovingLossEpoch = 1e9

        self.early_stop_epochs = 50
        self.alpha = 0.95 #for update moving loss
        self.lr_change= cfg.adjust_lr
        self.base_epochs = cfg.base_epochs


        self.nms_threshold = cfg.nms_threshold
        self.conf_threshold = cfg.dc_threshold
        self.save_pred = False
        
        #load from epoch if required
        if start:
            if (start=='-1')or(start==-1):
                self.load_last_epoch()
            else:
                self.load_epoch(start)
        else:
            self.start = 0
        self.net = self.net.to(self.device)
        if not (cmp_net is None):
            self.net_ = self.net_.to(self.device)

    def load_last_epoch(self):
        files = os.listdir(self.checkpoints)
        idx = 0
        for name in files:
            if name[-3:]=='.pt':
                epoch = name[6:-3]
                if epoch=='best' or epoch=='bestm':
                  continue
                idx = max(idx,int(epoch))
        if idx==0:
            exit()
        else:
            self.load_epoch(str(idx))
    def save_epoch(self,idx,epoch):
        saveDict = {'net':self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler':self.lr_sheudler.state_dict(),
                    'epoch':epoch,
                    'mAP':self.best_mAP,
                    'mAP_epoch':self.best_mAP_epoch,
                    'movingLoss':self.movingLoss,
                    'bestmovingLoss':self.bestMovingLoss,
                    'bestmovingLossEpoch':self.bestMovingLossEpoch}
        path = os.path.join(self.checkpoints,'epoch_'+idx+'.pt')
        torch.save(saveDict,path)                  
    def load_epoch(self,idx):
        model_path = os.path.join(self.checkpoints,'epoch_'+idx+'.pt')
        if os.path.exists(model_path):
            print('load:'+model_path)
            info = torch.load(model_path)
            self.net.load_state_dict(info['net'])
            if not(self.lr_change):
                self.optimizer.load_state_dict(info['optimizer'])#might have bugs about device
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
                self.lr_sheudler.load_state_dict(info['lr_scheduler'])
            self.start = info['epoch']+1
            self.best_mAP = info['mAP']
            self.best_mAP_epoch = info['mAP_epoch']
            self.movingLoss = info['movingLoss']
            self.bestMovingLoss = info['bestmovingLoss']
            self.bestMovingLossEpoch = info['bestmovingLossEpoch']
        else:
            print('no such model at:',model_path)
            exit()
    def _updateRunningLoss(self,loss,epoch):
        if self.bestMovingLoss>loss:
            self.bestMovingLoss = loss
            self.bestMovingLossEpoch = epoch
            self.save_epoch('bestm',epoch)
    def logMemoryUsage(self, additionalString=""):
        if torch.cuda.is_available():
            print(additionalString + "Memory {:.0f}Mb max, {:.0f}Mb current".format(
                torch.cuda.max_memory_allocated() / 1024 / 1024, torch.cuda.memory_allocated() / 1024 / 1024))
    def set_lr(self,lr):
        #adjust learning rate manually
        for param_group in self.optimizer.param_groups:
            param_group['lr']=lr
        #tbi:might set different lr to different kind of parameters
    def adjust_lr(self,lr_factor):
        #adjust learning rate manually
        for param_group in self.optimizer.param_groups:
            param_group['lr']*=lr_factor
    def warm_up(self,epoch):
        if len(self.base_epochs)==0:
            return False
        if epoch <= self.base_epochs[-1]:
            if epoch in self.base_epochs:
                self.adjust_lr(0.1)
            return True
        else:
            return False
    def train_one_epoch(self):
        self.optimizer.zero_grad()
        running_loss ={'xy':0.0,'wh':0.0,'conf':0.0,'cls':0.0,'obj':0.0,'all':0.0,'iou':0.0,'giou':0.0}
        self.net.train()
        n = len(self.trainset)
        for data in tqdm(self.trainset):
            inputs,labels = data
            labels = labels.to(self.device).float()
            display,loss = self.net(inputs.to(self.device).float(),gts=labels)
            #display,loss = self.net_(inputs.to(self.device).float(),gts=labels)
            #exit()            
            del inputs,labels
            for k in running_loss:
                if k in display.keys():
                    if np.isnan(display[k]):
                        continue
                    running_loss[k] += display[k]/n
            loss.backward()
            #solve gradient explosion problem caused by large learning rate or small batch size
            #nn.utils.clip_grad_value_(self.net.parameters(), clip_value=2.0) 
            #nn.utils.clip_grad_norm_(self.net.parameters(),max_norm=2.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            del loss
        self.logMemoryUsage()
        print(f'#Gt not matched:{self.net.loss.not_match}')
        self.net.loss.reset_notmatch()
        return running_loss
    def train(self):
        print("strat train:",self.name)
        print("start from epoch:",self.start)
        print("=============================")
        self.optimizer.zero_grad()
        print(self.optimizer.param_groups[0]['lr'])
        epoch = self.start
        stop_epochs = 0
        #torch.autograd.set_detect_anomaly(True)
        while epoch < self.total and stop_epochs<self.early_stop_epochs:
            running_loss = self.train_one_epoch()            
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.write_loss(epoch,running_loss,lr)
            #step lr
            self._updateRunningLoss(running_loss['all'],epoch)
            if not self.warm_up(epoch):
                self.lr_sheudler.step(running_loss['all'])
            lr_ = self.optimizer.param_groups[0]['lr']
            if lr_ == self.cfg.min_lr:
                stop_epochs +=1
            if (epoch+1)%self.save_every_k_epoch==0:
                self.save_epoch(str(epoch),epoch)
            if (epoch+1)%self.val_every_k_epoch==0:                
                metrics = self.validate(epoch,'val',self.save_pred)
                self.logger.write_metrics(epoch,metrics,tosave)
                mAP = metrics['mAP']
                if mAP >= self.best_mAP:
                    self.best_mAP = mAP
                    self.best_mAP_epoch = epoch
                    print("best so far, saving......")
                    self.save_epoch('best',epoch)
                if self.trainval:
                    metrics = self.validate(epoch,'train',self.save_pred)
                    self.logger.write_metrics(epoch,metrics,tosave,mode='Trainval')
                    mAP = metrics['mAP']
            print(f"best so far with {self.best_mAP} at epoch:{self.best_mAP_epoch}")
            epoch +=1
                
        print("Best mAP: {:.4f} at epoch {}".format(self.best_mAP, self.best_mAP_epoch))
        self.save_epoch(str(epoch-1),epoch-1)
    def validate(self,epoch,mode,save=False):
        self.net.eval()
        res = {}
        print('start Validation Epoch:',epoch)
        if mode=='val':
            valset = self.valset
        else:
            valset = self.trainval
        with torch.no_grad():
            mAP = 0
            count = 0
            batch_metrics={}
            for th in thresholds:
                batch_metrics[th] = []
            gt_labels = []
            pd_num = 0
            for data in tqdm(valset):
                inputs,labels,info = data
                pds = self.net(inputs.to(self.device).float())
                nB = pds.shape[0]
                gt_labels += labels[:,1].tolist()               
                for b in range(nB):
                    pred = pds[b].view(-1,self.cfg.cls_num+5)
                    name = info['img_id'][b]
                    size = info['size'][b]
                    pad = info['pad'][b]
                    pred[:,:4] *= max(size)
                    pred[:,0] -= pad[1]
                    pred[:,1] -= pad[0]
                    if save:
                        pds_ = list(pred.cpu().numpy().astype(float))
                        pds_ = [list(pd) for pd in pds_]
                        result ={'bboxes':pds_,'pad':pad,'size':size}
                        res[name] = result
                    pred_nms = nms(pred,self.conf_threshold, self.nms_threshold)                    
                    gt = labels[labels[:,0]==b,1:].reshape(-1,5)
                    #pred_nms_ = np.round(pred_nms.cpu().numpy().astype(np.float32),1)
                    #print(pred_nms_)
                    #print(gt)                 
                    pd_num+=pred_nms.shape[0]
                    '''if save:
                        print(pred_nms)
                        print(gt)'''
                    count+=1
                    for th in batch_metrics:
                        batch_metrics[th].append(cal_tp(pred_nms,gt,th))
        metrics = {}
        for th in batch_metrics:
            tps,scores,pd_labels = [np.concatenate(x, 0) for x in list(zip(*batch_metrics[th]))]
            precision, recall, AP,_,_ = ap_per_class(tps, scores, pd_labels, gt_labels)
            mAP += np.mean(AP)
            if th in plot:
                metrics['AP/'+str(th)] = np.mean(AP)
                metrics['Precision/'+str(th)] = np.mean(precision)
                metrics['Recall/'+str(th)] = np.mean(recall)
        metrics['mAP'] = mAP/len(thresholds)
        if save:
            json.dump(res,open(os.path.join(self.predictions,'pred_epoch_'+str(epoch)+'.json'),'w'))
        
        return metrics
    def test(self):
        self.net.eval()
        res = {}
        with torch.no_grad():
            for data in tqdm(self.testset):
                inputs,info = data
                pds = self.net(inputs.to(self.device).float())
                nB = pds.shape[0]
                for b in range(nB):
                    pred = pds[b].view(-1,self.cfg.cls_num+5)
                    name = info['img_id'][b]
                    tsize = info['size'][b]
                    pad = info['pad'][b]
                    pred[:,:4]*= max(tsize)
                    pred[:,0] -= pad[1]
                    pred[:,1] -= pad[0]
                    cls_confs,cls_labels = torch.max(pred[:,5:],dim=1,keepdim=True)
                    pred_nms = torch.cat((pred[:,:5],cls_confs,cls_labels.float()),dim=1)                    
                    #pred_nms = nms(pred,self.conf_threshold, self.nms_threshold)
                    pds_ = list(pred_nms.cpu().numpy().astype(float))
                    pds_ = [list(pd) for pd in pds_]
                    res[name] = pds_
        
        json.dump(res,open(os.path.join(self.predictions,'pred_test.json'),'w'))
    def validate_random(self):
        self.net.eval()
        self.valset.shuffle = True
        bs = self.valset.batch_size
        imgs = list(range(bs))
        preds = list(range(bs))
        gts = list(range(bs))
        sizes = list(range(bs))
        with torch.no_grad():
            inputs,labels,info = next(iter(self.valset))
            pds = self.net(inputs.to(self.device).float())
            for b in range(bs):           
                pred = pds[b].view(-1,self.cfg.cls_num+5)
                pred_nms = nms(pred,self.conf_threshold, self.nms_threshold)
                size = info['size'][b]
                pad = info['pad'][b]                  
                pred_nms[:,:4] *= max(size)
                pred_nms[:,0] -= pad[1]
                pred_nms[:,1] -= pad[0]
                imgs[b] = inputs[b]
                preds[b] = pred_nms
                gts[b] = labels[labels[:,0]==b,1:].reshape(-1,5)
                sizes[b] = size 
        return imgs,preds,gts,sizes

        


                


        




