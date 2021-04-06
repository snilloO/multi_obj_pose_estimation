import numpy as np
import json
import os
import torch
from tqdm import tqdm
import argparse

from utils import non_maximum_supression as nms
from utils import cal_tp_per_item as eval_per_img
from utils import ap_per_class
from config import Config
def gen_gts(anno):
    gts = torch.zeros((anno['obj_num'],5),dtype=torch.float)
    if anno['obj_num'] == 0:
        return gts
    labels = torch.tensor(anno['labels']) #ignore hard
    assert labels.shape[-1] == 6
    gts[:,0] =  labels[:,0]
    gts[:,1] = (labels[:,1]+labels[:,3])/2
    gts[:,2] = (labels[:,2]+labels[:,4])/2
    gts[:,3] = labels[:,3] - labels[:,1]
    gts[:,4] = labels[:,4] - labels[:,2]
    return gts
def main(args):
    print(args.mode)
    cfg = Config(mode=args.mode)
    gts = json.load(open(cfg.file))
    nms_threshold = args.nms_threshold
    conf_threshold = args.conf_threshold
    print(f"nms threshold:{nms_threshold}\nconfidence threshold:{conf_threshold}")
    plot = [0.5,0.75] 
    thresholds = np.around(np.arange(0.5,0.76,0.05),2)
    pds = json.load(open(os.path.join(cfg.checkpoint,args.exp,'pred',args.name+'.json')))
    mAP = 0
    batch_metrics={}
    for th in thresholds:
        batch_metrics[th] = []
    gt_labels = []
    print(len(gts),len(pds))
    for img in tqdm(gts.keys()):
        res = pds[img]
        bboxes = torch.tensor(res['bboxes'])
        gt = gen_gts(gts[img])
        gt_labels += gt[:,0].tolist()  
        pred_nms = nms(bboxes,conf_threshold, nms_threshold)
        for th in batch_metrics:
            batch_metrics[th].append(eval_per_img(pred_nms,gt,th))
    metrics = {}
    for th in batch_metrics:
        tps,scores,pd_labels = [np.concatenate(x, 0) for x in list(zip(*batch_metrics[th]))]
        precision, recall, AP,_,_ = ap_per_class(tps, scores, pd_labels, gt_labels,plot=True)
        mAP += np.mean(AP)
        if th in plot:
            metrics['AP/'+str(th)] = np.mean(AP)
            metrics['Precision/'+str(th)] = np.mean(precision)
            metrics['Recall/'+str(th)] = np.mean(recall)
    metrics['mAP'] = mAP/len(thresholds)
    for k in metrics:
        print(k,':',metrics[k])
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",type=str,default='exp',help="name of exp")
    parser.add_argument("--mode",type=str,default='trainval',help="only validation")
    parser.add_argument("--name",type=str,default='pred_test',help="name of prediction")
    parser.add_argument("-n","--nms_threshold",type=float,default='0.5',help="nms threshod")
    parser.add_argument("-c","--conf_threshold",type=float,default='0.5',help="confidence threshod")
    args = parser.parse_args()
    main(args)