import matplotlib.pyplot as plt 
import math
import torch
import numpy as np
import os 
import json
from tqdm import tqdm

def init_weights(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight.data)
        #print(m)
    elif type(m) == torch.nn.BatchNorm2d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        #print(m)
def iou_wo_center(w1,h1,w2,h2):
    #assuming at the same center
    #return a vector nx1
    inter = torch.min(w1,w2)*torch.min(h1,h2)
    union = w1*h1 + w2*h2 - inter
    ious = inter/union
    ious[ious!=ious] = torch.tensor(0.0,device='cuda') #avoid nans
    return ious
def generalized_iou(bbox1,bbox2):
    #return shape nx1
    bbox1 = bbox1.view(-1,4)
    bbox2 = bbox2.view(-1,4)
    assert bbox1.shape[0]==bbox2.shape[0]
    #tranfer xc,yc,w,h to xmin ymin xmax ymax
    xmin1 = bbox1[:,0] - bbox1[:,2]/2
    xmin2 = bbox2[:,0] - bbox2[:,2]/2
    ymin1 = bbox1[:,1] - bbox1[:,3]/2
    ymin2 = bbox2[:,1] - bbox2[:,3]/2
    xmax1 = bbox1[:,0] + bbox1[:,2]/2
    xmax2 = bbox2[:,0] + bbox2[:,2]/2
    ymax1 = bbox1[:,1] + bbox1[:,3]/2
    ymax2 = bbox2[:,1] + bbox2[:,3]/2

    inter_xmin = torch.max(xmin1,xmin2)
    inter_xmax = torch.min(xmax1,xmax2)
    inter_ymin = torch.max(ymin1,ymin2)
    inter_ymax = torch.min(ymax1,ymax2)
    cover_xmin = torch.min(xmin1,xmin2)
    cover_xmax = torch.max(xmax1,xmax2)
    cover_ymin = torch.min(ymin1,ymin2)
    cover_ymax = torch.max(ymax1,ymax2)

    inter_w = inter_xmax-inter_xmin
    inter_h = inter_ymax-inter_ymin
    mask = ((inter_w>=0 )&( inter_h >=0)).to(torch.float)
    # detect not overlap
    cover = (cover_xmax-cover_xmin)*(cover_ymax-cover_ymin)
    #inter_h[inter_h<0] = 0
    inter = inter_w*inter_h*mask
    #keep iou<0 to avoid gradient diasppear
    area1 = bbox1[:,2]*bbox1[:,3]
    area2 = bbox2[:,2]*bbox2[:,3]
    union = area1+area2 - inter
    ious = inter/union
    gious = ious-(cover-union)/cover
    ious[ious!=ious] = torch.tensor(0.0,device=bbox1.device) #avoid nans
    gious[gious!=gious] = torch.tensor(0.0,device=bbox1.device) #avoid nans
    return ious,gious
def cal_gious_matrix(bbox1,bbox2):
    #return mxn matrix
    bbox1 = bbox1.view(-1,4)
    bbox2 = bbox2.view(-1,4)
    
    #tranfer xc,yc,w,h to xmin ymin xmax ymax
    xmin1 = bbox1[:,0] - bbox1[:,2]/2
    xmin2 = bbox2[:,0] - bbox2[:,2]/2
    ymin1 = bbox1[:,1] - bbox1[:,3]/2
    ymin2 = bbox2[:,1] - bbox2[:,3]/2
    xmax1 = bbox1[:,0] + bbox1[:,2]/2
    xmax2 = bbox2[:,0] + bbox2[:,2]/2
    ymax1 = bbox1[:,1] + bbox1[:,3]/2
    ymax2 = bbox2[:,1] + bbox2[:,3]/2

    inter_xmin = torch.max(xmin1.view(-1,1),xmin2.view(1,-1))
    inter_xmax = torch.min(xmax1.view(-1,1),xmax2.view(1,-1))
    inter_ymin = torch.max(ymin1.view(-1,1),ymin2.view(1,-1))
    inter_ymax = torch.min(ymax1.view(-1,1),ymax2.view(1,-1))
    cover_xmin = torch.min(xmin1.view(-1,1),xmin2.view(1,-1))
    cover_xmax = torch.max(xmax1.view(-1,1),xmax2.view(1,-1))
    cover_ymin = torch.min(ymin1.view(-1,1),ymin2.view(1,-1))
    cover_ymax = torch.max(ymax1.view(-1,1),ymax2.view(1,-1))

    inter_w = inter_xmax-inter_xmin
    inter_h = inter_ymax-inter_ymin
    mask = ((inter_w>=0 )&( inter_h >=0)).to(torch.float)

    # detect not overlap
    cover = (cover_xmax-cover_xmin)*(cover_ymax-cover_ymin)
    #inter_h[inter_h<0] = 0
    inter = inter_w*inter_h*mask
    #keep iou<0 to avoid gradient diasppear
    area1 = bbox1[:,2]*bbox1[:,3]
    area2 = bbox2[:,2]*bbox2[:,3]
    union = area1.view(-1,1)+area2.view(1,-1)
    union -= inter

    ious = inter/union
    gious = iou-(cover-union)/cover
    ious[ious!=ious] = torch.tensor(0.0,device='cuda') #avoid nans
    gous[gous!=gous] = torch.tensor(0.0,device='cuda') #avoid nans 
    return ious,gious
def iou_wt_center(bbox1,bbox2):
    #only for torch, return a vector nx1
    bbox1 = bbox1.view(-1,4)
    bbox2 = bbox2.view(-1,4)
    
    #tranfer xc,yc,w,h to xmin ymin xmax ymax
    xmin1 = bbox1[:,0] - bbox1[:,2]/2
    xmin2 = bbox2[:,0] - bbox2[:,2]/2
    ymin1 = bbox1[:,1] - bbox1[:,3]/2
    ymin2 = bbox2[:,1] - bbox2[:,3]/2
    xmax1 = bbox1[:,0] + bbox1[:,2]/2
    xmax2 = bbox2[:,0] + bbox2[:,2]/2
    ymax1 = bbox1[:,1] + bbox1[:,3]/2
    ymax2 = bbox2[:,1] + bbox2[:,3]/2

    inter_xmin = torch.max(xmin1,xmin2)
    inter_xmax = torch.min(xmax1,xmax2)
    inter_ymin = torch.max(ymin1,ymin2)
    inter_ymax = torch.min(ymax1,ymax2)

    inter_w = inter_xmax-inter_xmin
    inter_h = inter_ymax-inter_ymin
    mask = ((inter_w>=0 )&( inter_h >=0)).to(torch.float)
    
    # detect not overlap
    
    #inter_h[inter_h<0] = 0
    inter = inter_w*inter_h*mask
    #keep iou<0 to avoid gradient diasppear
    area1 = bbox1[:,2]*bbox1[:,3]
    area2 = bbox2[:,2]*bbox2[:,3]
    union = area1+area2 - inter
    ious = inter/union
    ious[ious!=ious] = torch.tensor(0.0,device='cuda')
    return ious
def to_cpu(tensor):
    return tensor.detach().cpu()
def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):

    BoolTensor = torch.cuda.BoolTensor if pred_boxes.is_cuda else torch.BoolTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    # Output tensors
    obj_mask = BoolTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = BoolTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([iou_wo_center(w,h,gwh[:,0],gwh[:,1]) for (w,h) in anchors])
    _, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()

    tconf = obj_mask.float()
    return  class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf