import torch.nn as nn
import torch
import numpy as np

from .utils import iou_wo_center,generalized_iou,build_targets
#Functional Utils
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()
def dice_loss1d(pd,gt,threshold=0.5):
    assert pd.shape == gt.shape
    if gt.shape[0]==0:
        return 0
    inter = torch.sum(pd*gt)
    pd_area = torch.sum(torch.pow(pd,2))
    gt_area = torch.sum(torch.pow(gt,2))
    dice = (2*inter+1)/(pd_area+gt_area+1)
    #fix nans
    dice[dice != dice] = dice.new_tensor([1.0])
    return 1-dice.mean()
class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, cfg):
        super(YOLOLayer, self).__init__()
        anchors = [cfg.anchors[i] for i in cfg.anchor_ind]
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = cfg.cls_num
        self.ignore_thres = cfg.ignore_threshold
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = cfg.obj_scale
        self.noobj_scale = cfg.noobj_scale
        self.img_dim = cfg.size
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w , a_h) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None,infer=False):

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        self.img_dim = img_dim[0]
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.softmax(prediction[..., 5:],dim=-1)  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = (x.data + self.grid_x)/self.grid_size
        pred_boxes[..., 1] = (y.data + self.grid_y)/self.grid_size
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4),
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )
        if infer:
            return output
        else:
            class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            res={}
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            res['wh']= loss_w.item() + loss_h.item()
            res['xy']= loss_x.item() + loss_y.item()
            res['obj'] = self.obj_scale *loss_conf_obj.item()
            res['conf'] = loss_conf.item()
            res['cls'] = loss_cls.item()
            res['all'] = total_loss.item()

            return res,total_loss
def dice_loss(pd,gt,threshold=0.5):
    dims = tuple(range(len(pd.shape)))
    inter = torch.sum(pd*gt,dim=dims)
    pd_area = torch.sum(torch.pow(pd,2),dim=dims)
    gt_area = torch.sum(torch.pow(gt,2),dim=dims)
    dice = (2*inter+1)/(pd_area+gt_area+1)
    #fix nans
    dice[dice != dice] = dice.new_tensor([1.0])
    return 1-dice.mean()

def make_grid_mesh(grid_size,device='cuda'):
    x = np.arange(0,grid_size,1)
    y = np.arange(0,grid_size,1)
    grid_x,grid_y = np.meshgrid(x,y)
    grid_x = torch.tensor(grid_x).view(1,1,grid_size,grid_size).to(dtype=torch.float,device=device)
    grid_y = torch.tensor(grid_y).view(1,1,grid_size,grid_size).to(dtype=torch.float,device=device)
    return grid_x,grid_y
def make_grid_mesh_xy(grid_size,device='cuda'):
    x = np.arange(0,grid_size[1],1)
    y = np.arange(0,grid_size[0],1)
    grid_x,grid_y = np.meshgrid(x,y)
    grid_x = torch.tensor(grid_x).to(dtype=torch.float,device=device)
    grid_y = torch.tensor(grid_y).to(dtype=torch.float,device=device)
    return grid_x,grid_y


### Anchor based
class YOLOLoss(nn.Module):
    def __init__(self,cfg=None):
        super(YOLOLoss,self).__init__()
        self.object_scale = cfg.obj_scale
        self.noobject_scale = cfg.noobj_scale
        self.cls_num = cfg.cls_num
        self.ignore_threshold = cfg.ignore_threshold
        self.device= 'cuda'
        self.target_num = 120
        anchors = [cfg.anchors[i] for i in cfg.anchor_ind]
        self.num_anchors = len(anchors)
        self.anchors = np.array(anchors).reshape(-1,2)
        self.channel_num = self.num_anchors*(self.cls_num+5)
        self.match_threshold = cfg.match_threshold
        self.cls_scale = cfg.cls_scale
        self.reg_scale = cfg.reg_scale
    def build_target(self,pds,gts):
        self.device ='cuda' if pds.is_cuda else 'cpu'
        nB,nA,nH,nW,_ = pds.shape
        assert nH==nW
        nC = self.cls_num
        #threshold = th
        nGts = len(gts)
        obj_mask = torch.zeros(nB,nA,nH,nW,dtype=torch.bool,device=self.device)
        noobj_mask = torch.ones(nB,nA,nH,nW,dtype=torch.bool,device=self.device)
        tbboxes = torch.zeros(nB,nA,nH,nW,4,dtype=torch.float,device=self.device)  
        tcls = torch.zeros(nB,nA,nH,nW,nC,dtype=torch.float,device=self.device) 
        if nGts==0:
            return obj_mask,noobj_mask,tbboxes,tcls,obj_mask.float()
        #convert target
        gt_boxes = gts[:,2:6]
        gws = gt_boxes[:,2]
        ghs = gt_boxes[:,3]

        ious = torch.stack([iou_wo_center(gws,ghs,w,h) for (w,h) in self.scaled_anchors])
        vals, best_n = ious.max(0)
        ind = torch.arange(vals.shape[0],device=self.device)
        '''ind = torch.argsort(vals)
        # so that obj with bigger iou will cover the smaller one 
        # useful for crowed scenes
        idx = torch.argsort(gts[ind,-1],descending=True)#sort as match num,then gt has not matched will be matched first
        ind = ind[idx]
        #discard the gts below the match threshold and has been matched
        best_n =best_n[ind]
        gts = gts[ind,:]
        gt_boxes = gt_boxes[ind,:]
        ious = ious[:,ind]
        '''
        
        batch = gts[:,0].long()
        labels = gts[:,1].long()
        gxs,gys = gt_boxes[:,0]*nW,gt_boxes[:,1]*nH
        gis,gjs = gxs.long(),gys.long()
        #calculate bbox ious with anchors      
        obj_mask[batch,best_n,gjs,gis] = 1
        noobj_mask[batch,best_n,gjs,gis] = 0
        ious = ious.t()
        #ignore big overlap but not the best
        for i,iou in enumerate(ious):
            noobj_mask[batch[i],iou > self.ignore_threshold,gjs[i],gis[i]] = 0

        selected = torch.zeros_like(obj_mask,dtype=torch.long).fill_(-1)
        
        tbboxes[batch,best_n,gjs,gis] = gt_boxes
        tcls[batch,best_n,gjs,gis,labels] = 1
        selected[batch,best_n,gjs,gis] = ind
        

        
        selected = torch.unique(selected[selected>=0])
        gts[selected,-1] += 1 #marked as matched

        
        return obj_mask,noobj_mask,tbboxes,tcls,obj_mask.float()
    
    def get_pds_and_targets(self,pred,infer=False,gts=None):
        grid_x,grid_y = make_grid_mesh_xy(self.grid_size,self.device)
        xs = torch.sigmoid(pred[...,0])#dxs
        ys = torch.sigmoid(pred[...,1])#dys
        ws = pred[...,2]
        hs = pred[...,3]
        conf = torch.sigmoid(pred[...,4])#Object score
        cls_score = torch.softmax(pred[..., 5:],dim=-1) 
        #grid,anchors
        

        pd_bboxes = torch.zeros_like(pred[...,:4],dtype=torch.float,device=self.device)
        pd_bboxes[...,0] = (xs + grid_x)/self.grid_size[1]
        pd_bboxes[...,1] = (ys + grid_y)/self.grid_size[0]
        pd_bboxes[...,2] = torch.exp(ws)*self.anchors_w
        pd_bboxes[...,3] = torch.exp(hs)*self.anchors_h
        nb = pred.shape[0]       
        if infer:   
            return torch.cat((pd_bboxes.view(nb,-1,4),conf.view(nb,-1,1),cls_score.view(nb,-1,self.cls_num)),dim=-1)
        else:
            pds_bbox = (xs,ys,ws,hs,pd_bboxes)
            obj_mask,noobj_mask,tbboxes,tcls,tconf = self.build_target(pd_bboxes,gts)
            tobj = (noobj_mask,tconf)
            return (pds_bbox,conf,cls_score),obj_mask,tbboxes,tobj,tcls
    
    def cal_bbox_loss(self,pds,tbboxes,obj_mask,res):
        xs,ys,ws,hs,_= pds
        txs,tys,tws,ths = tbboxes.permute(4,0,1,2,3).contiguous()
        tws /= self.anchors_w
        ths /= self.anchors_h
        loss_x = mse_loss(xs[obj_mask],txs[obj_mask]-txs[obj_mask].floor())
        loss_y = mse_loss(ys[obj_mask],tys[obj_mask]-tys[obj_mask].floor())
        loss_xy = loss_x + loss_y

        loss_w = mse_loss(ws[obj_mask],torch.log(tws[obj_mask]+1e-16))
        loss_h = mse_loss(hs[obj_mask],torch.log(ths[obj_mask]+1e-16))
        loss_wh = loss_w + loss_h
        res['wh']=loss_wh.item()
        res['xy']=loss_xy.item()
        loss_bbox = loss_xy+loss_wh #mse_loss(pd_bboxes[obj_mask],tbboxes[obj_mask])
        if torch.isnan(loss_bbox):
            print("why??????????")
            exit()
        return loss_bbox,res
    
    def cal_cls_loss(self,pds,target,obj_mask,res):
        loss_cls = bce_loss(pds[obj_mask],target[obj_mask])
        res['cls'] = loss_cls.item()
        return loss_cls,res
    
    def cal_obj_loss(self,pds,target,obj_mask,res):
        noobj_mask,tconf = target
        loss_conf_obj = bce_loss(pds[obj_mask],tconf[obj_mask])
        loss_conf_noobj = bce_loss(pds[noobj_mask],tconf[noobj_mask])
        loss_conf = self.noobject_scale*loss_conf_noobj+self.object_scale*loss_conf_obj
        res['obj'] = loss_conf_obj.item()
        res['conf'] = loss_conf.item()
        return loss_conf,res
    
    def forward(self,out,gts=None,size=None,infer=False):
        nb,_,nh,nw = out.shape
        self.device ='cuda' if out.is_cuda else 'cpu'
        self.grid_size = (nh,nw)
        self.stride = (size[0]/nh,size[1]/nw)
        pred = out.view(nb,self.num_anchors,self.cls_num+5,nh,nw).permute(0,1,3,4,2).contiguous()
        #reshape to nB,nA,nH,nW,bboxes
        self.scaled_anchors = torch.tensor([(a_w, a_h) for a_w, a_h in self.anchors],dtype=torch.float,device=self.device)
        self.anchors_w = (self.scaled_anchors[:,0]).reshape((1, self.num_anchors, 1, 1))
        self.anchors_h = (self.scaled_anchors[:,1]).reshape((1, self.num_anchors, 1, 1))       
        
        if infer:
            return self.get_pds_and_targets(pred,infer)
        else:
            pds,obj_mask,tbboxes,tobj,tcls = self.get_pds_and_targets(pred,infer,gts)
        pds_bbox,pds_obj,pds_cls = pds
        loss_obj,res = self.cal_obj_loss(pds_obj,tobj,obj_mask,{})  
        nm = obj_mask.float().sum()                   
        if nm>0:
            loss_reg,res = self.cal_bbox_loss(pds_bbox,tbboxes,obj_mask,res)
            loss_cls,res = self.cal_cls_loss(pds_cls,tcls,obj_mask,res)
            total = nm*self.reg_scale*loss_reg+loss_obj+self.cls_scale*loss_cls*nm
        else:
            total = loss_obj
        res['all'] = total.item()
        return res,total
class YOLOLoss_iou(YOLOLoss):
    def cal_bbox_loss(self,pds,tbboxes,obj_mask,res):
        pd_bboxes = pds[-1]
        if obj_mask.float().max()>0:#avoid no gt_objs
            ious,gous = generalized_iou(pd_bboxes[obj_mask],tbboxes[obj_mask])
            loss_iou = 1 - ious.mean()
            loss_gou = 1 - gous.mean()
        else:
            loss_iou = torch.tensor(0.0,dtype=torch.float,device=self.device)
            loss_gou = torch.tensor(0.0,dtype=torch.float,device=self.device)
        res['iou'] = loss_iou.item()
        res['gou'] = loss_gou.item()
        return loss_iou,res
class YOLOLoss_gou(YOLOLoss):
    def cal_bbox_loss(self,pds,tbboxes,obj_mask,res):
        pd_bboxes = pds[-1]
        if obj_mask.float().max()>0:#avoid no gt_objs
            ious,gious = generalized_iou(pd_bboxes[obj_mask],tbboxes[obj_mask])
            loss_iou = 1 - ious.mean()
            loss_giou = 1 - gious.mean()
        else:
            loss_iou = torch.tensor(0.0,dtype=torch.float,device=self.device)
            loss_giou = torch.tensor(0.0,dtype=torch.float,device=self.device)
        res['iou'] = loss_iou.item()
        res['giou'] = loss_giou.item()
        return loss_giou,res
class YOLOLoss_com(YOLOLoss):
    def cal_bbox_loss(self,pds,tbboxes,obj_mask,res):
        xs,ys,ws,hs,pd_bboxes = pds
        txs,tys,tws,ths = tbboxes.permute(4,0,1,2,3).contiguous()
        tws /= self.anchors_w
        ths /= self.anchors_h
        loss_x = mse_loss(xs[obj_mask],txs[obj_mask]-txs[obj_mask].floor())
        loss_y = mse_loss(ys[obj_mask],tys[obj_mask]-tys[obj_mask].floor())
        loss_xy = loss_x + loss_y

        loss_w = mse_loss(ws[obj_mask],torch.log(tws[obj_mask]+1e-16))
        loss_h = mse_loss(hs[obj_mask],torch.log(ths[obj_mask]+1e-16))
        loss_wh = loss_w + loss_h
        res['wh']=loss_wh.item()
        res['xy']=loss_xy.item()
        loss_bbox = loss_xy+loss_wh 
        if torch.isnan(loss_bbox):
            exit()
        if obj_mask.float().max()>0:#avoid no gt_objs
            ious,gous = generalized_iou(pd_bboxes[obj_mask],tbboxes[obj_mask])
            loss_iou = 1 - ious.mean()
            loss_gou = 1 - gous.mean()
        else:
            loss_iou = torch.tensor(0.0,dtype=torch.float,device=self.device)
            loss_gou = torch.tensor(0.0,dtype=torch.float,device=self.device)
        res['iou'] = loss_iou.item()
        res['gou'] = loss_gou.item()
        return loss_gou+loss_bbox,res

class LossAPI(nn.Module):
    def __init__(self,cfg,loss):
        super(LossAPI,self).__init__()
        self.bbox_losses = cfg.anchor_divide.copy()
        self.bbox_losses_ = cfg.anchor_divide.copy()
        self.not_match = 0
        for i,ind in enumerate(cfg.anchor_divide):
            cfg.anchor_ind = ind
            self.bbox_losses[i] = Losses[loss](cfg)
            self.bbox_losses_[i] = Losses['yoloo'](cfg)

    def forward(self,outs,gt=None,size=None,infer=False):
        if infer:
            res = []
            for out,loss in zip(outs,self.bbox_losses):#,self.bbox_losses_):
                result = loss(out,gt,size,infer=True)
                res.append(result)
                '''result_ = loss_(out,gt,size,infer=True)
                print(result)
                print()
                print(result_)'''
                
            return torch.cat(res,dim=1)
        else:
            res ={'xy':0.0,'wh':0.0,'conf':0.0,'cls':0.0,'obj':0.0,'all':0.0,'iou':0.0,'giou':0.0}
            totals = []
            match =torch.zeros((gt.shape[0],1),dtype=torch.float,device=gt.device)
            gt = torch.cat((gt,match),-1)
            for out,loss in zip(outs,self.bbox_losses): 
                ret,total = loss(out,gt,size)
                for k in ret:
                    res[k] += ret[k]/len(self.bbox_losses)
                totals.append(total)
                '''print(ret)    
            print()
            for out,loss in zip(outs,self.bbox_losses_): 
                ret,total = loss(out,gt,size)
                print(ret)'''
            not_match = int((gt[:,-1]==0).sum())
            if not_match>0:
                self.not_match += not_match
            return res,torch.stack(totals).sum()
    def reset_notmatch(self):
        self.not_match = 0
Losses = {'yolo':YOLOLoss,'yoloo':YOLOLayer,'yolo_iou':YOLOLoss_iou,'yolo_gou':YOLOLoss_gou,'yolo_com':YOLOLoss_com}



        







        