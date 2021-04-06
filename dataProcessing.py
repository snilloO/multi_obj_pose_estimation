import torch.utils.data as data
import torch
import json
import numpy as np
import random
import cv2
import os
from torch.nn import functional as F

ls = 1 #0 for only bboxes,1 for labels and bboxes
#stack functions for collate_fn
#Notice: all dicts need have same keys and all lists should have same length
def stack_dicts(dicts):
    if len(dicts)==0:
        return None
    res = {}
    for k in dicts[0].keys():
        res[k] = [obj[k] for obj in dicts]
    return res

def stack_list(lists):
    if len(lists)==0:
        return None
    res = list(range(len(lists[0])))
    for k in range(len(lists[0])):
        res[k] = torch.stack([obj[k] for obj in lists])
    return res
def rand(item):
    try:
        tmp=[]
        for i in item:
            tmp.append(random.uniform(-i,i))
    except:
        if random.random()<0.5:
            return random.uniform(-i,i)
        else:
            return 0
    finally:
        return tuple(tmp)   
def get_croppable_part(labels,size):
    h,w = size
    min_x = torch.min(labels[:,ls]-labels[:,ls+2]/2)
    min_y = torch.min(labels[:,ls+1]-labels[:,ls+3]/2)
    max_x = torch.max(labels[:,ls]+labels[:,ls+2]/2)
    max_y = torch.max(labels[:,ls+1]+labels[:,ls+3]/2)
    return (max(min_x,0),max(0,min_y),min(w-1,max_x),min(h-1,max_y))
def valid_scale(src,vs):
    img = cv2.cvtColor(src,cv2.COLOR_RGB2HSV).astype(np.float)
    img[:,:,2] *= (1+vs)
    img[:,:,2][img[:,:,2]>255] = 255
    img = cv2.cvtColor(img.astype(np.int8),cv2.COLOR_HSV2RGB).astype(np.float)
    return img
def resize(src,tsize):
    dst = cv2.resize(src,(tsize[1],tsize[0]),interpolation=cv2.INTER_LINEAR)
    return dst
def translate(src,labels,trans):
    h,w,_ = src.shape
    if labels.shape[0]>0:
        mx,my,mxx,mxy = get_croppable_part(labels,(h,w))
        tx = random.uniform(-min(mx,w*trans),min(w*trans,w-mxx-1))
        ty = random.uniform(-min(my,h*trans),min(h*trans,h-mxy-1))
    else:
        tx = random.uniform(-w*trans,w*trans)
        ty = random.uniform(-h*trans,h*trans)
    mat = np.array([[1,0,tx],[0,1,ty]])
    dst = cv2.warpAffine(src,mat,(w,h))
    labels[:,ls] += tx
    labels[:,ls+1] += ty
    
    return dst,labels
def crop(src,labels,crop):
    h,w,_ = src.shape
    if ((w<10)or(h<10)):
        return src,labels
    if labels.shape[0]>0:
        mx,my,mxx,mxy = get_croppable_part(labels,(h,w))
        txm = int(random.uniform(0,min(mx,w*crop)))
        tym = int(random.uniform(0,min(my,h*crop)))
        txmx = int(random.uniform(max(mxx,w*(1-crop)),w))
        tymx = int(random.uniform(max(mxy,h*(1-crop)),h))
        labels[:,ls] -= txm
        labels[:,ls+1] -= tym
    else:
        txm = int(random.uniform(0,w*crop))
        tym = int(random.uniform(0,h*crop))
        txmx = int(random.uniform(w*(1-crop),w-0.1))
        tymx = int(random.uniform(h*(1-crop),h-0.1))
    dst = src.copy()
    dst = dst[tym:tymx+1,txm:txmx+1,:]
    
    return dst,labels
def rotate(src,labels,ang,scale):
    h,w,_ = src.shape
    center =(w/2,h/2)
    mat = cv2.getRotationMatrix2D(center, ang, scale)
    dst = cv2.warpAffine(src,mat,(w,h))
    labels_ = labels.clone()
    if labels.shape[0]>0:
        xs,ys,ws,hs = labels[:,ls:].t()
        n = len(xs)
        sx = abs(mat[0,0])
        sy = abs(mat[0,1])
        pts = np.stack([xs,ys,np.ones([n])],axis=1).T
        tpts = torch.tensor(np.dot(mat,pts).T,dtype=torch.float)
        labels_[:,ls] = tpts[:,0]
        labels_[:,ls+1] = tpts[:,1]
        labels_[:,ls+2] = (sx*ws + sy*hs)*scale
        labels_[:,ls+3] = (sx*hs + sy*ws)*scale
        mask = (tpts[:,0]>0)&(tpts[:,0]<w)&(tpts[:,1]>0)&(tpts[:,1]<h)
        labels_ = labels_[mask,:]
    return dst,labels_
def flip(src,labels):
    w = src.shape[1]
    dst = cv2.flip(src,1)
    labels[:,ls] = w-1-labels[:,ls]
    return dst,labels
def color_normalize(img,mean):
    img = img.astype(np.float)
    if img.max()>1:
        img /= 255
    img -= np.array(mean)/255
    return img

class VOC_dataset(data.Dataset):
    def __init__(self,cfg,mode='train'):
        self.img_path = cfg.img_path
        self.cfg = cfg
        data = json.load(open(cfg.file,'r'))
        self.imgs = list(data.keys())
        self.annos = data
        self.mode = mode
        self.accm_batch = 0
        self.size = cfg.size
        self.aug = cfg.augment
    def __len__(self):
        return len(self.imgs)

    def img_to_tensor(self,img):
        data = torch.tensor(np.transpose(img,[2,0,1]),dtype=torch.float)
        if data.max()>1:
            data /= 255.0
        return data
    def gen_gts(self,anno):
        gts = torch.zeros((anno['obj_num'],ls+4),dtype=torch.float)
        if anno['obj_num'] == 0:
            return gts
        labels = torch.tensor(anno['labels'])[:,:ls+4]
        assert labels.shape[-1] == ls+4
        gts[:,0] = labels[:,0]
        gts[:,ls] = (labels[:,ls]+labels[:,ls+2])/2
        gts[:,ls+1] = (labels[:,ls+1]+labels[:,ls+3])/2
        gts[:,ls+2] = (labels[:,ls+2]-labels[:,ls])
        gts[:,ls+3] = (labels[:,ls+3]-labels[:,ls+1])
        return gts
        
    def normalize_gts(self,labels,size):
        #transfer
        if len(labels)== 0:
            return labels
        labels[:,ls:]/=size 
        return labels

    def pad_to_square(self,img):
        h,w,_= img.shape
        ts = max(h,w)
        diff1 = abs(h-ts)
        diff2 = abs(w-ts)
        pad = (diff1//2,diff2//2,diff1-diff1//2,diff2-diff2//2)
        img = cv2.copyMakeBorder(img,pad[0],pad[2],pad[1],pad[3],cv2.BORDER_CONSTANT,0)
        return img,(pad[0],pad[1])

    def __getitem__(self,idx):
        name = self.imgs[idx]
        anno = self.annos[name]
        img = cv2.imread(os.path.join(self.img_path,name+'.jpg'))
        ##print(img.shape)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h,w = img.shape[:2]        
        labels = self.gen_gts(anno)
        #print(name)
        if self.mode=='train':
            aug = []
            if self.aug:
                if (random.randint(0,1)==1) and self.cfg.flip:
                    img,labels = flip(img,labels)
                    aug.append('flip')
                if (random.randint(0,1)==1) and self.cfg.trans:
                    img,labels = translate(img,labels,self.cfg.trans)
                    aug.append('trans')
                if (random.randint(0,1)==1) and self.cfg.crop:
                    img,labels = crop(img,labels,self.cfg.crop)
                    aug.append('crop')
                if (random.randint(0,1)==1) and self.cfg.rot:
                    ang = random.uniform(-self.cfg.rot,self.cfg.rot)
                    scale = random.uniform(1-self.cfg.scale,1+self.cfg.scale)
                    img,labels = rotate(img,labels,ang,scale)
                    aug.append('rot')
            img,pad = self.pad_to_square(img)
            size = img.shape[0]
            labels[:,ls]+= pad[1]
            labels[:,ls+1]+= pad[0]
            data = self.img_to_tensor(img)
            labels = self.normalize_gts(labels,size)
            return data,labels      
        else:
            #validation set
            img,pad = self.pad_to_square(img)
            img = resize(img,(self.cfg.size,self.cfg.size))
            data = self.img_to_tensor(img)
            info ={'size':(h,w),'img_id':name,'pad':pad}
            if self.mode=='val':
                return data,labels,info
            else:
                return data,info
    def collate_fn(self,batch):
        if self.mode=='test':
            data,info = list(zip(*batch))
            data = torch.stack(data)
            info = stack_dicts(info)
            return data,info 
        elif self.mode=='val':
            data,labels,info = list(zip(*batch))
            info = stack_dicts(info)
            data = torch.stack(data)
        elif self.mode=='train':
            data,labels = list(zip(*batch))
            if (self.accm_batch % 10 == 0)and (self.aug):
                self.size = random.choice(self.cfg.sizes)
            tsize = (self.size,self.size)
            self.accm_batch += 1
            data = torch.stack([F.interpolate(img.unsqueeze(0),tsize,mode='bilinear').squeeze(0) for img in data]) #multi-scale-training   
        tmp =[]
                   
                
        for i,bboxes in enumerate(labels):
            if len(bboxes)>0:
                label = torch.zeros(len(bboxes),ls+5)
                label[:,1:] = bboxes
                label[:,0] = i
                tmp.append(label)
        if len(tmp)>0:
            labels = torch.cat(tmp,dim=0)
            labels = labels.reshape(-1,ls+5)
        else:
            labels = torch.tensor(tmp,dtype=torch.float).reshape(-1,ls+5)
        if self.mode=='train':
            return data,labels
        else:
            return data,labels,info

                





