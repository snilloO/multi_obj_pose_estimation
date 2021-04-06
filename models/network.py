import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from .backbone import ResNet,conv1x1,conv3x3,Darknet
from .loss_funcs import  LossAPI
from .models import Darknetv3
from .utils import init_weights
def NetAPI(cfg,net,loss,init=True):
    networks = {'yolo':YOLO,'yolo_spp':YOLO_SPP,'yoloo':Darknetv3}
    network = networks[net](cfg,loss)
    if init:
        network.initialization()
    return network

class NonResidual(nn.Module):
    multiple=2
    def __init__(self,in_channels,channels,stride=1):
        super(NonResidual,self).__init__()
        self.conv1 = conv1x1(in_channels,channels,stride)
        self.relu = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm2d(channels,momentum=0.9, eps=1e-5)
        self.conv2 = conv3x3(channels,channels*NonResidual.multiple)
        self.bn2 = nn.BatchNorm2d(channels*NonResidual.multiple,momentum=0.9, eps=1e-5)
    def forward(self,x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)

        return y
class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x
class YOLO(nn.Module):
    def __init__(self,cfg,loss):
        super(YOLO,self).__init__()
        self.path = os.path.join(cfg.pre_trained_path,'yolov3.weights')
        self.encoders = Darknet(os.path.join(cfg.pre_trained_path,'yolov3.weights'))
        self.out_channels = self.encoders.out_channels.copy()
        self.in_channel = self.out_channels.pop(0)
        self.relu = nn.LeakyReLU(0.1)
        decoders = []
        channels = [512,256,128]
        for i,ind in enumerate(cfg.anchor_divide):
            decoder = self.make_prediction(len(ind)*(cfg.cls_num+5),NonResidual,channels[i],upsample=i!=0)
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)
        self.loss = LossAPI(cfg,loss)
    def initialization(self):
        for m in self.modules():
            init_weights(m)
        with open(self.path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights
        ptr = 0
        stack = []
        cutoff = None
        if "darknet53.conv.74" in self.path:
            cutoff = 75
        bnum =0
        cnum = 0
        bbnum = 0
        for i,m in enumerate(self.modules()):            
            if i==cutoff:
                 break
            if type(m) == nn.Conv2d:
                if type(m.bias) == type(m.weight):
                    #exist bias
                    num_b = m.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(m.bias)
                    m.bias.data.copy_(conv_b)
                    ptr += num_b
                    # Load conv. weights
                    num_w = m.weight.numel()
                    conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(m.weight)
                    m.weight.data.copy_(conv_w)
                    ptr += num_w
                    cnum +=1
                    bbnum+=1
                else:
                    stack.append(m)
            if type(m) == nn.BatchNorm2d:
                num_b = m.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(m.bias)
                m.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(m.weight)
                m.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(m.running_mean)
                m.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(m.running_var)
                m.running_var.data.copy_(bn_rv)
                ptr += num_b
                m = stack.pop(0)
                bnum+=1
                # Load conv. weights
                num_w = m.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(m.weight)
                m.weight.data.copy_(conv_w)
                ptr += num_w
                cnum +=1 
                assert len(stack)==0
        print("Mine:",cnum,bnum,bbnum,ptr)
        print("finish load from path:",self.path)
    def make_prediction(self,out_channel,block,channel,upsample=True):
        if upsample:
            upsample = nn.Sequential(conv1x1(self.in_channel,channel),nn.BatchNorm2d(channel,momentum=0.9, eps=1e-5),
                                           self.relu,Upsample(scale_factor=2,mode='nearest'))
            cat_channel = self.out_channels.pop(0)
            self.in_channel = channel + cat_channel
        else:
            upsample = nn.Identity()
        decoders=[block(self.in_channel,channel),block(channel*block.multiple,channel)]
        decoders.append(nn.Sequential(conv1x1(channel*block.multiple,channel),nn.BatchNorm2d(channel,momentum=0.9, eps=1e-5),self.relu))        
        pred = nn.Sequential(conv3x3(channel,channel*block.multiple),nn.BatchNorm2d(channel*block.multiple,momentum=0.9, eps=1e-5),self.relu,
                conv1x1(channel*block.multiple,out_channel,bias=True))
        self.in_channel = channel
        return nn.ModuleList([upsample,nn.Sequential(*decoders),pred])
    def forward(self,x,optimizer=None,gts=None):
        size = x.shape[-2:]
        feats = self.encoders(x)
        #channels:[1024,512,256,128,64]
        #spatial :[8,16,32,64,128] suppose inp is 256
        outs = list(range(len(self.decoders)))
        x = feats.pop(0)
        y = []
        for i,decoders in enumerate(self.decoders):
            up,decoder,pred = decoders
            x = torch.cat([up(x)]+y,dim=1)
            x = decoder(x)
            out = pred(x)
            outs[i] = out
            y = [feats.pop(0)]
        if self.training:
            display,loss = self.loss(outs,gts,size)
            if optimizer!=None:
                # for network like GAN
                pass
            else:          
                return display,loss
        else:
            return  self.loss(outs,size=size,infer=True)
class YOLO_SPP(YOLO):
    def __init__(self,cfg,loss):
        super(YOLO_SPP,self).__init__(cfg,loss)
        self.encoders = Darknet(os.path.join(cfg.pre_trained_path,'yolov3-spp.weights'))
        self.out_channels = self.encoders.out_channels.copy()
        self.in_channel = self.out_channels.pop(0)
        self.relu = nn.LeakyReLU(0.1)
        decoders = []
        channels = [512,256,128]
        pool_size = [1,5,9,13]
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=ks,stride=1,padding=(ks-1) // 2) for ks in pool_size])
        for i,ind in enumerate(cfg.anchor_divide):
            if i==0:
                decoder = self.make_prediction_SPP(len(ind)*(cfg.cls_num+5),NonResidual,channels[i])
            else:
                decoder = self.make_prediction(len(ind)*(cfg.cls_num+5),NonResidual,channels[i])
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)
        self.path = os.path.join(cfg.pre_trained_path,'yolov3-spp.weights')
    def make_prediction_SPP(self,out_channel,block,channel):
        upsample = nn.Sequential(NonResidual(self.in_channel,channel),
                                   conv1x1(channel*NonResidual.multiple,channel),nn.BatchNorm2d(channel,momentum=0.9, eps=1e-5),self.relu)
        self.in_channel = channel*4
        decoders=[block(self.in_channel,channel)]
        decoders.append(nn.Sequential(conv1x1(channel*block.multiple,channel),nn.BatchNorm2d(channel,momentum=0.9, eps=1e-5),self.relu))        
        pred = nn.Sequential(conv3x3(channel,channel*block.multiple),nn.BatchNorm2d(channel*block.multiple,momentum=0.9, eps=1e-5),self.relu,
                conv1x1(channel*block.multiple,out_channel,bias=True))
        self.in_channel = channel
        return nn.ModuleList([upsample,nn.Sequential(*decoders),pred])
    def forward(self,x,optimizer=None,gts=None):
        size = x.shape[-2:]
        feats = self.encoders(x)
        #channels:[1024,512,256,128,64]
        #spatial :[8,16,32,64,128] suppose inp is 256
        outs = []
        x = feats.pop(0)
        y = []
        for i,decoders in enumerate(self.decoders):
            up,decoder,pred = decoders
            x = up(x)
            if i==0:                
                x = torch.cat([maxpool(x) for maxpool in self.pools],dim=1)
            x = torch.cat([x]+y,dim=1)
            x = decoder(x)
            out = pred(x)
            outs.append(out)
            y = [feats.pop(0)]
        if self.training:
            display,loss = self.loss(outs,gts,size)
            if optimizer!=None:
                # for network like GAN
                pass
            else:          
                return display,loss
        else:
            return  self.loss(outs,size=size,infer=True)

    




    