import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import os
import numpy as np
import torch
#keep same varaible name from Resnet to use imagenet pretrained weight
depths = {18:[2,2,2,2],34:[3,4,6,3],50:[3,4,6,3],101:[3,4,23,3],152:[3,8,36,3],53:[1,2,8,8,4]}
channels = [64,128,256,512]
cahnnels_yolo = [32,64,128,256,512]
urls = {
    'res18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'res34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'res50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'res101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'res152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
def conv3x3(in_channels, out_channels, stride=1,bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

def conv1x1(in_channels,out_channels,stride=1,bias=False):
    return nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=bias)

#bias will be added in normalization layer
class BasicBlock(nn.Module):
    multiple=1
    def __init__(self,in_channels,channels,stride=1):
        super(BasicBlock,self).__init__()
        self.conv1 = conv3x3(in_channels,channels)
        self.relu = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = conv3x3(channels,channels,stride)
        if in_channels != channels or stride!=1:
            self.downsample =  nn.Sequential(conv1x1(in_channels,channels,stride),
                                            nn.BatchNorm2d(channels))
        else:
            self.downsample = nn.Identity()
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self,x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)

        if self.downsample != None:
            x = self.downsample(x)
        y += x
        y = self.relu(y)

        return y
class Bottleneck(nn.Module):
    multiple = 4
    def __init__(self,in_channels,channels,stride=1):
        super(Bottleneck,self).__init__()
        self.conv1 = conv1x1(in_channels,channels)
        self.relu = nn.LeakyReLU(0.01)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = conv3x3(channels,channels,stride)
        if in_channels != channels*Bottleneck.multiple or stride!=1:
            self.downsample =  nn.Sequential(conv1x1(in_channels,channels*Bottleneck.multiple,stride),
                                            nn.BatchNorm2d(channels*Bottleneck.multiple))
        else:
            self.downsample = nn.Identity()
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = conv1x1(channels,Bottleneck.multiple*channels)
        self.bn3 = nn.BatchNorm2d(Bottleneck.multiple*channels)
    def forward(self,x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)

        y = self.conv3(y)
        y = self.bn3(y)

        x = self.downsample(x)
        y += x
        y = self.relu(y)

        return y

class BaseBlock(nn.Module):
    #for Darknet
    multiple=2
    def __init__(self,in_channels,channels,stride=1):
        super(BaseBlock,self).__init__()
        self.conv1 = conv1x1(in_channels,channels,stride)
        self.relu = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm2d(channels,momentum=0.9, eps=1e-5)
        self.conv2 = conv3x3(channels,channels*BaseBlock.multiple)
        if in_channels != channels*BaseBlock.multiple or stride!=1:
            self.downsample =  nn.Sequential(conv1x1(in_channels,channels*BaseBlock.multiple,stride),
                                            nn.BatchNorm2d(channels*BaseBlock.multiple,momentum=0.9, eps=1e-5))
        else:
            self.downsample = nn.Identity()
        self.bn2 = nn.BatchNorm2d(channels*BaseBlock.multiple,momentum=0.9, eps=1e-5)
    def forward(self,x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)

        if self.downsample != None:
            x = self.downsample(x)
        y += x
        

        return y
class Darknet(nn.Module):
    def __init__(self,path):
        super(Darknet,self).__init__()
        self.depths = [1,2,8,8,4]
        self.levels = len(self.depths)
        self.channels = [32,64,128,256,512]
        self.relu = nn.LeakyReLU(0.1)
        self.conv1 = conv3x3(3,32)
        self.bn1 = nn.BatchNorm2d(32,momentum=0.9, eps=1e-5)
        self.in_channel = self.channels[0]
        encoders = []
        self.out_channels =[]
        for i in range(self.levels):
            encoders.append(self.make_encoders(self.channels[i],BaseBlock,depth=self.depths[i],downsample=True))
        self.encoders = nn.ModuleList(encoders)
        self.path = path
    def make_encoders(self,channel,block,depth=1,downsample=False):
        blocks = []
        if downsample:
            out_channel = channel*2
            blocks.append(nn.Sequential(conv3x3(self.in_channel,out_channel,stride=2),
                          nn.BatchNorm2d(out_channel,momentum=0.9, eps=1e-5),self.relu))
            self.in_channel = out_channel
        for _ in range(depth):
            blocks.append(block(self.in_channel,channel))
            self.in_channel = channel*block.multiple
        self.out_channels.insert(0,self.in_channel)
        return nn.Sequential(*blocks)

    def forward(self,x):
        #suppose x has shape: 3,256,256
               
        x = self.conv1(x)
        x = self.bn1(x) 
        x = self.relu(x)       
        
        
        #32,256,256

        feats = [x]
        for encoder in self.encoders:
            x = encoder(x)
            feats.insert(0,x)
        return feats
    def load_dark_net(self):
        # Open the weights file
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
        num = 0
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
            else:
                print(type(m))
        print(cnum,bnum,bbnum,ptr)
        print("finish load from path:",self.path)

class ResNet(nn.Module):
    def __init__(self,depth):
        super(ResNet,self).__init__()
        self.depths = depths[depth]
        self.in_channel = channels[0] #64
        self.conv1 = nn.Conv2d(3,self.in_channel,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.LeakyReLU(0.01)
        self.url = urls['res'+str(depth)]
        self.channels=[]
        if depth < 50 :
            block = BasicBlock
        else:
            block = Bottleneck
        self.max_pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1= self.stack_blocks(self.depths[0],channels[0],block)
        self.layer2 = self.stack_blocks(self.depths[1],channels[1],block,stride=2)
        self.layer3 = self.stack_blocks(self.depths[2],channels[2],block,stride=2)
        self.layer4 = self.stack_blocks(self.depths[3],channels[3],block,stride=2)
    def stack_blocks(self,depth,channels,block,stride=1):
        blocks = [block(self.in_channel,channels,stride)]
        for _ in range(1,depth):
            blocks.append(block(channels*block.multiple,channels))
        self.in_channel = channels*block.multiple
        self.channels.insert(0,self.in_channel)
        return nn.Sequential(*blocks)
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        return [x4,x3,x2,x1]
    def load_from_url(self):
        state_dict = self.state_dict()
        pretrained_state_dict = model_zoo.load_url(self.url)
        for k, v in pretrained_state_dict.items():
            if k not in state_dict:
                continue
            state_dict[k] = v
        self.load_state_dict(state_dict)










