from __future__ import print_function
import os
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import random
import math
import shutil
from torchvision import datasets, transforms
from torch.autograd import Variable # Useful info about autograd: http://pytorch.org/docs/master/notes/autograd.html

from darknet_multi import Darknet
from MeshPly import MeshPly
from utils import *    
from cfg import parse_cfg
import dataset_multi as dataset
from region_loss_multi import RegionLoss
from tqdm import tqdm


def makedirs(path):
    if not os.path.exists( path ):
        os.makedirs( path )

# Adjust learning rate during training, learning schedule can be changed in network config file
def adjust_learning_rate(optimizer, batch):
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch > steps[i]:
            lr = lr * scale
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/batch_size
    return optimizer.param_groups[0]['lr']

def train(epoch):

    global processed_batches
    
    # Initialize timer
    t0 = time.time()
    # Get the dataloader for training dataset
    train_loader = torch.utils.data.DataLoader(dataset.listDataset_pre(trainlist,shape=(init_width, init_height),
                                                            	   shuffle=True,
                                                            	   transform=transforms.Compose([transforms.ToTensor(),]), 
                                                            	   train=True, 
                                                            	   seen=model.module.seen,
                                                            	   batch_size=batch_size,
                                                            	   num_workers=num_workers, bg_file_names=bg_file_names),
                                                batch_size=batch_size, shuffle=False, **kwargs)

    # TRAINING
    lr = adjust_learning_rate(optimizer, epoch)    
    logging('epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), lr*1000))
    #log_file.write('epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), lr))
    # Start training
    model.train()
    t1 = time.time()
    avg_time = torch.zeros(9)
    niter = 0
    # Iterate through batches
    training_losses=[]
    for data, target in tqdm(iter(train_loader)):
        t2 = time.time()
        # adjust learning rate
        processed_batches = processed_batches + 1
        # Pass the data to GPU
        data = data.cuda()
        t3 = time.time()
        # Wrap tensors in Variable class for automatic differentiation
        data, target = Variable(data), Variable(target)
        t4 = time.time()
        # Zero the gradients before running the backward pass
        optimizer.zero_grad()
        t5 = time.time()
        # Forward pass
        output = model(data)
        t6 = time.time()
        model.module.seen = model.module.seen + data.data.size(0)
        region_loss.seen = region_loss.seen + data.data.size(0)
        # Compute loss, grow an array of losses for saving later on
        loss = region_loss(output, target)
        #training_iters.append(epoch * math.ceil(len(train_loader.dataset) / float(batch_size) ) + niter)        
        niter += 1
        t7 = time.time()
        # Backprop: compute gradient of the loss with respect to model parameters
        loss.backward()
        t8 = time.time()
        # Update weights
        optimizer.step()
        t9 = time.time()
        # Print time statistics
        t1 = time.time()
        training_losses.append(float(loss.item())/batch_size)
    t1 = time.time()
    avg = sum(training_losses)/len(training_losses)
    print('%d\t%f\t%f\n'%(epoch+1,lr*1000,avg))
    log_file.write('%d\t%f\t%f\n'%(epoch+1,lr*1000,avg))
    return epoch * math.ceil(len(train_loader.dataset) / float(batch_size) ) + niter - 1,avg

if __name__ == "__main__":

    # Training settings
    datacfg       = sys.argv[1]
    cfgfile       = sys.argv[2]
    weightfile    = sys.argv[3]

    # Parse configuration files
    print(datacfg)
    model_name = os.path.splitext(os.path.split(cfgfile)[-1])[0]
    print(model_name)
    pre = True
    data_options   = read_data_cfg(datacfg)
    net_options    = parse_cfg(cfgfile)[0]
    trainlist      = data_options['train']
    nsamples      = file_lines(trainlist)
    gpus          = data_options['gpus']  # e.g. 0,1,2,3
    gpus = '0'
    num_workers   = int(data_options['num_workers'])
    #backupdir     = data_options['backup']
    checkpoint = os.path.join(data_options['checkpoint'],model_name)+'-no-scale'
    if not os.path.exists(checkpoint):
        makedirs(checkpoint)
    log_file = os.path.join(checkpoint,'log.txt')
    log_file = open(log_file,'a+')
    if not pre:
        test_log_file = os.path.join(checkpoint,'test_log.txt')
        test_log_file = open(test_log_file,'a+')
    batch_size    = int(net_options['batch'])
    #max_batches   = int(net_options['max_batches'])
    learning_rate = float(net_options['learning_rate'])
    momentum      = float(net_options['momentum'])
    decay         = float(net_options['decay'])    
    bg_file_names = get_all_files('../dataset/VOCdevkit/VOC2012/JPEGImages')

    # Train parameters
    max_epochs    = 250 # max_batches*batch_size/nsamples+1
    use_cuda      = True
    seed          = int(time.time())
    eps           = 1e-5
    save_interval = 30 # epoches
    #dot_interval  = 70 # batches
    steps         = [25]+list(range(50,max_epochs,50))
    scales        = [0.5,0.5,0.5,0.1,0.1,0.1,0.1]
    best_acc       = -1 

    # Test parameters
    conf_thresh   = 0.05
    nms_thresh    = 0.4
    match_thresh  = 0.5
    iou_thresh    = 0.5
    im_width      = 640
    im_height     = 480 

    # Specify which gpus to use
    torch.manual_seed(seed)

    # Specifiy the model and the loss
    model       = Darknet(cfgfile)
    region_loss = model.loss

    # Model settings
    # model.load_weights(weightfile)
    # Model settings
    model.load_weights_until_last(weightfile)
    model.print_network()
    model.seen        = 0
    region_loss.iter  = model.iter
    region_loss.seen  = model.seen
    processed_batches = model.seen/batch_size
    init_width        = model.width
    init_height       = model.height
    init_epoch        = model.seen//nsamples

    # Variable to save
    training_iters          = []
    training_losses         = []
    testing_iters           = []
    testing_errors_pixel    = []
    testing_accuracies      = []


    # Specify the number of workers
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}


    # Pass the model to GPU
    params_dict = dict(model.named_parameters())
    params = []
    #init_epoch        = model.seen//nsamples
    for key, value in params_dict.items():
        if key.find('.bn') >= 0 or key.find('.bias') >= 0:
            params += [{'params': [value], 'weight_decay': 0.0}]
        else:
            params += [{'params': [value], 'weight_decay': decay*batch_size}]
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam optimization
    if use_cuda:
        #model = model.cuda()  # Multiple GPU parallelism
         model = torch.nn.DataParallel(model).cuda() # Multiple GPU parallelism

    # Get the optimizer
    for epoch in range(init_epoch, max_epochs): 
        # TRAIN
        niter,loss = train(epoch)
        model.module.save_weights('%s/init.weights' % (checkpoint))
        if loss < 0.1:
            break                
        log_file.flush()
    log_file.close()
    
