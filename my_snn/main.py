# Copyright (c) 2024 Byeonghyeon Kim 
# github site: https://github.com/bhkim003/ByeonghyeonKim
# email: bhkim003@snu.ac.kr
 
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.datasets
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

import time

from snntorch import spikegen
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML

from tqdm import tqdm


from apex.parallel import DistributedDataParallel as DDP

# my module import
from modules import *

# modules 폴더에 새모듈.py 만들면
# modules/__init__py 파일에 form .새모듈 import * 하셈
# 그리고 새모듈.py에서 from modules.새모듈 import * 하셈


### DDP Setting #######################################
### DDP Setting #######################################
parser = argparse.ArgumentParser(description='my_snn training')

# local_rank는 command line에서 따로 줄 필요는 없지만, 선언은 필요
parser.add_argument("--local_rank", default=0, type=int)

# User's argument
# parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('--resume', '-r', action='store_true',
#                     help='resume from checkpoint')

args = parser.parse_args() # 이거 적어줘야됨. parser argument선언하고

args.gpu = args.local_rank
torch.cuda.set_device(args.gpu)
torch.distributed.init_process_group(backend="nccl", init_method="env://")
args.world_size = torch.distributed.get_world_size()
### DDP Setting #######################################
### DDP Setting #######################################

print('gpu what', args.gpu)

def my_snn_system(devices = "0,1,2,3", # DDP 쓸 땐 안 씀
                    my_seed = 42,
                    TIME = 8,
                    BATCH = 256,
                    IMAGE_SIZE = 32,
                    which_data = 'CIFAR10',
                    data_path = '/data2',
                    rate_coding = True,
    
                    lif_layer_v_init = 0.0,
                    lif_layer_v_decay = 0.6,
                    lif_layer_v_threshold = 1.2,
                    lif_layer_v_reset = 0.0,
                    lif_layer_sg_width = 1,

                    # synapse_conv_in_channels = IMAGE_PIXEL_CHANNEL,
                    synapse_conv_kernel_size = 3,
                    synapse_conv_stride = 1,
                    synapse_conv_padding = 1,
                    synapse_conv_trace_const1 = 1,
                    synapse_conv_trace_const2 = 0.6,

                    # synapse_fc_out_features = CLASS_NUM,
                    synapse_fc_trace_const1 = 1,
                    synapse_fc_trace_const2 = 0.6,

                    pre_trained = False,
                    convTrue_fcFalse = True,
                    cfg = [64, 64],
                    pre_trained_path = "net_save/save_now_net.pth",
                    learning_rate = 0.0001,
                    epoch_num = 100,
                    verbose_interval = 100,
                    tdBN_on = False,
                    BN_on = False,

                    surrogate = 'sigmoid',

                    gradient_verbose = False,

                    BPTT_on = False,

                    scheduler_name = 'no',
                  ):


    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    # os.environ["CUDA_VISIBLE_DEVICES"]= devices ## DDP 쓸땐 안씀

    
    torch.manual_seed(my_seed)


    

    train_loader, test_loader, synapse_conv_in_channels, synapse_fc_out_features = data_loader(
            which_data,
            data_path, 
            rate_coding, 
            BATCH, 
            IMAGE_SIZE)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if pre_trained == False:
        if (convTrue_fcFalse == False):
            net = MY_SNN_FC(cfg, synapse_conv_in_channels, IMAGE_SIZE, synapse_fc_out_features,
                     synapse_fc_trace_const1, synapse_fc_trace_const2, 
                     lif_layer_v_init, lif_layer_v_decay, 
                     lif_layer_v_threshold, lif_layer_v_reset,
                     lif_layer_sg_width,
                     tdBN_on,
                     BN_on, TIME,
                     surrogate,
                     BPTT_on).to(device)
        else:
            net = MY_SNN_CONV(cfg, synapse_conv_in_channels, IMAGE_SIZE,
                     synapse_conv_kernel_size, synapse_conv_stride, 
                     synapse_conv_padding, synapse_conv_trace_const1, 
                     synapse_conv_trace_const2, 
                     lif_layer_v_init, lif_layer_v_decay, 
                     lif_layer_v_threshold, lif_layer_v_reset,
                     lif_layer_sg_width,
                     synapse_fc_out_features, synapse_fc_trace_const1, synapse_fc_trace_const2,
                     tdBN_on,
                     BN_on, TIME,
                     surrogate,
                     BPTT_on).to(device)
        # net = torch.nn.DataParallel(net) # ddp 쓸땐 하면 안 됨.
        device = args.gpu
        net = net.to(args.gpu)
        net = DDP(net, delay_allreduce=True)
    else:
        net = torch.load(pre_trained_path)
        device = args.gpu
        net = net.to(args.gpu)

    print(net)


    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)


    if (scheduler_name == 'StepLR'):
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif (scheduler_name == 'ExponentialLR'):
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif (scheduler_name == 'ReduceLROnPlateau'):
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    elif (scheduler_name == 'CosineAnnealingLR'):
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    elif (scheduler_name == 'OneCycleLR'):
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(train_loader), epochs=100)
    else:
        pass # 'no' scheduler

    val_acc = 0
    for epoch in range(epoch_num):
        if torch.distributed.get_rank() == 0:
            print('EPOCH', epoch)
        epoch_start_time = time.time()
        running_loss = 0.0
        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), desc='train', dynamic_ncols=True, position=0, leave=True):
            net.train()

            iter_one_train_time_start = time.time()

            inputs, labels = data

            if rate_coding == True :
                inputs = spikegen.rate(inputs, num_steps=TIME)
            else :
                inputs = inputs.repeat(TIME, 1, 1, 1, 1)
            
            # inputs: [Time, Batch, Channel, Height, Width]   

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # inputs: [Time, Batch, Channel, Height, Width]   
            inputs = inputs.permute(1, 0, 2, 3, 4) # net에 넣어줄때는 batch가 젤 앞 차원으로 와야함. # dataparallel때매
            # inputs: [Batch, Time, Channel, Height, Width]   
        
            outputs = net(inputs)

            batch = BATCH 
            if labels.size(0) != BATCH: 
                batch = labels.size(0)


            ####### training accruacy ######
            correct = 0
            total = 0
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted[0:batch] == labels).sum().item()
            if torch.distributed.get_rank() == 0:    
                if i % verbose_interval == 9:
                    print(f'iter: {i} / {len(train_loader)}')  
                    # print(f'training acc: {100 * correct / total:.2f}%, lr={[param_group["lr"] for param_group in optimizer.param_groups]:.10f}')
                    print(f'{epoch}-{i} training acc: {100 * correct / total:.2f}%, lr={[f"{lr:.10f}" for lr in (param_group["lr"] for param_group in optimizer.param_groups)]}')
            ################################
            training_acc_string = f'training acc: {100 * correct / total:.2f}%'

            loss = criterion(outputs[0:batch,:], labels)
            loss.backward()

            # # optimizer.zero_grad()와 loss.backward() 호출 후에 실행해야 합니다.
            if torch.distributed.get_rank() == 0:    
                if (gradient_verbose == True):
                    if (i % verbose_interval == 9):
                        print('\n\nepoch', epoch, 'iter', i)
                        for name, param in net.named_parameters():
                            if param.requires_grad:
                                print('\n\n\n\n' , name, param.grad)
            
            optimizer.step()

            running_loss += loss.item()
            if torch.distributed.get_rank() == 0: 
                # print("Epoch: {}, Iter: {}, Loss: {}".format(epoch + 1, i + 1, running_loss / 100))
                pass   
                
            iter_one_train_time_end = time.time()
            elapsed_time = iter_one_train_time_end - iter_one_train_time_start  # 실행 시간 계산
            if torch.distributed.get_rank() == 0:    
                if (i % verbose_interval == 9):
                    print(f"iter_one_train_time: {elapsed_time} seconds")

            if i % verbose_interval == 9:
                iter_one_val_time_start = time.time()
                
                correct = 0
                total = 0
                with torch.no_grad():
                    net.eval()
                    how_many_val_image=0
                    for data in test_loader:
                        how_many_val_image += 1
                        inputs, labels = data
            
                        if rate_coding == True :
                            inputs = spikegen.rate(inputs, num_steps=TIME)
                        else :
                            inputs = inputs.repeat(TIME, 1, 1, 1, 1)

                        
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = net(inputs.permute(1, 0, 2, 3, 4))
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        batch = BATCH 
                        if labels.size(0) != BATCH: 
                            batch = labels.size(0)
                        correct += (predicted[0:batch] == labels).sum().item()
                        val_loss = criterion(outputs[0:batch,:], labels)
                    if torch.distributed.get_rank() == 0:    
                        print(f'{epoch}-{i} validation acc: {100 * correct / total:.2f}%, lr={[f"{lr:.10f}" for lr in (param_group["lr"] for param_group in optimizer.param_groups)]}')

                iter_one_val_time_end = time.time()
                elapsed_time = iter_one_val_time_end - iter_one_val_time_start  # 실행 시간 계산
                if torch.distributed.get_rank() == 0:    
                    print(f"iter_one_val_time: {elapsed_time} seconds")
                if val_acc < correct / total:
                    val_acc = correct / total
                    torch.save(net.state_dict(), "net_save/save_now_net_weights.pth")
                    torch.save(net, "net_save/save_now_net.pth")
                    torch.save(net.module.state_dict(), "net_save/save_now_net_weights2.pth")
                    torch.save(net.module, "net_save/save_now_net2.pth")
        
        if (scheduler_name != 'no'):
            if (scheduler_name == 'ReduceLROnPlateau'):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        
        epoch_time_end = time.time()
        epoch_time = epoch_time_end - epoch_start_time  # 실행 시간 계산
        
        if torch.distributed.get_rank() == 0:    
            print(f"epoch_time: {epoch_time} seconds")
            print('\n')







### my_snn control board ########################


decay = 0.95

my_snn_system(  devices = "0,1,2,3,4,5", #!!! DDP 쓸 땐 안 씀
                my_seed = 42,
                TIME = 8,
                BATCH = 128,
                IMAGE_SIZE = 32,
                which_data = 'CIFAR10',# 'CIFAR10' 'MNIST' 'FASHION_MNIST'
                data_path = '/data2', # YOU NEED TO CHANGE THIS
                rate_coding = True, # True # False

                lif_layer_v_init = 0.0,
                lif_layer_v_decay = decay,
                lif_layer_v_threshold = 1.2,
                lif_layer_v_reset = 0.0, #현재 안씀. 걍 빼기 해버림
                lif_layer_sg_width = 1.0, # surrogate sigmoid 쓸 때는 의미없음

                # synapse_conv_in_channels = IMAGE_PIXEL_CHANNEL,
                synapse_conv_kernel_size = 3,
                synapse_conv_stride = 1,
                synapse_conv_padding = 1,
                synapse_conv_trace_const1 = 1,
                synapse_conv_trace_const2 = decay, # lif_layer_v_decay

                # synapse_fc_out_features = CLASS_NUM,
                synapse_fc_trace_const1 = 1,
                synapse_fc_trace_const2 = decay, # lif_layer_v_decay

                pre_trained = False, # True # False
                convTrue_fcFalse = True, # True # False
                cfg = [8, 16, 'P', 16, 32, 'P', [32, 32], 'P', [32, 32]],
                # cfg = [64, [64, 64], 64], # 끝에 linear classifier 하나 자동으로 붙습니다
                pre_trained_path = "net_save/save_now_net.pth",
                learning_rate = 0.00001,
                epoch_num = 200,
                verbose_interval = 100, #숫자 크게 하면 꺼짐
                tdBN_on = False,  # True # False
                BN_on = True,  # True # False
                
                surrogate = 'rough_rectangle', # 'rectangle' 'sigmoid' 'rough_rectangle'
                
                gradient_verbose = False,  # True # False  # weight gradient 각 layer마다 띄워줌

                BPTT_on = False,  # True # False

                scheduler_name = 'StepLR' # 'no' 'StepLR' 'ExponentialLR' 'ReduceLROnPlateau' 'CosineAnnealingLR' 'OneCycleLR'
                )

'''
cfg 종류 = {
[64, 64]
[64, 64, 64, 64]
[64, [64, 64], 64]
[64, 128, 256]
[64, 128, 'P', 256, 512, 'P', [512, 512], 'P', [512, 512]]
[64, 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512]
[64, 64]
} 
'''

'''
CUDA_VISIBLE_DEVICES=2,3,4 python -m torch.distributed.launch --nproc_per_node=3 main.py
CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch --nproc_per_node=4 main.py
CUDA_VISIBLE_DEVICES=1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node=5 main.py
'''
