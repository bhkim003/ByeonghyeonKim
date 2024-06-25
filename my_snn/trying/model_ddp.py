import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

import time
import timeit

from apex.parallel import DistributedDataParallel as DDP

#실행코드
'''
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node=6 model_ddp.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 model_ddp.py
CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=7 model_ddp.py
'''


""" DDP Setting Start """
parser = argparse.ArgumentParser(description='my_snn CIFAR10 Training')

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




best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch





# Model
class SYNAPSE_FC_METHOD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spike_one_time, spike_now, weight, bias):
        ctx.save_for_backward(spike_one_time, spike_now, weight, bias)
        return F.linear(spike_one_time, weight, bias=bias)

    @staticmethod
    def backward(ctx, grad_output_current):
        #############밑에부터 수정해라#######
        spike_one_time, spike_now, weight, bias = ctx.saved_tensors
        
        ## 이거 클론해야되는지 모르겠음!!!!
        grad_output_current_clone = grad_output_current.clone()

        grad_input_spike = grad_weight = grad_bias = None


        if ctx.needs_input_grad[0]:
            grad_input_spike = grad_output_current_clone @ weight
        if ctx.needs_input_grad[2]:
            grad_weight = grad_output_current_clone.t() @ spike_now
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output_current_clone.sum(0)

        return grad_input_spike, None, grad_weight, grad_bias

     
class SYNAPSE_FC(nn.Module):
    def __init__(self, in_features, out_features, trace_const1=1, trace_const2=0.7):
        super(SYNAPSE_FC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.trace_const1 = trace_const1
        self.trace_const2 = trace_const2

        # self.weight = torch.randn(self.out_features, self.in_features, requires_grad=True)
        # self.bias = torch.randn(self.out_features, requires_grad=True)
        self.weight = nn.Parameter(torch.randn(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.randn(self.out_features))

    def forward(self, spike):
        # spike: [Time, Batch, Features]   
        Time = spike.shape[0]
        Batch = spike.shape[1] 
        output_current = torch.zeros(Time, Batch, self.out_features, device=spike.device)

        # spike_detach = spike.detach().clone()
        spike_detach = spike.detach()
        spike_past = torch.zeros_like(spike_detach[0], device=spike.device)
        spike_now = torch.zeros_like(spike_detach[0], device=spike.device)

        for t in range(Time):
            spike_now = self.trace_const1*spike_detach[t] + self.trace_const2*spike_past
            output_current[t]= SYNAPSE_FC_METHOD.apply(spike[t], spike_now, self.weight, self.bias) 
            spike_past = spike_now

        return output_current 


class SYNAPSE_CONV_METHOD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spike_one_time, spike_now, weight, bias, stride=1, padding=1):
        ctx.save_for_backward(spike_one_time, spike_now, weight, bias, torch.tensor([stride], requires_grad=False), torch.tensor([padding], requires_grad=False))
        return F.conv2d(spike_one_time, weight, bias=bias, stride=stride, padding=padding)

    @staticmethod
    def backward(ctx, grad_output_current):
        spike_one_time, spike_now, weight, bias, stride, padding = ctx.saved_tensors
        stride=stride.item()
        padding=padding.item()
        
        ## 이거 클론해야되는지 모르겠음!!!!
        grad_output_current_clone = grad_output_current.clone()

        grad_input_spike = grad_weight = grad_bias = None


        if ctx.needs_input_grad[0]:
            grad_input_spike = F.conv_transpose2d(grad_output_current_clone, weight, stride=stride, padding=padding)
        if ctx.needs_input_grad[2]:
            grad_weight = torch.nn.grad.conv2d_weight(spike_now, weight.shape, grad_output_current_clone,
                                                      stride=stride, padding=padding)
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output_current_clone.sum((0, -1, -2))

        return grad_input_spike, None, grad_weight, grad_bias, None, None

     



class SYNAPSE_CONV(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, trace_const1=1, trace_const2=0.7):
        super(SYNAPSE_CONV, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.trace_const1 = trace_const1
        self.trace_const2 = trace_const2

        # self.weight = torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, requires_grad=True)
        # self.bias = torch.randn(self.out_channels, requires_grad=True)
        self.weight = nn.Parameter(torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.randn(self.out_channels))

    def forward(self, spike):
        # spike: [Time, Batch, Channel, Height, Width]   
        Time = spike.shape[0]
        Batch = spike.shape[1] 
        Channel = self.out_channels
        Height = (spike.shape[3] + self.padding*2 - self.kernel_size) // self.stride + 1
        Width = (spike.shape[4] + self.padding*2 - self.kernel_size) // self.stride + 1
        output_current = torch.zeros(Time, Batch, Channel, Height, Width, device=spike.device)

        # spike_detach = spike.detach().clone()
        spike_detach = spike.detach()
        spike_past = torch.zeros_like(spike_detach[0])
        spike_now = torch.zeros_like(spike_detach[0])

        for t in range(Time):
            spike_now = self.trace_const1*spike_detach[t] + self.trace_const2*spike_past
            output_current[t]= SYNAPSE_CONV_METHOD.apply(spike[t], spike_now, self.weight, self.bias, self.stride, self.padding) 
            spike_past = spike_now

        return output_current 



class LIF_METHOD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_current_one_time, v_one_time, v_decay, v_threshold, v_reset, sg_width):
        v_one_time = v_one_time * v_decay + input_current_one_time # leak + pre-synaptic current integrate
        spike = (v_one_time >= v_threshold).float() #fire
        ctx.save_for_backward(v_one_time, torch.tensor([v_decay], requires_grad=False), 
                              torch.tensor([v_threshold], requires_grad=False), 
                              torch.tensor([v_reset], requires_grad=False), 
                              torch.tensor([sg_width], requires_grad=False)) # save before reset
        v_one_time = (v_one_time - spike * v_threshold).clamp_min(0) # reset
        return spike, v_one_time

    @staticmethod
    def backward(ctx, grad_output_spike, grad_output_v):
        v_one_time, v_decay, v_threshold, v_reset, sg_width = ctx.saved_tensors
        v_decay=v_decay.item()
        v_threshold=v_threshold.item()
        v_reset=v_reset.item()
        sg_width=sg_width.item()

        grad_input_current = grad_output_spike.clone()
        # grad_temp_v = grad_output_v.clone() # not used

        ################ select one of the following surrogate gradient functions ################
        #===========surrogate gradient function (rectangle)
        grad_input_current = grad_input_current * ((v_one_time - v_threshold).abs() < sg_width/2).float() / sg_width

        #===========surrogate gradient function (sigmoid)
        # sig = torch.sigmoid((v_one_time - v_threshold))
        # grad_input_current =  sig*(1-sig)*grad_input_current

        #===========surrogate gradient function (rough rectangle)
        # v_minus_th = (v_one_time - v_threshold)
        # grad_input_current[v_minus_th <= -.5] = 0
        # grad_input_current[v_minus_th > .5] = 0
        ###########################################################################################
        return grad_input_current, None, None, None, None, None

class LIF_layer(nn.Module):
    def __init__ (self, v_init = 0.0, v_decay = 0.8, v_threshold = 0.5, v_reset = 0.0, sg_width = 1):
        super(LIF_layer, self).__init__()
        self.v_init = v_init
        self.v_decay = v_decay
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.sg_width = sg_width

    def forward(self, input_current):
        v = torch.full_like(input_current, fill_value = self.v_init, dtype = torch.float) # v (membrane potential) init
        post_spike = torch.full_like(input_current, fill_value = self.v_init, device=input_current.device, dtype = torch.float) # v (membrane potential) init
        # i와 v와 post_spike size는 여기서 다 같음: [Time, Batch, Channel, Height, Width] 

        Time = v.shape[0]
        for t in range(Time):
            # leaky하고 input_current 더하고 fire하고 reset까지 (backward직접처리)
            post_spike[t], v[t] = LIF_METHOD.apply(input_current[t], v[t], 
                                            self.v_decay, self.v_threshold, self.v_reset, self.sg_width) 

        return post_spike   
    
    

data_path = '/data2/cifar10'

torch.manual_seed(42)

# HEPER PARAMETER
TIME = 8
BATCH = 256
IMAGE_PIXEL_CHANNEL = 3
IMAGE_SIZE = 32
CLASS_NUM = 10

## SYNAPSE_CONV 레이어의 하이퍼파라미터
synapse_conv_in_channels = IMAGE_PIXEL_CHANNEL
# synapse_conv_out_channels = layer별 지정
# synapse_conv_kernel_size = layer별 지정
synapse_conv_stride = 1
synapse_conv_padding = 1
synapse_conv_trace_const1 = 1
synapse_conv_trace_const2 = 0.7

## LIF_layer 레이어의 하이퍼파라미터
lif_layer_v_init = 0.0
lif_layer_v_decay = 0.8
lif_layer_v_threshold = 1.2
lif_layer_v_reset = 0.0
lif_layer_sg_width = 1

## SYNAPSE_FC 레이어의 하이퍼파라미터
# synapse_fc_in_features = 마지막CONV_OUT_CHANNEL * H * W
synapse_fc_out_features = CLASS_NUM
synapse_fc_trace_const1 = 1
synapse_fc_trace_const2 = 0.7


class MY_SNN_MK1(nn.Module):
    def __init__(self):
        super(MY_SNN_MK1, self).__init__()

        in_channels = synapse_conv_in_channels
        out_channels = 64
        self.synapse_conv1 = SYNAPSE_CONV(in_channels=in_channels, 
                                          out_channels=out_channels, 
                                          kernel_size=3, 
                                          stride=synapse_conv_stride, 
                                          padding=synapse_conv_padding, 
                                          trace_const1=synapse_conv_trace_const1, 
                                          trace_const2=synapse_conv_trace_const2)
        


        in_channels = 64
        out_channels = 64
        self.synapse_conv2 = SYNAPSE_CONV(in_channels=in_channels, 
                                          out_channels=out_channels, 
                                          kernel_size=3, 
                                          stride=synapse_conv_stride, 
                                          padding=synapse_conv_padding, 
                                          trace_const1=synapse_conv_trace_const1, 
                                          trace_const2=synapse_conv_trace_const2)
        

        
        self.lif_layer = LIF_layer(v_init=lif_layer_v_init, 
                                   v_decay=lif_layer_v_decay, 
                                   v_threshold=lif_layer_v_threshold, 
                                   v_reset=lif_layer_v_reset, 
                                   sg_width=lif_layer_sg_width)
        


        self.synapse_FC = SYNAPSE_FC(in_features=64*IMAGE_SIZE*IMAGE_SIZE,  # 마지막CONV의 OUT_CHANNEL * H * W
                                      out_features=CLASS_NUM, 
                                      trace_const1=synapse_fc_trace_const1, 
                                      trace_const2=synapse_fc_trace_const2)
        

    def forward(self, spike_input):
        spike_input = self.synapse_conv1(spike_input)
        spike_input = self.lif_layer(spike_input)
                                     
        spike_input = self.synapse_conv2(spike_input)
        spike_input = self.lif_layer(spike_input)

        spike_input = spike_input.view(spike_input.size(0), spike_input.size(1), -1)
        
        spike_input = self.synapse_FC(spike_input)
        spike_input = spike_input.sum(axis=0)
        return spike_input




############################################################
####################### DATASET ############################
transform_train = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]) #https://github.com/kuangliu/pytorch-cifar/issues/19

transform_test = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),])

trainset = torchvision.datasets.CIFAR10(root=data_path,
                                      train=True,
                                      download=True,
                                      transform=transform_train)


testset = torchvision.datasets.CIFAR10(root=data_path,
                                     train=False,
                                     download=True,
                                     transform=transform_test)

train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)

test_sampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False)

train_loader = DataLoader(trainset,
                          batch_size =BATCH,
                          num_workers =2,
                          sampler=train_sampler)
test_loader = DataLoader(testset,
                          batch_size =BATCH,
                          num_workers =2,
                          sampler=test_sampler)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck') 
####################### DATASET END ############################

net = MY_SNN_MK1()
device = args.gpu
net = net.to(args.gpu)
net = DDP(net, delay_allreduce=True)



criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(100):
    if torch.distributed.get_rank() == 0:
        print('epoch', epoch)
    epoch_start_time = time.time()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        if torch.distributed.get_rank() == 0:
            print('\niter', i)
        iter_one_train_time_start = time.time()

        inputs, labels = data
        inputs = inputs.repeat(TIME, 1, 1, 1, 1)
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)

        batch = BATCH 
        if labels.size(0) != BATCH: 
            batch = labels.size(0)

        loss = criterion(outputs[0:batch,:], labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if torch.distributed.get_rank() == 0:
            print("Epoch: {}, Iter: {}, Loss: {}".format(epoch + 1, i + 1, running_loss / 100))
        running_loss = 0.0

        iter_one_train_time_end = time.time()
        elapsed_time = iter_one_train_time_end - iter_one_train_time_start  # 실행 시간 계산
        if torch.distributed.get_rank() == 0:    
            print(f"iter_one_train_time: {elapsed_time} seconds")


        correct = 0
        total = 0

        acc = 0
        if i % 100 == 9:
            iter_one_val_time_start = time.time()

            with torch.no_grad():
                how_many_val_image=0
                for data in test_loader:
                    how_many_val_image += 1
                    images, labels = data
                    images = images.repeat(TIME, 1, 1, 1, 1)
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    batch = BATCH 
                    if labels.size(0) != BATCH: 
                        batch = labels.size(0)
                    if torch.distributed.get_rank() == 0:    
                        print(f"batch {batch}")
                    correct += (predicted[0:batch] == labels).sum().item()
                    # if how_many_val_image > 10:
                    #     break
                if torch.distributed.get_rank() == 0:
                    print(f'validation acc: {100 * correct / total:.2f}%')


            iter_one_val_time_end = time.time()
            elapsed_time = iter_one_val_time_end - iter_one_val_time_start  # 실행 시간 계산
            if torch.distributed.get_rank() == 0:
                print(f"iter_one_val_time: {elapsed_time} seconds")
            if acc < correct / total:
                acc = correct / total
                torch.save(net.state_dict(), "net_save/save_now_net_weights.pth")
                torch.save(net, "net_save/save_now_net.pth")
                torch.save(net.module.state_dict(), "net_save/save_now_net_weights2.pth")
                torch.save(net.module, "net_save/save_now_net2.pth")
    epoch_time_end = time.time()
    epoch_time = epoch_time_end - epoch_start_time  # 실행 시간 계산
    if torch.distributed.get_rank() == 0:    
        print(f"{epoch} epoch_time: {epoch_time} seconds")
    print('\n')


'''
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node=6 model_ddp.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 model_ddp.py
'''