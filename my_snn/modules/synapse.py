import sys
import os
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


from modules.data_loader import *
from modules.network import *
from modules.neuron import *
from modules.synapse import *

##### OTTT Synapse ###########################################################
##### OTTT Synapse ###########################################################
##### OTTT Synapse ###########################################################
class SYNAPSE_CONV(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, trace_const1=1, trace_const2=0.7, TIME=8):
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

        self.TIME = TIME

    def forward(self, spike):
        # spike: [Time, Batch, Channel, Height, Width]   
        # print('spike.shape', spike.shape)
        Time = spike.shape[0]
        assert Time == self.TIME, 'Time dimension should be same as TIME'
        Batch = spike.shape[1] 
        Channel = self.out_channels
        Height = (spike.shape[3] + self.padding*2 - self.kernel_size) // self.stride + 1
        Width = (spike.shape[4] + self.padding*2 - self.kernel_size) // self.stride + 1

        # output_current = torch.zeros(Time, Batch, Channel, Height, Width, device=spike.device)
        output_current = []
        
        # spike_detach = spike.detach().clone()
        spike_detach = spike.detach()
        spike_past = torch.zeros_like(spike_detach[0])
        spike_now = torch.zeros_like(spike_detach[0])
        for t in range(Time):
            # print(f'time:{t}', torch.sum(spike_detach[t]/ torch.numel(spike_detach[t])))
            spike_now = self.trace_const1*spike_detach[t] + self.trace_const2*spike_past

            # output_current[t]= SYNAPSE_CONV_METHOD.apply(spike[t], spike_now, self.weight, self.bias, self.stride, self.padding) 
            output_current.append( SYNAPSE_CONV_METHOD.apply(spike[t], spike_now, self.weight, self.bias, self.stride, self.padding) )
            
            spike_past = spike_now
            # print(f'time:{t}', torch.sum(output_current[t]/ torch.numel(output_current[t])))

        output_current = torch.stack(output_current, dim=0)
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
   
class SYNAPSE_FC(nn.Module):
    def __init__(self, in_features, out_features, trace_const1=1, trace_const2=0.7, TIME=8):
        super(SYNAPSE_FC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.trace_const1 = trace_const1
        self.trace_const2 = trace_const2

        # self.weight = torch.randn(self.out_features, self.in_features, requires_grad=True)
        # self.bias = torch.randn(self.out_features, requires_grad=True)
        self.weight = nn.Parameter(torch.randn(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.randn(self.out_features))

        self.TIME = TIME

    def forward(self, spike):
        # spike: [Time, Batch, Features]   
        Time = spike.shape[0]
        assert Time == self.TIME, 'Time dimension should be same as TIME'
        Batch = spike.shape[1] 

        # output_current = torch.zeros(Time, Batch, self.out_features, device=spike.device)
        output_current = []

        # spike_detach = spike.detach().clone()
        spike_detach = spike.detach()
        spike_past = torch.zeros_like(spike_detach[0], device=spike.device)
        spike_now = torch.zeros_like(spike_detach[0], device=spike.device)

        for t in range(Time):
            spike_now = self.trace_const1*spike_detach[t] + self.trace_const2*spike_past
            # output_current[t]= SYNAPSE_FC_METHOD.apply(spike[t], spike_now, self.weight, self.bias) 
            output_current.append( SYNAPSE_FC_METHOD.apply(spike[t], spike_now, self.weight, self.bias) )
            
            spike_past = spike_now

        output_current = torch.stack(output_current, dim=0)
        return output_current 
    



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

##### OTTT Synapse ###########################################################
##### OTTT Synapse ###########################################################
##### OTTT Synapse ###########################################################










############## BPTT Synapse ##################################################
############## BPTT Synapse ##################################################
############## BPTT Synapse ##################################################
class SYNAPSE_CONV_BPTT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, trace_const1=1, trace_const2=0.7, TIME=8):
        super(SYNAPSE_CONV_BPTT, self).__init__()
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

        self.TIME = TIME

    def forward(self, spike):
        # spike: [Time, Batch, Channel, Height, Width]   
        # print('spike.shape', spike.shape)
        Time = spike.shape[0]
        assert Time == self.TIME, 'Time dimension should be same as TIME'
        Batch = spike.shape[1] 
        Channel = self.out_channels
        Height = (spike.shape[3] + self.padding*2 - self.kernel_size) // self.stride + 1
        Width = (spike.shape[4] + self.padding*2 - self.kernel_size) // self.stride + 1

        # output_current = torch.zeros(Time, Batch, Channel, Height, Width, device=spike.device)
        output_current = []
        
        for t in range(Time):
            # print(f'time:{t}', torch.sum(spike_detach[t]/ torch.numel(spike_detach[t])))
            output_current.append(F.conv2d(spike[t], self.weight, bias=self.bias, stride=self.stride, padding=self.padding))
            # print(f'time:{t}', torch.sum(output_current[t]/ torch.numel(output_current[t])))

        output_current = torch.stack(output_current, dim=0)
        return output_current

class SYNAPSE_FC_BPTT(nn.Module):
    def __init__(self, in_features, out_features, trace_const1=1, trace_const2=0.7, TIME=8):
        super(SYNAPSE_FC_BPTT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.trace_const1 = trace_const1
        self.trace_const2 = trace_const2

        # self.weight = torch.randn(self.out_features, self.in_features, requires_grad=True)
        # self.bias = torch.randn(self.out_features, requires_grad=True)
        self.weight = nn.Parameter(torch.randn(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.randn(self.out_features))

        self.TIME = TIME

    def forward(self, spike):
        # spike: [Time, Batch, Features]   
        Time = spike.shape[0]
        assert Time == self.TIME, 'Time dimension should be same as TIME'
        Batch = spike.shape[1] 

        output_current = []

        for t in range(Time):
            output_current.append(F.linear(spike[t], weight = self.weight, bias= self.bias))

        output_current = torch.stack(output_current, dim=0)
        return output_current 
############## BPTT Synapse ##################################################
############## BPTT Synapse ##################################################
############## BPTT Synapse ##################################################