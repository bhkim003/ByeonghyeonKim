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
        # Kaiming 초기화
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bias, 0)

        self.TIME = TIME

    def forward(self, spike):
        # spike: [Time, Batch, Channel, Height, Width]   
        # print('spike.shape', spike.shape)
        Time = spike.shape[0]
        assert Time == self.TIME, f'Time dimension {Time} should be same as TIME {self.TIME}'
        Batch = spike.shape[1] 
        Channel = self.out_channels
        Height = (spike.shape[3] + self.padding*2 - self.kernel_size) // self.stride + 1
        Width = (spike.shape[4] + self.padding*2 - self.kernel_size) // self.stride + 1

        # output_current = torch.zeros(Time, Batch, Channel, Height, Width, device=spike.device)
        output_current = []
        
        # spike_detach = spike.detach().clone()
        spike_detach = spike.detach()
        spike_past = torch.zeros_like(spike_detach[0],requires_grad=False)
        spike_now = torch.zeros_like(spike_detach[0],requires_grad=False)
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


        # print('grad_input_spike_conv', grad_input_spike)
        # print('grad_weight_conv', grad_weight)
        # print('grad_bias_conv', grad_bias)
        # print('grad_input_spike_conv', ctx.needs_input_grad[0])
        # print('grad_weight_conv', ctx.needs_input_grad[2])
        # print('grad_bias_conv', ctx.needs_input_grad[3])

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
        # Xavier 초기화
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0)

        # nn.init.normal_(m.weight, 0, 0.01)
        # nn.init.constant_(m.bias, 0)
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
        spike_past = torch.zeros_like(spike_detach[0], device=spike.device,requires_grad=False)
        spike_now = torch.zeros_like(spike_detach[0], device=spike.device,requires_grad=False)

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

        # print('grad_input_spike_FC', grad_input_spike)
        # print('grad_weight_FC', grad_weight)
        # print('grad_bias_FC', grad_bias)
        # print('grad_input_spike_FC', ctx.needs_input_grad[0])
        # print('grad_weight_FC', ctx.needs_input_grad[2])
        # print('grad_bias_FC', ctx.needs_input_grad[3])
        
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
        # Kaiming 초기화
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bias, 0)

        self.TIME = TIME

        # self.ann_module = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding)
        
    
    def forward(self, spike):
        # spike: [Time, Batch, Channel, Height, Width]   
        # print('spike.shape', spike.shape)
        Time = spike.shape[0]
        assert Time == self.TIME, f'Time {Time} dimension should be same as TIME {self.TIME}'
        Batch = spike.shape[1] 
        Channel = self.out_channels
        Height = (spike.shape[3] + self.padding*2 - self.kernel_size) // self.stride + 1
        Width = (spike.shape[4] + self.padding*2 - self.kernel_size) // self.stride + 1

        # output_current = []
        # for t in range(Time):
        #     # print(f'time:{t}', torch.sum(spike_detach[t]/ torch.numel(spike_detach[t])))
        #     output_current.append(F.conv2d(spike[t], self.weight, bias=self.bias, stride=self.stride, padding=self.padding))
        #     # print(f'time:{t}', torch.sum(output_current[t]/ torch.numel(output_current[t])))
        # output_current = torch.stack(output_current, dim=0)
        
        T, B, *spatial_dims = spike.shape
        spike = F.conv2d(spike.reshape(T * B, *spatial_dims), self.weight, bias=self.bias, stride=self.stride, padding=self.padding)
        TB, *spatial_dims = spike.shape
        spike = spike.view(T , B, *spatial_dims).contiguous() 
        output_current = spike

        return output_current
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


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
        # # Kaiming 초기화
        # nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.constant_(self.bias, 0)
        # Xavier 초기화
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0)

        # nn.init.normal_(m.weight, 0, 0.01)
        # nn.init.constant_(m.bias, 0)

        self.TIME = TIME

    def forward(self, spike):
        # # spike: [Time, Batch, Features]   
        # Time = spike.shape[0]
        # assert Time == self.TIME, 'Time dimension should be same as TIME'
        # Batch = spike.shape[1] 

        # output_current = []

        # for t in range(Time):
        #     output_current.append(F.linear(spike[t], weight = self.weight, bias= self.bias))

        # output_current = torch.stack(output_current, dim=0)


        
        T, B, *spatial_dims = spike.shape
        assert T == self.TIME, 'Time dimension should be same as TIME'
        spike = spike.reshape(T * B, *spatial_dims)

        spike = F.linear(spike, weight = self.weight, bias= self.bias)

        TB, *spatial_dims = spike.shape
        spike = spike.view(T , B, *spatial_dims).contiguous() 
        output_current = spike

        return output_current 
############## BPTT Synapse ##################################################
############## BPTT Synapse ##################################################
############## BPTT Synapse ##################################################




############## Separable Conv Synapse #######################################
############## Separable Conv Synapse #######################################
############## Separable Conv Synapse #######################################
    
class SYNAPSE_SEPARABLE_CONV_BPTT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, trace_const1=1, trace_const2=0.7, TIME=8):
        super(SYNAPSE_SEPARABLE_CONV_BPTT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.trace_const1 = trace_const1
        self.trace_const2 = trace_const2

        self.conv_depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.conv_pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    
        self._initialize_weights()

    def _initialize_weights(self):
        # Xavier initialization for conv_depthwise weights
        nn.init.kaiming_uniform_(self.conv_depthwise.weight)
        if self.conv_depthwise.bias is not None:
            nn.init.constant_(self.conv_depthwise.bias, 0)
        
        # Xavier initialization for conv_pointwise weights
        nn.init.kaiming_uniform_(self.conv_pointwise.weight)
        if self.conv_pointwise.bias is not None:
            nn.init.constant_(self.conv_pointwise.bias, 0)

    def forward(self, x):
        T, B, *spatial_dims = x.shape
        x = x.reshape(T * B, *spatial_dims)

        x = self.conv_depthwise(x)
        x = self.conv_pointwise(x)

        TB, *spatial_dims = x.shape
        x = x.view(T , B, *spatial_dims).contiguous() 
        return x
    



class SYNAPSE_DEPTHWISE_CONV_BPTT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, trace_const1=1, trace_const2=0.7, TIME=8):
        super(SYNAPSE_DEPTHWISE_CONV_BPTT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.trace_const1 = trace_const1
        self.trace_const2 = trace_const2

        assert in_channels == out_channels, 'in_channels should be same as out_channels for depthwise conv'
        self.conv_depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self._initialize_weights()

    def _initialize_weights(self):
        # Xavier initialization for conv_depthwise weights
        nn.init.kaiming_uniform_(self.conv_depthwise.weight)
        if self.conv_depthwise.bias is not None:
            nn.init.constant_(self.conv_depthwise.bias, 0)

    def forward(self, x):
        T, B, *spatial_dims = x.shape
        x = x.reshape(T * B, *spatial_dims)

        x = self.conv_depthwise(x)

        TB, *spatial_dims = x.shape
        x = x.view(T , B, *spatial_dims).contiguous() 
        return x
############## Separable Conv Synapse #######################################
############## Separable Conv Synapse #######################################
############## Separable Conv Synapse #######################################
    

