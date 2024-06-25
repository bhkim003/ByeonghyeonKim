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


##### Several functions for model implementation #########################
##### Several functions for model implementation #########################
##### Several functions for model implementation #########################
class DimChanger_for_pooling(nn.Module):
    def __init__(self, module):
        super(DimChanger_for_pooling, self).__init__()
        self.ann_module = module

    def forward(self, x):
        timestep, batch_size, *dim = x.shape
        output = self.ann_module(x.reshape(timestep * batch_size, *dim))
        _, *dim = output.shape
        output = output.view(timestep, batch_size, *dim).contiguous()
        return output


class DimChanger_for_FC(nn.Module):
    def __init__(self):
        super(DimChanger_for_FC, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        return x

class tdBatchNorm(nn.BatchNorm2d):
    def __init__(self, channel):
        super(tdBatchNorm, self).__init__(channel)
        # according to tdBN paper, the initialized weight is changed to alpha*Vth
        # self.weight.data.mul_(0.5)

    def forward(self, x):
        T, B, *spatial_dims = x.shape
        out = super().forward(x.reshape(T * B, *spatial_dims))
        TB, *spatial_dims = out.shape
        out = out.view(T, B, *spatial_dims).contiguous()
        return out
    
    
class tdBatchNorm_FC(nn.BatchNorm1d):
    def __init__(self, channel):
        super(tdBatchNorm_FC, self).__init__(channel)
        # according to tdBN paper, the initialized weight is changed to alpha*Vth
        # self.weight.data.mul_(0.5)

    def forward(self, x):
        T, B, *spatial_dims = x.shape
        out = super().forward(x.reshape(T * B, *spatial_dims))
        TB, *spatial_dims = out.shape
        out = out.view(T, B, *spatial_dims).contiguous()
        return out


class BatchNorm(nn.Module):
    def __init__(self, out_channels, TIME):
        super(BatchNorm, self).__init__()
        self.out_channels = out_channels
        self.TIME = TIME
        self.bn_layers = nn.ModuleList([nn.BatchNorm2d(self.out_channels) for _ in range(self.TIME)])

    def forward(self, x):
        # out = torch.zeros_like(x, device=x.device) #Time, Batch, Channel, Height, Width
        out = [] #Time, Batch, Channel, Height, Width
        for t in range(self.TIME):
            out.append(self.bn_layers[t](x[t]))
        out = torch.stack(out, dim=0)
        return out
    
class BatchNorm_FC(nn.Module):
    def __init__(self, out_channels, TIME):
        super(BatchNorm_FC, self).__init__()
        self.out_channels = out_channels
        self.TIME = TIME
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(self.out_channels) for _ in range(self.TIME)])

    def forward(self, x):
        # out = torch.zeros_like(x, device=x.device) #Time, Batch, Channel, Height, Width
        out = [] #Time, Batch, Channel, Height, Width
        for t in range(self.TIME):
            out.append(self.bn_layers[t](x[t]))
        out = torch.stack(out, dim=0)
        return out
##### Several functions for model implementation #########################
##### Several functions for model implementation #########################
##### Several functions for model implementation #########################




######## make_layers for Conv ############################################
######## make_layers for Conv ############################################
######## make_layers for Conv ############################################
class MY_SNN_CONV(nn.Module):
    def __init__(self, cfg, in_c, IMAGE_SIZE,
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
                     BPTT_on):
        super(MY_SNN_CONV, self).__init__()
        self.layers = make_layers_conv(cfg, in_c, IMAGE_SIZE,
                                    synapse_conv_kernel_size, synapse_conv_stride, 
                                    synapse_conv_padding, synapse_conv_trace_const1, 
                                    synapse_conv_trace_const2, 
                                    lif_layer_v_init, lif_layer_v_decay, 
                                    lif_layer_v_threshold, lif_layer_v_reset,
                                    lif_layer_sg_width,
                                    tdBN_on,
                                    BN_on, TIME,
                                    surrogate,
                                    BPTT_on,
                                    synapse_fc_out_features)


    def forward(self, spike_input):
        # inputs: [Batch, Time, Channel, Height, Width]   
        spike_input = spike_input.permute(1, 0, 2, 3, 4)
        # inputs: [Time, Batch, Channel, Height, Width]   
        spike_input = self.layers(spike_input)

        spike_input = spike_input.sum(axis=0)
        return spike_input
    

def make_layers_conv(cfg, in_c, IMAGE_SIZE,
                     synapse_conv_kernel_size, synapse_conv_stride, 
                     synapse_conv_padding, synapse_conv_trace_const1, 
                     synapse_conv_trace_const2, 
                     lif_layer_v_init, lif_layer_v_decay, 
                     lif_layer_v_threshold, lif_layer_v_reset,
                     lif_layer_sg_width,
                     tdBN_on,
                     BN_on, TIME,
                     surrogate,
                     BPTT_on,
                     synapse_fc_out_features):
    
    layers = []
    in_channels = in_c
    img_size_var = IMAGE_SIZE
    for which in cfg:
        if type(which) == list:
            # residual block 
            layer = ResidualBlock_conv(which, in_channels, IMAGE_SIZE,
                     synapse_conv_kernel_size, synapse_conv_stride, 
                     synapse_conv_padding, synapse_conv_trace_const1, 
                     synapse_conv_trace_const2, 
                     lif_layer_v_init, lif_layer_v_decay, 
                     lif_layer_v_threshold, lif_layer_v_reset,
                     lif_layer_sg_width,
                     tdBN_on,
                     BN_on, TIME,
                     surrogate,
                     BPTT_on,
                     synapse_fc_out_features)
            assert in_channels == layer.in_channels, 'pre-residu, post-residu channel should be same'
            in_channels = layer.in_channels
            assert img_size_var == layer.img_size_var, 'pre-residu, post-residu img_size_var should be same'
            img_size_var = layer.img_size_var
            layers.append( layer)
        elif which == 'P':
            layers += [DimChanger_for_pooling(nn.AvgPool2d(kernel_size=2, stride=2))]
            # layers += [DimChanger_for_pooling(nn.MaxPool2d(kernel_size=2, stride=2))]
            img_size_var = img_size_var // 2
        else:
            out_channels = which
            if (BPTT_on == False):
                layers += [SYNAPSE_CONV(in_channels=in_channels,
                                        out_channels=out_channels, 
                                        kernel_size=synapse_conv_kernel_size, 
                                        stride=synapse_conv_stride, 
                                        padding=synapse_conv_padding, 
                                        trace_const1=synapse_conv_trace_const1, 
                                        trace_const2=synapse_conv_trace_const2,
                                        TIME=TIME)]
            else:
                layers += [SYNAPSE_CONV_BPTT(in_channels=in_channels,
                                        out_channels=out_channels, 
                                        kernel_size=synapse_conv_kernel_size, 
                                        stride=synapse_conv_stride, 
                                        padding=synapse_conv_padding, 
                                        trace_const1=synapse_conv_trace_const1, 
                                        trace_const2=synapse_conv_trace_const2,
                                        TIME=TIME)]
            
            img_size_var = (img_size_var - synapse_conv_kernel_size + 2*synapse_conv_padding)//synapse_conv_stride + 1
           
            in_channels = which
            
            if (tdBN_on == True):
                layers += [tdBatchNorm(in_channels)] # 여기서 in_channel이 out_channel임

            if (BN_on == True):
                layers += [BatchNorm(in_channels, TIME)]

            layers += [LIF_layer(v_init=lif_layer_v_init, 
                                    v_decay=lif_layer_v_decay, 
                                    v_threshold=lif_layer_v_threshold, 
                                    v_reset=lif_layer_v_reset, 
                                    sg_width=lif_layer_sg_width,
                                    surrogate=surrogate,
                                    BPTT_on=BPTT_on)]
            
    layers += [DimChanger_for_FC()]
    if (BPTT_on == False):
        layers += [SYNAPSE_FC(in_features=in_channels*img_size_var*img_size_var,  # 마지막CONV의 OUT_CHANNEL * H * W
                                        out_features=synapse_fc_out_features, 
                                        trace_const1=synapse_conv_trace_const1, 
                                        trace_const2=synapse_conv_trace_const2,
                                        TIME=TIME)]
    else:
        layers += [SYNAPSE_FC_BPTT(in_features=in_channels*img_size_var*img_size_var,  # 마지막CONV의 OUT_CHANNEL * H * W
                                        out_features=synapse_fc_out_features, 
                                        trace_const1=synapse_conv_trace_const1, 
                                        trace_const2=synapse_conv_trace_const2,
                                        TIME=TIME)]

    return nn.Sequential(*layers)



class ResidualBlock_conv(nn.Module):
    def __init__(self, layers, in_c, IMAGE_SIZE,
                     synapse_conv_kernel_size, synapse_conv_stride, 
                     synapse_conv_padding, synapse_conv_trace_const1, 
                     synapse_conv_trace_const2, 
                     lif_layer_v_init, lif_layer_v_decay, 
                     lif_layer_v_threshold, lif_layer_v_reset,
                     lif_layer_sg_width,
                     tdBN_on,
                     BN_on, TIME,
                     surrogate,
                     BPTT_on,
                     synapse_fc_out_features):
        super(ResidualBlock_conv, self).__init__()
        self.layers, self.in_channels, self.img_size_var= make_layers_conv_residual(layers, in_c, IMAGE_SIZE,
                     synapse_conv_kernel_size, synapse_conv_stride, 
                     synapse_conv_padding, synapse_conv_trace_const1, 
                     synapse_conv_trace_const2, 
                     lif_layer_v_init, lif_layer_v_decay, 
                     lif_layer_v_threshold, lif_layer_v_reset,
                     lif_layer_sg_width,
                     tdBN_on,
                     BN_on, TIME,
                     surrogate,
                     BPTT_on,
                     synapse_fc_out_features)
    
    def forward(self, x):
        return self.layers(x) + x
     

def make_layers_conv_residual(cfg, in_c, IMAGE_SIZE,
                     synapse_conv_kernel_size, synapse_conv_stride, 
                     synapse_conv_padding, synapse_conv_trace_const1, 
                     synapse_conv_trace_const2, 
                     lif_layer_v_init, lif_layer_v_decay, 
                     lif_layer_v_threshold, lif_layer_v_reset,
                     lif_layer_sg_width,
                     tdBN_on,
                     BN_on, TIME,
                     surrogate,
                     BPTT_on,
                     synapse_fc_out_features):
    
    layers = []
    in_channels = in_c
    img_size_var = IMAGE_SIZE
    for which in cfg:
        if which == 'P':
            layers += [DimChanger_for_pooling(nn.AvgPool2d(kernel_size=2, stride=2))]
            # layers += [DimChanger_for_pooling(nn.MaxPool2d(kernel_size=2, stride=2))]
            img_size_var = img_size_var // 2
        else:
            out_channels = which
            if (BPTT_on == False):
                layers += [SYNAPSE_CONV(in_channels=in_channels,
                                        out_channels=out_channels, 
                                        kernel_size=synapse_conv_kernel_size, 
                                        stride=synapse_conv_stride, 
                                        padding=synapse_conv_padding, 
                                        trace_const1=synapse_conv_trace_const1, 
                                        trace_const2=synapse_conv_trace_const2,
                                        TIME=TIME)]
            else:
                layers += [SYNAPSE_CONV_BPTT(in_channels=in_channels,
                                        out_channels=out_channels, 
                                        kernel_size=synapse_conv_kernel_size, 
                                        stride=synapse_conv_stride, 
                                        padding=synapse_conv_padding, 
                                        trace_const1=synapse_conv_trace_const1, 
                                        trace_const2=synapse_conv_trace_const2,
                                        TIME=TIME)]
            
            img_size_var = (img_size_var - synapse_conv_kernel_size + 2*synapse_conv_padding)//synapse_conv_stride + 1
           
            in_channels = which
            
            if (tdBN_on == True):
                layers += [tdBatchNorm(in_channels)] # 여기서 in_channel이 out_channel임

            if (BN_on == True):
                layers += [BatchNorm(in_channels, TIME)]

            layers += [LIF_layer(v_init=lif_layer_v_init, 
                                    v_decay=lif_layer_v_decay, 
                                    v_threshold=lif_layer_v_threshold, 
                                    v_reset=lif_layer_v_reset, 
                                    sg_width=lif_layer_sg_width,
                                    surrogate=surrogate,
                                    BPTT_on=BPTT_on)]
    return nn.Sequential(*layers), in_channels, img_size_var
######## make_layers for Conv ############################################
######## make_layers for Conv ############################################
######## make_layers for Conv ############################################






######## make_layers for FC ############################################
######## make_layers for FC ############################################
######## make_layers for FC ############################################
class MY_SNN_FC(nn.Module):
    def __init__(self, cfg, in_c, IMAGE_SIZE, out_c,
                     synapse_fc_trace_const1, synapse_fc_trace_const2, 
                     lif_layer_v_init, lif_layer_v_decay, 
                     lif_layer_v_threshold, lif_layer_v_reset,
                     lif_layer_sg_width,
                     tdBN_on,
                     BN_on, TIME,
                     surrogate,
                     BPTT_on):
        super(MY_SNN_FC, self).__init__()

        self.layers = make_layers_fc(cfg, in_c, IMAGE_SIZE, out_c,
                     synapse_fc_trace_const1, synapse_fc_trace_const2, 
                     lif_layer_v_init, lif_layer_v_decay, 
                     lif_layer_v_threshold, lif_layer_v_reset,
                     lif_layer_sg_width,
                     tdBN_on,
                     BN_on, TIME,
                     surrogate,
                     BPTT_on)

    def forward(self, spike_input):
        # inputs: [Batch, Time, Channel, Height, Width]   
        spike_input = spike_input.permute(1, 0, 2, 3, 4)
        # inputs: [Time, Batch, Channel, Height, Width]   
        spike_input = spike_input.view(spike_input.size(0), spike_input.size(1), -1)
        
        spike_input = self.layers(spike_input)

        spike_input = spike_input.sum(axis=0)

        return spike_input
    

def make_layers_fc(cfg, in_c, IMAGE_SIZE, out_c,
                     synapse_fc_trace_const1, synapse_fc_trace_const2, 
                     lif_layer_v_init, lif_layer_v_decay, 
                     lif_layer_v_threshold, lif_layer_v_reset,
                     lif_layer_sg_width,
                     tdBN_on,
                     BN_on, TIME,
                     surrogate,
                     BPTT_on):

    layers = []
    img_size = IMAGE_SIZE
    in_channels = in_c * img_size * img_size
    class_num = out_c
    for which in cfg:
        if type(which) == list:
            # residual block 
            layer = ResidualBlock_fc(which, in_channels, IMAGE_SIZE, out_c,
                     synapse_fc_trace_const1, synapse_fc_trace_const2, 
                     lif_layer_v_init, lif_layer_v_decay, 
                     lif_layer_v_threshold, lif_layer_v_reset,
                     lif_layer_sg_width,
                     tdBN_on,
                     BN_on, TIME,
                     surrogate,
                     BPTT_on)
            
            assert in_channels == layer.in_channels, 'pre-residu, post-residu channel should be same'
            in_channels = layer.in_channels
            layers.append( layer)
        else:
            out_channels = which
            if(BPTT_on == False):
                layers += [SYNAPSE_FC(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
                                            out_features=out_channels, 
                                            trace_const1=synapse_fc_trace_const1, 
                                            trace_const2=synapse_fc_trace_const2,
                                            TIME=TIME)]
            else:
                layers += [SYNAPSE_FC_BPTT(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
                                            out_features=out_channels, 
                                            trace_const1=synapse_fc_trace_const1, 
                                            trace_const2=synapse_fc_trace_const2,
                                            TIME=TIME)]
            in_channels = which
        
        if (tdBN_on == True):
            layers += [tdBatchNorm_FC(in_channels)] # 여기서 in_channel이 out_channel임

        if (BN_on == True):
            layers += [BatchNorm_FC(in_channels, TIME)]

        layers += [LIF_layer(v_init=lif_layer_v_init, 
                                v_decay=lif_layer_v_decay, 
                                v_threshold=lif_layer_v_threshold, 
                                v_reset=lif_layer_v_reset, 
                                sg_width=lif_layer_sg_width,
                                surrogate=surrogate,
                                BPTT_on=BPTT_on)]

    
    out_channels = class_num
    if(BPTT_on == False):
        layers += [SYNAPSE_FC(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
                                    out_features=out_channels, 
                                    trace_const1=synapse_fc_trace_const1, 
                                    trace_const2=synapse_fc_trace_const2,
                                    TIME=TIME)]
    else:
        layers += [SYNAPSE_FC_BPTT(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
                                    out_features=out_channels, 
                                    trace_const1=synapse_fc_trace_const1, 
                                    trace_const2=synapse_fc_trace_const2,
                                    TIME=TIME)]
        
    return nn.Sequential(*layers)



class ResidualBlock_fc(nn.Module):
    def __init__(self, layers, in_channels, IMAGE_SIZE, out_c,
                     synapse_fc_trace_const1, synapse_fc_trace_const2, 
                     lif_layer_v_init, lif_layer_v_decay, 
                     lif_layer_v_threshold, lif_layer_v_reset,
                     lif_layer_sg_width,
                     tdBN_on,
                     BN_on, TIME,
                     surrogate,
                     BPTT_on):
        super(ResidualBlock_fc, self).__init__()
        self.layers, self.in_channels = make_layers_fc_residual(layers, in_channels, IMAGE_SIZE, out_c,
                     synapse_fc_trace_const1, synapse_fc_trace_const2, 
                     lif_layer_v_init, lif_layer_v_decay, 
                     lif_layer_v_threshold, lif_layer_v_reset,
                     lif_layer_sg_width,
                     tdBN_on,
                     BN_on, TIME,
                     surrogate,
                     BPTT_on)
    
    def forward(self, x):
        return self.layers(x) + x
    


def make_layers_fc_residual(cfg, in_c, IMAGE_SIZE, out_c,
                     synapse_fc_trace_const1, synapse_fc_trace_const2, 
                     lif_layer_v_init, lif_layer_v_decay, 
                     lif_layer_v_threshold, lif_layer_v_reset,
                     lif_layer_sg_width,
                     tdBN_on,
                     BN_on, TIME,
                     surrogate,
                     BPTT_on):

    layers = []
    img_size = IMAGE_SIZE
    in_channels = in_c
    class_num = out_c
    for which in cfg:

        out_channels = which

        if(BPTT_on == False):
            layers += [SYNAPSE_FC(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
                                        out_features=out_channels, 
                                        trace_const1=synapse_fc_trace_const1, 
                                        trace_const2=synapse_fc_trace_const2,
                                        TIME=TIME)]
        else:
            layers += [SYNAPSE_FC_BPTT(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
                                        out_features=out_channels, 
                                        trace_const1=synapse_fc_trace_const1, 
                                        trace_const2=synapse_fc_trace_const2,
                                        TIME=TIME)]
            


        in_channels = which
        
        if (tdBN_on == True):
            layers += [tdBatchNorm_FC(in_channels)] # 여기서 in_channel이 out_channel임

        if (BN_on == True):
            layers += [BatchNorm_FC(in_channels, TIME)]

        layers += [LIF_layer(v_init=lif_layer_v_init, 
                                v_decay=lif_layer_v_decay, 
                                v_threshold=lif_layer_v_threshold, 
                                v_reset=lif_layer_v_reset, 
                                sg_width=lif_layer_sg_width,
                                surrogate=surrogate,
                                BPTT_on=BPTT_on)]

    return nn.Sequential(*layers), in_channels
######## make_layers for FC ############################################
######## make_layers for FC ############################################
######## make_layers for FC ############################################

