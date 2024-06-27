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

import math

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

        # spike_input = spike_input.sum(axis=0)
        spike_input = spike_input.mean(axis=0)
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
    classifier_making = False
    for which in cfg:
        if (classifier_making == False):
            if type(which) == list:
                # residual block 
                layer = ResidualBlock_conv(which, in_channels, img_size_var,
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
                # print('\n\n\nimg_size_var !!!', img_size_var, 'layer.img_size_var', layer.img_size_var, 'which', which,'\n\n\n')
                assert img_size_var == layer.img_size_var, 'pre-residu, post-residu img_size_var should be same'
                img_size_var = layer.img_size_var
                layers.append( layer)
            elif which == 'P':
                layers += [DimChanger_for_pooling(nn.AvgPool2d(kernel_size=2, stride=2))]
                img_size_var = img_size_var // 2
            elif which == 'M':
                layers += [DimChanger_for_pooling(nn.MaxPool2d(kernel_size=2, stride=2))]
                img_size_var = img_size_var // 2
            elif which == 'L':
                classifier_making = True
                layers += [DimChanger_for_FC()]
                in_channels = in_channels*img_size_var*img_size_var
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
        else: # classifier_making
            if (BPTT_on == False):
                layers += [SYNAPSE_FC(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
                                                out_features=which, 
                                                trace_const1=synapse_conv_trace_const1, 
                                                trace_const2=synapse_conv_trace_const2,
                                                TIME=TIME)]
            else:
                layers += [SYNAPSE_FC_BPTT(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
                                                out_features=which, 
                                                trace_const1=synapse_conv_trace_const1, 
                                                trace_const2=synapse_conv_trace_const2,
                                                TIME=TIME)]
            in_channels = which
                

    if classifier_making == False:
        layers += [DimChanger_for_FC()]
        in_channels = in_channels*img_size_var*img_size_var

    if (BPTT_on == False):
        layers += [SYNAPSE_FC(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
                                        out_features=synapse_fc_out_features, 
                                        trace_const1=synapse_conv_trace_const1, 
                                        trace_const2=synapse_conv_trace_const2,
                                        TIME=TIME)]
    else:
        layers += [SYNAPSE_FC_BPTT(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
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
            img_size_var = img_size_var // 2
        elif which == 'M':
            layers += [DimChanger_for_pooling(nn.MaxPool2d(kernel_size=2, stride=2))]
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

        spike_input = spike_input.mean(axis=0)

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





## from NDA paper code #############


class VGG(nn.Module):

    def __init__(self, cfg, num_classes=10, batch_norm=True, in_c=3,
                 lif_layer_v_threshold=0.5, lif_layer_v_decay=0.25, lif_layer_sg_width=1.0):
        super(VGG, self).__init__()

        self.features, out_c = make_layers_nda(cfg, batch_norm, in_c, lif_layer_v_threshold, lif_layer_v_decay, lif_layer_sg_width)
        self.avgpool = SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(
            SeqToANNContainer(nn.Linear(out_c, num_classes)),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d): #정규화
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
                
    def add_dimension(self, x):
        return add_dimention(x, self.T)
    
    
    def forward(self, x):
        # print('1    ',x.size())
        x = self.add_dim(x) if len(x.shape) == 4 else x
        # print('2      ',x.size())

        x = self.features(x)
        # print('3        ',x.size())
        x = self.avgpool(x)
        # print('4     ',x.size())
        x = torch.flatten(x, 1) if len(x.shape) == 4 else torch.flatten(x, 2)
        # print('5      ',x.size())
        x = self.classifier(x)
        # print('6        ',x.size())
        x = x.mean(axis=1)
        return x


def make_layers_nda(cfg, batch_norm=True, in_c=3, lif_layer_v_threshold = 0.5, lif_layer_v_decay = 0.25, lif_layer_sg_width = 1.0):
    layers = []
    in_channels = in_c
    i = 0
    for v in cfg:
        # avgpool이면 H,W절반, conv면 H,W유지.  
        # print('i', i, 'v', v)
        i+=1
        if v == 'P':
            layers += [SpikeModule(nn.AvgPool2d(kernel_size=2, stride=2))]
        elif v == 'M':
            layers += [SpikeModule(nn.MaxPool2d(kernel_size=2, stride=2))]
        else:
            conv2d = SpikeModule(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))

            lif = LIFSpike(lif_layer_v_threshold = 0.5, lif_layer_v_decay = 0.25, lif_layer_sg_width = 1.0) # 이거 걍 **lif_parameters에 아무것도 없어도 default값으로 알아서 됨.
            # print('lif파라메타' , **lif_parameters)
            # print('\n\n')

            if batch_norm:
                bn = tdBatchNorm(v)
                layers += [conv2d, bn, lif]
            else:
                layers += [conv2d, lif]

            in_channels = v

    return nn.Sequential(*layers), in_channels



class SeqToANNContainer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        # print('x_seq',x_seq.size())
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        # print('y_shape',y_shape)
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        # print('y_seq',y_seq.size())
        y_shape.extend(y_seq.shape[1:])
        # print('y_shape',y_shape)

        # print('y_seq.view(y_shape)',y_seq.view(y_shape).size())
        return y_seq.view(y_shape)


class SpikeModule(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.ann_module = module

    def forward(self, x):
        B, T, *spatial_dims = x.shape
        a = x.shape
        out = self.ann_module(x.reshape(B * T, *spatial_dims))
        b = out.shape

        # self.ann_module이 maxpool일 수도 있고 conv일 수도 있음.
        # print('\n\n this is spikemodule')
        # print(a, b)
        # torch.Size([64, 10, 64, 48, 48]) torch.Size([640, 64, 24, 24])
        # torch.Size([64, 10, 2, 48, 48]) torch.Size([640, 64, 48, 48])

        
        # if (a[-1] != b[-1]) :
        #     print(a, b)
        
        # print(x[0][0][0])

        # 여기 BT --> B * T로 고쳐줬음
        # 여기 걍 뭐가 이상함.
        # 왜 밑에 줄은 BT이고,
        # 밑 밑 줄은 왜 B,T이지?
        BT, *spatial_dims = out.shape
        out = out.view(B , T, *spatial_dims).contiguous() # 요소들을 정렬시켜줌.
        return out


def fire_function(gamma):
    class ZIF(torch.autograd.Function): # zero is firing
        @staticmethod
        def forward(ctx, input):
            out = (input >= 0).float()
            # gradient를 위해 input을 저장하는 코드인듯 ㅇㅇ
            # 예의주시해봐
            ctx.save_for_backward(input)
            return out


        # 걍 근데 이건 1/2보다 작으면 1
        # 1/2보다 크면 0인데?
        @staticmethod
        def backward(ctx, grad_output):
            # forward에서 저장해놨던 input가져오는거임
            (input, ) = ctx.saved_tensors
            grad_input = grad_output.clone()
            tmp = (input.abs() < gamma/2).float() / gamma
            # 사각형 형태의 surrogate gradient임.
            # 1/2 0    ----
            # -1/2 0   |  |
            # 1/2 1    ----
            # -1/2 1
            grad_input = grad_input * tmp
            return grad_input, None

    return ZIF.apply


class LIFSpike(nn.Module):
    def __init__(self, lif_layer_v_threshold = 0.5, lif_layer_v_decay = 0.25, lif_layer_sg_width = 1.0):
        super(LIFSpike, self).__init__()
        self.thresh = lif_layer_v_threshold
        self.tau = lif_layer_v_decay
        self.gamma = lif_layer_sg_width

    def forward(self, x):
        mem = torch.zeros_like(x[:, 0])
        # print('\n\nmem size', mem.size())
        # print(x)
        # print(x[0])
        # print(x[0][0])
        # print(x[0][0][0])
        # print('xsize!!',x.size())

        # mem size torch.Size([64, 512, 6, 6])
        # xsize!! torch.Size([64, 10, 512, 6, 6])

        spikes = []
        T = x.shape[1]
        for t in range(T):
            mem = mem * self.tau + x[:, t, ...] #걍 인덱스별로 각각 덧셈
            spike = fire_function(self.gamma)(mem - self.thresh)
            mem = (1 - spike) * mem #spike나감과 동시에 reset
            spikes.append(spike)

        # print('spikes size',spikes.size())
        # print('torch.stack(spikes,dim=1)', torch.stack(spikes, dim=1).size())
            
        # print('xsize22222!!',torch.stack(spikes, dim=1).size())
        
        return torch.stack(spikes, dim=1)



#     tensor.clone()	새롭게 할당	계산 그래프에 계속 상주
# tensor.detach()	공유해서 사용	계산 그래프에 상주하지 않음
# tensor.clone().detach()	새롭게 할당	계산 그래프에 상주하지 않음

def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(1, T, 1, 1, 1)
    # T= 10 시계열 데이터 추가
    return x


class tdBatchNorm(nn.BatchNorm2d):
    def __init__(self, channel):
        super(tdBatchNorm, self).__init__(channel)
        # according to tdBN paper, the initialized weight is changed to alpha*Vth
        self.weight.data.mul_(0.5)

    def forward(self, x):
        # print('tdBN - 1', x.size())
        B, T, *spatial_dims = x.shape
        out = super().forward(x.reshape(B * T, *spatial_dims))
        # print('tdBN - 3', out.size())
        BT, *spatial_dims = out.shape
        out = out.view(B, T, *spatial_dims).contiguous()
        # print('tdBN - 5', out.size())
        return out
    
