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

from torch import Tensor
from torch.nn.common_types import _size_any_t, _size_1_t, _size_2_t, _size_3_t, _ratio_any_t
from typing import Optional, List, Tuple, Union
from typing import Callable

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
class DimChanger_for_FC_sstep(nn.Module):
    def __init__(self):
        super(DimChanger_for_FC_sstep, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x
    
    
class DimChanger_for_change_0_1(nn.Module):
    def __init__(self):
        super(DimChanger_for_change_0_1, self).__init__()

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)
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
                     BPTT_on,
                     OTTT_sWS_on):
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
                                    synapse_fc_out_features,
                                    OTTT_sWS_on)
        # for i in self.layers:
        #     print(i, len(list(i.parameters())))

        # print(self.layers)

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
                     synapse_fc_out_features,
                     OTTT_sWS_on):
    
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
                        synapse_fc_out_features,
                        OTTT_sWS_on)
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
            elif which == 'D':
                layers += [DimChanger_for_pooling(nn.AdaptiveAvgPool2d((1, 1)))]
                img_size_var = 1
            elif which == 'L':
                classifier_making = True
                layers += [DimChanger_for_FC()]
                in_channels = in_channels*img_size_var*img_size_var
            else:
                if (which >= 10000 and which < 20000):
                    out_channels = which - 10000
                    layers += [SYNAPSE_SEPARABLE_CONV_BPTT(
                                            in_channels=in_channels,
                                            out_channels=out_channels, 
                                            kernel_size=synapse_conv_kernel_size, 
                                            stride=synapse_conv_stride, 
                                            padding=synapse_conv_padding, 
                                            trace_const1=synapse_conv_trace_const1, 
                                            trace_const2=synapse_conv_trace_const2,
                                            TIME=TIME)]
                elif (which >= 20000 and which < 30000):
                    out_channels = which - 20000
                    layers += [SYNAPSE_DEPTHWISE_CONV_BPTT(
                                            in_channels=in_channels,
                                            out_channels=out_channels, 
                                            kernel_size=synapse_conv_kernel_size, 
                                            stride=synapse_conv_stride, 
                                            padding=synapse_conv_padding, 
                                            trace_const1=synapse_conv_trace_const1, 
                                            trace_const2=synapse_conv_trace_const2,
                                            TIME=TIME)]
                else:
                    out_channels = which
                    if (BPTT_on == False):
                        if (layers == []):
                            first_conv = True
                        else:
                            first_conv = False
                        # layers += [SYNAPSE_CONV(in_channels=in_channels,
                        #                         out_channels=out_channels, 
                        #                         kernel_size=synapse_conv_kernel_size, 
                        #                         stride=synapse_conv_stride, 
                        #                         padding=synapse_conv_padding, 
                        #                         trace_const1=synapse_conv_trace_const1, 
                        #                         trace_const2=synapse_conv_trace_const2,
                        #                         TIME=TIME, OTTT_sWS_on=OTTT_sWS_on, first_conv=first_conv)]
                        layers += [SYNAPSE_CONV_trace(in_channels=in_channels,
                                                out_channels=out_channels, 
                                                kernel_size=synapse_conv_kernel_size, 
                                                stride=synapse_conv_stride, 
                                                padding=synapse_conv_padding, 
                                                trace_const1=synapse_conv_trace_const1, 
                                                trace_const2=synapse_conv_trace_const2,
                                                TIME=TIME, OTTT_sWS_on=OTTT_sWS_on, first_conv=first_conv)]
                        
                    else:
                        layers += [SYNAPSE_CONV_BPTT(in_channels=in_channels,
                                                out_channels=out_channels, 
                                                kernel_size=synapse_conv_kernel_size, 
                                                stride=synapse_conv_stride, 
                                                padding=synapse_conv_padding, 
                                                trace_const1=synapse_conv_trace_const1, 
                                                trace_const2=synapse_conv_trace_const2,
                                                TIME=TIME)]
                        # layers += [SpikeModule(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))]
                    
                
                img_size_var = (img_size_var - synapse_conv_kernel_size + 2*synapse_conv_padding)//synapse_conv_stride + 1
            
                in_channels = out_channels
                

                # batchnorm or tdBN 추가 ##########################
                if (tdBN_on == True):
                    layers += [tdBatchNorm(in_channels)] # 여기서 in_channel이 out_channel임

                if (BN_on == True):
                    layers += [BatchNorm(in_channels, TIME)]
                #################################################


                # LIF 뉴런 추가 ##################################
                if (lif_layer_v_threshold >= 0 and lif_layer_v_threshold < 10000):
                    # layers += [LIF_layer(v_init=lif_layer_v_init, 
                    #                         v_decay=lif_layer_v_decay, 
                    #                         v_threshold=lif_layer_v_threshold, 
                    #                         v_reset=lif_layer_v_reset, 
                    #                         sg_width=lif_layer_sg_width,
                    #                         surrogate=surrogate,
                    #                         BPTT_on=BPTT_on)]
                    layers += [LIF_layer_trace(v_init=lif_layer_v_init, 
                                            v_decay=lif_layer_v_decay, 
                                            v_threshold=lif_layer_v_threshold, 
                                            v_reset=lif_layer_v_reset, 
                                            sg_width=lif_layer_sg_width,
                                            surrogate=surrogate,
                                            BPTT_on=BPTT_on, 
                                            trace_const1=synapse_conv_trace_const1, 
                                            trace_const2=synapse_conv_trace_const2)]
                elif (lif_layer_v_threshold >= 10000 and lif_layer_v_threshold < 20000):
                    # NDA의 LIF 뉴런 쓰고 싶을 때 
                    lif_layer_v_threshold -= 10000
                    layers += [DimChanger_for_change_0_1()]
                    layers += [LIFSpike(lif_layer_v_threshold = lif_layer_v_threshold, 
                                lif_layer_v_decay = lif_layer_v_decay, lif_layer_sg_width = lif_layer_sg_width)] # 이거 걍 **lif_parameters에 아무것도 없어도 default값으로 알아서 됨.
                    layers += [DimChanger_for_change_0_1()]
                    lif_layer_v_threshold += 10000
                #################################################
                
                ## OTTT sWS하면 스케일링해줘야됨
                if OTTT_sWS_on == True:
                    layers += [Scale(2.74)]

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
                

    if classifier_making == False: # cfg에 'L'한번도 없을때
        layers += [DimChanger_for_FC()]
        in_channels = in_channels*img_size_var*img_size_var
        
    if (BPTT_on == False):
        # layers += [SYNAPSE_FC(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
        #                                 out_features=synapse_fc_out_features, 
        #                                 trace_const1=synapse_conv_trace_const1, 
        #                                 trace_const2=synapse_conv_trace_const2,
        #                                 TIME=TIME)]
        layers += [SYNAPSE_FC_trace(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
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

    # return nn.Sequential(*layers)
    return OTTTSequential(*layers)



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
                     synapse_fc_out_features,
                     OTTT_sWS_on):
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
                     synapse_fc_out_features,
                     OTTT_sWS_on)
    
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
                     synapse_fc_out_features,
                     OTTT_sWS_on):
    
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
        elif which == 'D':
            layers += [DimChanger_for_pooling(nn.AdaptiveAvgPool2d((1, 1)))]
            img_size_var = 1
        else:
            if (which >= 10000 and which < 20000):
                out_channels = which - 10000
                layers += [SYNAPSE_SEPARABLE_CONV_BPTT(
                                        in_channels=in_channels,
                                        out_channels=out_channels, 
                                        kernel_size=synapse_conv_kernel_size, 
                                        stride=synapse_conv_stride, 
                                        padding=synapse_conv_padding, 
                                        trace_const1=synapse_conv_trace_const1, 
                                        trace_const2=synapse_conv_trace_const2,
                                        TIME=TIME)]
                
            elif (which >= 20000 and which < 30000):
                out_channels = which - 20000
                layers += [SYNAPSE_DEPTHWISE_CONV_BPTT(
                                        in_channels=in_channels,
                                        out_channels=out_channels, 
                                        kernel_size=synapse_conv_kernel_size, 
                                        stride=synapse_conv_stride, 
                                        padding=synapse_conv_padding, 
                                        trace_const1=synapse_conv_trace_const1, 
                                        trace_const2=synapse_conv_trace_const2,
                                        TIME=TIME)]
            
            
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
                                                TIME=TIME, OTTT_sWS_on=OTTT_sWS_on, first_conv=False)]
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
           
            in_channels = out_channels
            
            if (tdBN_on == True):
                layers += [tdBatchNorm(in_channels)] # 여기서 in_channel이 out_channel임

            if (BN_on == True):
                layers += [BatchNorm(in_channels, TIME)]

            # LIF 뉴런 추가 ##################################
            if (lif_layer_v_threshold >= 0 and lif_layer_v_threshold < 10000):
                layers += [LIF_layer(v_init=lif_layer_v_init, 
                                        v_decay=lif_layer_v_decay, 
                                        v_threshold=lif_layer_v_threshold, 
                                        v_reset=lif_layer_v_reset, 
                                        sg_width=lif_layer_sg_width,
                                        surrogate=surrogate,
                                        BPTT_on=BPTT_on)]
            elif (lif_layer_v_threshold >= 10000 and lif_layer_v_threshold < 20000):
                # NDA의 LIF 뉴런 쓰고 싶을 때 
                lif_layer_v_threshold -= 10000
                layers += [DimChanger_for_change_0_1()]
                layers += [LIFSpike(lif_layer_v_threshold = lif_layer_v_threshold, 
                            lif_layer_v_decay = lif_layer_v_decay, lif_layer_sg_width = lif_layer_sg_width)] # 이거 걍 **lif_parameters에 아무것도 없어도 default값으로 알아서 됨.
                layers += [DimChanger_for_change_0_1()]
                lif_layer_v_threshold += 10000
            #################################################

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
        # spike_input = spike_input.sum(axis=0)

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
                # layers += [SYNAPSE_FC(in_features=in_channels, 
                #                             out_features=out_channels, 
                #                             trace_const1=synapse_fc_trace_const1, 
                #                             trace_const2=synapse_fc_trace_const2,
                #                             TIME=TIME)]
                layers += [SYNAPSE_FC_trace(in_features=in_channels, 
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


        # LIF 뉴런 추가 ##################################
        if (lif_layer_v_threshold >= 0 and lif_layer_v_threshold < 10000):
            layers += [LIF_layer(v_init=lif_layer_v_init, 
                                    v_decay=lif_layer_v_decay, 
                                    v_threshold=lif_layer_v_threshold, 
                                    v_reset=lif_layer_v_reset, 
                                    sg_width=lif_layer_sg_width,
                                    surrogate=surrogate,
                                    BPTT_on=BPTT_on)]
        elif (lif_layer_v_threshold >= 10000 and lif_layer_v_threshold < 20000):
            # NDA의 LIF 뉴런 쓰고 싶을 때 
            lif_layer_v_threshold -= 10000
            layers += [DimChanger_for_change_0_1()]
            layers += [LIFSpike(lif_layer_v_threshold = lif_layer_v_threshold, 
                        lif_layer_v_decay = lif_layer_v_decay, lif_layer_sg_width = lif_layer_sg_width)] # 이거 걍 **lif_parameters에 아무것도 없어도 default값으로 알아서 됨.
            layers += [DimChanger_for_change_0_1()]
            lif_layer_v_threshold += 10000
        #################################################

    
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

        

        # LIF 뉴런 추가 ##################################
        if (lif_layer_v_threshold >= 0 and lif_layer_v_threshold < 10000):
            layers += [LIF_layer(v_init=lif_layer_v_init, 
                                    v_decay=lif_layer_v_decay, 
                                    v_threshold=lif_layer_v_threshold, 
                                    v_reset=lif_layer_v_reset, 
                                    sg_width=lif_layer_sg_width,
                                    surrogate=surrogate,
                                    BPTT_on=BPTT_on)]
        elif (lif_layer_v_threshold >= 10000 and lif_layer_v_threshold < 20000):
            # NDA의 LIF 뉴런 쓰고 싶을 때 
            lif_layer_v_threshold -= 10000
            layers += [DimChanger_for_change_0_1()]
            layers += [LIFSpike(lif_layer_v_threshold = lif_layer_v_threshold, 
                        lif_layer_v_decay = lif_layer_v_decay, lif_layer_sg_width = lif_layer_sg_width)] # 이거 걍 **lif_parameters에 아무것도 없어도 default값으로 알아서 됨.
            layers += [DimChanger_for_change_0_1()]
            lif_layer_v_threshold += 10000
        #################################################

    return nn.Sequential(*layers), in_channels
######## make_layers for FC ############################################
######## make_layers for FC ############################################
######## make_layers for FC ############################################






####### make_layers for ottt conv single step ############################################
####### make_layers for ottt conv single step ############################################
####### make_layers for ottt conv single step ############################################
class MY_SNN_CONV_ottt_sstep(nn.Module):
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
                     BPTT_on,
                     OTTT_sWS_on):
        super(MY_SNN_CONV_ottt_sstep, self).__init__()
        self.layers = make_layers_conv_ottt_sstep(cfg, in_c, IMAGE_SIZE,
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
                                    synapse_fc_out_features,
                                    OTTT_sWS_on)
        # for i in self.layers:
        #     print(i, len(list(i.parameters())))

        # print(self.layers)

    def forward(self, spike_input):
        # inputs: [Batch, Channel, Height, Width]   
        spike_input = self.layers(spike_input)
        # spike_input = spike_input.sum(axis=0)
        # spike_input = spike_input.mean(axis=0)
        return spike_input
    

def make_layers_conv_ottt_sstep(cfg, in_c, IMAGE_SIZE,
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
                     synapse_fc_out_features,
                     OTTT_sWS_on):
    
    layers = []
    in_channels = in_c
    img_size_var = IMAGE_SIZE
    classifier_making = False
    for which in cfg:
        if (classifier_making == False):
            if type(which) == list:
                assert False, 'not implemented'
                # # residual block 
                # layer = ResidualBlock_conv(which, in_channels, img_size_var,
                #         synapse_conv_kernel_size, synapse_conv_stride, 
                #         synapse_conv_padding, synapse_conv_trace_const1, 
                #         synapse_conv_trace_const2, 
                #         lif_layer_v_init, lif_layer_v_decay, 
                #         lif_layer_v_threshold, lif_layer_v_reset,
                #         lif_layer_sg_width,
                #         tdBN_on,
                #         BN_on, TIME,
                #         surrogate,
                #         BPTT_on,
                #         synapse_fc_out_features,
                #         OTTT_sWS_on)
                # assert in_channels == layer.in_channels, 'pre-residu, post-residu channel should be same'
                # in_channels = layer.in_channels
                # # print('\n\n\nimg_size_var !!!', img_size_var, 'layer.img_size_var', layer.img_size_var, 'which', which,'\n\n\n')
                # assert img_size_var == layer.img_size_var, 'pre-residu, post-residu img_size_var should be same'
                # img_size_var = layer.img_size_var
                # layers.append( layer)
            elif which == 'P':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
                img_size_var = img_size_var // 2
            elif which == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                img_size_var = img_size_var // 2
            elif which == 'D':
                layers += [nn.AdaptiveAvgPool2d((1, 1))]
                img_size_var = 1
            elif which == 'L':
                assert False, 'not implemented'
                # classifier_making = True
                # layers += [DimChanger_for_FC_sstep()]
                # in_channels = in_channels*img_size_var*img_size_var
            else:
                if (which >= 10000 and which < 20000):
                    assert False, 'not implemented'
                elif (which >= 20000 and which < 30000):
                    assert False, 'not implemented'
                else:
                    out_channels = which
                    if (BPTT_on == False):
                        if (layers == []):
                            first_conv = True
                        else:
                            first_conv = False
                        # layers += [SYNAPSE_CONV(in_channels=in_channels,
                        #                         out_channels=out_channels, 
                        #                         kernel_size=synapse_conv_kernel_size, 
                        #                         stride=synapse_conv_stride, 
                        #                         padding=synapse_conv_padding, 
                        #                         trace_const1=synapse_conv_trace_const1, 
                        #                         trace_const2=synapse_conv_trace_const2,
                        #                         TIME=TIME, OTTT_sWS_on=OTTT_sWS_on, first_conv=first_conv)]
                        # layers += [SYNAPSE_CONV_trace(in_channels=in_channels,
                        #                         out_channels=out_channels, 
                        #                         kernel_size=synapse_conv_kernel_size, 
                        #                         stride=synapse_conv_stride, 
                        #                         padding=synapse_conv_padding, 
                        #                         trace_const1=synapse_conv_trace_const1, 
                        #                         trace_const2=synapse_conv_trace_const2,
                        #                         TIME=TIME, OTTT_sWS_on=OTTT_sWS_on, first_conv=first_conv)]
                        layers += [SYNAPSE_CONV_trace_sstep(in_channels=in_channels,
                                                out_channels=out_channels, 
                                                kernel_size=synapse_conv_kernel_size, 
                                                stride=synapse_conv_stride, 
                                                padding=synapse_conv_padding, 
                                                trace_const1=synapse_conv_trace_const1, 
                                                trace_const2=synapse_conv_trace_const2,
                                                TIME=TIME, OTTT_sWS_on=OTTT_sWS_on, first_conv=first_conv)]
                        
                    else:
                        assert False, 'not implemented'
                        # layers += [SYNAPSE_CONV_BPTT(in_channels=in_channels,
                        #                         out_channels=out_channels, 
                        #                         kernel_size=synapse_conv_kernel_size, 
                        #                         stride=synapse_conv_stride, 
                        #                         padding=synapse_conv_padding, 
                        #                         trace_const1=synapse_conv_trace_const1, 
                        #                         trace_const2=synapse_conv_trace_const2,
                        #                         TIME=TIME)]
                        # layers += [SpikeModule(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))]
                    
                
                img_size_var = (img_size_var - synapse_conv_kernel_size + 2*synapse_conv_padding)//synapse_conv_stride + 1
            
                in_channels = out_channels
                

                # batchnorm or tdBN 추가 ##########################
                if (tdBN_on == True):
                    assert False, 'not implemented'
                    # layers += [tdBatchNorm(in_channels)] # 여기서 in_channel이 out_channel임

                if (BN_on == True):
                    assert False, 'not implemented'
                    # layers += [BatchNorm(in_channels, TIME)]
                #################################################


                # LIF 뉴런 추가 ##################################
                if (lif_layer_v_threshold >= 0 and lif_layer_v_threshold < 10000):
                    # layers += [LIF_layer(v_init=lif_layer_v_init, 
                    #                         v_decay=lif_layer_v_decay, 
                    #                         v_threshold=lif_layer_v_threshold, 
                    #                         v_reset=lif_layer_v_reset, 
                    #                         sg_width=lif_layer_sg_width,
                    #                         surrogate=surrogate,
                    #                         BPTT_on=BPTT_on)]
                    # layers += [LIF_layer_trace(v_init=lif_layer_v_init, 
                    #                         v_decay=lif_layer_v_decay, 
                    #                         v_threshold=lif_layer_v_threshold, 
                    #                         v_reset=lif_layer_v_reset, 
                    #                         sg_width=lif_layer_sg_width,
                    #                         surrogate=surrogate,
                    #                         BPTT_on=BPTT_on, 
                    #                         trace_const1=synapse_conv_trace_const1, 
                    #                         trace_const2=synapse_conv_trace_const2)]
                    layers += [LIF_layer_trace_sstep(v_init=lif_layer_v_init, 
                                            v_decay=lif_layer_v_decay, 
                                            v_threshold=lif_layer_v_threshold, 
                                            v_reset=lif_layer_v_reset, 
                                            sg_width=lif_layer_sg_width,
                                            surrogate=surrogate,
                                            BPTT_on=BPTT_on, 
                                            trace_const1=synapse_conv_trace_const1, 
                                            trace_const2=synapse_conv_trace_const2)]
                elif (lif_layer_v_threshold >= 10000 and lif_layer_v_threshold < 20000):
                    assert False, 'not implemented'
                    # # NDA의 LIF 뉴런 쓰고 싶을 때 
                    # lif_layer_v_threshold -= 10000
                    # layers += [DimChanger_for_change_0_1()]
                    # layers += [LIFSpike(lif_layer_v_threshold = lif_layer_v_threshold, 
                    #             lif_layer_v_decay = lif_layer_v_decay, lif_layer_sg_width = lif_layer_sg_width)] # 이거 걍 **lif_parameters에 아무것도 없어도 default값으로 알아서 됨.
                    # layers += [DimChanger_for_change_0_1()]
                    # lif_layer_v_threshold += 10000
                #################################################
                
                ## OTTT sWS하면 스케일링해줘야됨
                if OTTT_sWS_on == True:
                    layers += [Scale(2.74)]

        else: # classifier_making
            assert False, 'not implemented'
            # if (BPTT_on == False):
            #     layers += [SYNAPSE_FC(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
            #                                     out_features=which, 
            #                                     trace_const1=synapse_conv_trace_const1, 
            #                                     trace_const2=synapse_conv_trace_const2,
            #                                     TIME=TIME)]
            # else:
            #     assert False, 'not implemented'
            #     layers += [SYNAPSE_FC_BPTT(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
            #                                     out_features=which, 
            #                                     trace_const1=synapse_conv_trace_const1, 
            #                                     trace_const2=synapse_conv_trace_const2,
            #                                     TIME=TIME)]
            # in_channels = which
                

    if classifier_making == False: # cfg에 'L'한번도 없을때
        layers += [DimChanger_for_FC_sstep()]
        in_channels = in_channels*img_size_var*img_size_var
        
    if (BPTT_on == False):
        # layers += [SYNAPSE_FC(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
        #                                 out_features=synapse_fc_out_features, 
        #                                 trace_const1=synapse_conv_trace_const1, 
        #                                 trace_const2=synapse_conv_trace_const2,
        #                                 TIME=TIME)]
        # layers += [SYNAPSE_FC_trace(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
        #                                 out_features=synapse_fc_out_features, 
        #                                 trace_const1=synapse_conv_trace_const1, 
        #                                 trace_const2=synapse_conv_trace_const2,
        #                                 TIME=TIME)]
        layers += [SYNAPSE_FC_trace_sstep(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
                                        out_features=synapse_fc_out_features, 
                                        trace_const1=synapse_conv_trace_const1, 
                                        trace_const2=synapse_conv_trace_const2,
                                        TIME=TIME)]
    else:
        assert False, 'not implemented'
        # layers += [SYNAPSE_FC_BPTT(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
        #                                 out_features=synapse_fc_out_features, 
        #                                 trace_const1=synapse_conv_trace_const1, 
        #                                 trace_const2=synapse_conv_trace_const2,
        #                                 TIME=TIME)]

    # return nn.Sequential(*layers)
    return OTTTSequential(*layers)
####### make_layers for ottt conv single step ############################################
####### make_layers for ottt conv single step ############################################
####### make_layers for ottt conv single step ############################################













## from NDA paper code #####################################
## from NDA paper code #####################################
## from NDA paper code #####################################
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
        # x = x.sum(axis=1)
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
            
            layers += [SpikeModule(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))]
            
            # layers += [DimChanger_for_change_0_1()]
            # layers += [SYNAPSE_CONV_BPTT(in_channels=in_channels,
            #                                 out_channels=v, 
            #                                 kernel_size=3, 
            #                                 stride=1, 
            #                                 padding=1, 
            #                                 trace_const1=1, 
            #                                 trace_const2=lif_layer_v_decay,
            #                                 TIME=10)]
            # layers += [DimChanger_for_change_0_1()]
            
            
            

            if batch_norm:
                layers += [tdBatchNorm(v)]
            else:
                pass


            layers += [LIFSpike(lif_layer_v_threshold = 0.5, lif_layer_v_decay = 0.25, lif_layer_sg_width = 1.0)] # 이거 걍 **lif_parameters에 아무것도 없어도 default값으로 알아서 됨.
            
            # layers += [DimChanger_for_change_0_1()]
            # layers += [LIF_layer(v_init=0, 
            #             v_decay=lif_layer_v_decay, 
            #             v_threshold=lif_layer_v_threshold, 
            #             v_reset=0, 
            #             sg_width=lif_layer_sg_width,
            #             surrogate='rough_rectangle',
            #             BPTT_on=True)]
            # layers += [DimChanger_for_change_0_1()]
            

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



# 저 위에 정의 해놨음
# class tdBatchNorm(nn.BatchNorm2d):
#     def __init__(self, channel):
#         super(tdBatchNorm, self).__init__(channel)
#         # according to tdBN paper, the initialized weight is changed to alpha*Vth
#         self.weight.data.mul_(0.5)

#     def forward(self, x):
#         # print('tdBN - 1', x.size())
#         B, T, *spatial_dims = x.shape
#         out = super().forward(x.reshape(B * T, *spatial_dims))
#         # print('tdBN - 3', out.size())
#         BT, *spatial_dims = out.shape
#         out = out.view(B, T, *spatial_dims).contiguous()
#         # print('tdBN - 5', out.size())
#         return out
## from NDA paper code #####################################
## from NDA paper code #####################################
## from NDA paper code #####################################







######### OTTT & spikingjelly functions #################################################################################
######### OTTT & spikingjelly functions #################################################################################
######### OTTT & spikingjelly functions #################################################################################
class Scale(nn.Module):
    def __init__(self, scale):
        super(Scale, self).__init__()
        self.scale = scale

    def forward(self, x):
        # print(x.size())
        return x * self.scale
    
class OTTTSequential(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, input):
        for module in self:
            if not isinstance(input, list):
                input = module(input)
                # print('1', module)
            else: 
                if isinstance(module, SYNAPSE_CONV_trace) or isinstance(module, SYNAPSE_FC_trace) or isinstance(module, SYNAPSE_CONV_trace_sstep) or isinstance(module, SYNAPSE_FC_trace_sstep): # e.g., Conv2d, Linear, etc.
                # if len(list(module.parameters())) > 0: # e.g., Conv2d, Linear, etc.
                    module = GradwithTrace(module)
                    # print('2', module)
                else: # e.g., Dropout, AvgPool, etc.
                    module = SpikeTraceOp(module)
                    # print('3', module)
                input = module(input)
            
        return input

class SpikeTraceOp(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x: Tensor):
        # x: [spike, trace], defined in OTTTLIFNode in neuron.py
        spike, trace = x[0], x[1]
        
        # print(self.module)
        spike = self.module(spike)
        with torch.no_grad():
            trace = self.module(trace)
        x = [spike, trace]

        return x
class GradwithTrace(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x: Tensor):
        # x: [spike, trace], defined in OTTTLIFNode in neuron.py
        spike, trace = x[0], x[1]
        
        with torch.no_grad():
            out = self.module(spike).detach()

        in_for_grad = ReplaceforGrad.apply(spike, trace)
        out_for_grad = self.module(in_for_grad)

        x = ReplaceforGrad.apply(out_for_grad, out)

        return x

class ReplaceforGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_r):
        return x_r

    @staticmethod
    def backward(ctx, grad):
        return (grad, grad)
######### OTTT & spikingjelly functions #################################################################################
######### OTTT & spikingjelly functions #################################################################################
######### OTTT & spikingjelly functions #################################################################################




