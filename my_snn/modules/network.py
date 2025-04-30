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

from apex.parallel import DistributedDataParallel as DDP

from modules.data_loader import *
from modules.network import *
from modules.neuron import *
from modules.synapse import *
from modules.old_fashioned import *
from modules.ae_network import *



########## 250423 REBORN ##################################################################
########## 250423 REBORN ##################################################################
########## 250423 REBORN ##################################################################
########## 250423 REBORN ##################################################################
class REBORN_MY_Sequential(nn.Sequential):
    def __init__(self, *args, BPTT_on, DFA_on, trace_on):
        super().__init__(*args)
        self.BPTT_on = BPTT_on
        self.DFA_on = DFA_on
        self.trace_on = trace_on

        if DFA_on == True:
            assert BPTT_on == False, 'DFA and BPTT cannot be used together'

        if (self.DFA_on == True):
            self.DFA_top = Top_Gradient()

        self.activation_shape_print = False
    def forward(self, input):
        if self.trace_on == True:
            dummies = []
            cnt = 0
            for module in self:
                if not isinstance(input, list): # input: only spike
                    if self.activation_shape_print:
                        print(cnt, input.size(),module)
                    cnt += 1
                    output = module(input)
                else: # input: [spike, trace]
                    if self.activation_shape_print:
                        print(cnt, input[0].shape,input[1].shape, module)
                    cnt += 1
                    assert input[0].shape == input[1].shape
                    if isinstance(module, SYNAPSE_CONV) or isinstance(module, SYNAPSE_FC): # e.g., Conv2d, Linear, etc.
                    # if len(list(module.parameters())) > 0: # e.g., Conv2d, Linear, etc.
                        module = GradwithTrace(module)
                    elif isinstance(module, Feedback_Receiver):
                        assert self.DFA_on == True, 'Feedback_Receiver should be used only when DFA_on is True'
                    else: # e.g., Dropout, AvgPool, etc.
                        module = SpikeTraceOp(module)

                    if isinstance(module, Feedback_Receiver):
                        output, dummy = module(input)
                        dummies.append(dummy)
                    else:
                        output = module(input)
                input = output

            if isinstance(input, list) == True:
                spike, trace = input[0], input[1]
                if self.DFA_on == True:
                    output = self.DFA_top(spike, *dummies)
                    # output = [output, trace] # 이거 해야될수도
            else:
                if self.DFA_on == True:
                    output = self.DFA_top(input, *dummies)

        else:
            cnt = 0
            for module in self:
                if self.activation_shape_print:
                    print(cnt, input.size(),module)
                cnt += 1
                output = module(input)
                input = output

        return output

class REBORN_MY_SNN_CONV(nn.Module):
    def __init__(self, cfg, in_c, IMAGE_SIZE,
                    synapse_conv_kernel_size, synapse_conv_stride, 
                    synapse_conv_padding, synapse_trace_const1, 
                    synapse_trace_const2, 
                    lif_layer_v_init, lif_layer_v_decay, 
                    lif_layer_v_threshold, lif_layer_v_reset,
                    lif_layer_sg_width,
                    synapse_fc_out_features,
                    tdBN_on,
                    BN_on, TIME,
                    surrogate,
                    BPTT_on,
                    DFA_on,
                    bias,
                    single_step,
<<<<<<< HEAD
                    last_lif):
=======
                    last_lif,
                    trace_on):
>>>>>>> d579b61ea41a4c477c2770887c38dcd3e52ccdfb
        super(REBORN_MY_SNN_CONV, self).__init__()
        self.layers = self.make_layers(cfg, in_c, IMAGE_SIZE,
                                    synapse_conv_kernel_size, synapse_conv_stride, 
                                    synapse_conv_padding, synapse_trace_const1, 
                                    synapse_trace_const2, 
                                    lif_layer_v_init, lif_layer_v_decay, 
                                    lif_layer_v_threshold, lif_layer_v_reset,
                                    lif_layer_sg_width,
                                    tdBN_on,
                                    BN_on, TIME,
                                    surrogate,
                                    BPTT_on,
                                    synapse_fc_out_features,
                                    DFA_on,
                                    bias,
                                    single_step,
<<<<<<< HEAD
                                    last_lif)
=======
                                    last_lif,
                                    trace_on)
>>>>>>> d579b61ea41a4c477c2770887c38dcd3e52ccdfb
        
        self.single_step = single_step

    def forward(self, spike_input):
        if self.single_step == False:
            # inputs: [Batch, Time, Channel, Height, Width]   
            spike_input = spike_input.permute(1, 0, 2, 3, 4)
            # inputs: [Time, Batch, Channel, Height, Width]  
        # else: # inputs: [Batch, Channel, Height, Width]  
 
        spike_input = self.layers(spike_input)

        if self.single_step == False:
            # spike_input = spike_input.sum(axis=0)
            spike_input = spike_input.mean(axis=0)
        return spike_input
    

    @staticmethod
    def make_layers(cfg, in_c, IMAGE_SIZE,
                        synapse_conv_kernel_size, synapse_conv_stride, 
                        synapse_conv_padding, synapse_trace_const1, 
                        synapse_trace_const2, 
                        lif_layer_v_init, lif_layer_v_decay, 
                        lif_layer_v_threshold, lif_layer_v_reset,
                        lif_layer_sg_width,
                        tdBN_on,
                        BN_on, TIME,
                        surrogate,
                        BPTT_on,
                        synapse_fc_out_features,
                        DFA_on,
                        bias,
                        single_step,
<<<<<<< HEAD
                        last_lif):
=======
                        last_lif,
                        trace_on):
>>>>>>> d579b61ea41a4c477c2770887c38dcd3e52ccdfb
        
        layers = []
        in_channels = in_c
        img_size_var = IMAGE_SIZE
        classifier_making = False
        for which in cfg:
            if (classifier_making == False):
                if which == 'P':
                    if single_step == False:
                        layers += [DimChanger_for_pooling(nn.AvgPool2d(kernel_size=2, stride=2))]
                    else:
                        layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
                    img_size_var = img_size_var // 2
                elif which == 'M':
                    if single_step == False:
                        layers += [DimChanger_for_pooling(nn.MaxPool2d(kernel_size=2, stride=2))]
                    else:
                        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                    img_size_var = img_size_var // 2
                elif which == 'D':
                    if single_step == False:
                        layers += [DimChanger_for_pooling(nn.AdaptiveAvgPool2d((1, 1)))]
                    else:
                        layers += [nn.AdaptiveAvgPool2d((1, 1))]
                    img_size_var = 1
                elif which == 'L':
                    classifier_making = True
                    layers += [DimChanger_for_FC()]
                    in_channels = in_channels*img_size_var*img_size_var
                else:
                    if (which >= 10000 and which < 20000):
                        out_channels = which - 10000
                        layers += [SYNAPSE_SEPARABLE_CONV(
                                                in_channels=in_channels,
                                                out_channels=out_channels, 
                                                kernel_size=synapse_conv_kernel_size, 
                                                stride=synapse_conv_stride, 
                                                padding=synapse_conv_padding, 
                                                TIME=TIME,
                                                bias=bias,
                                                sstep=single_step)]
                    elif (which >= 20000 and which < 30000):
                        out_channels = which - 20000
                        layers += [SYNAPSE_DEPTHWISE_CONV(
                                                in_channels=in_channels,
                                                out_channels=out_channels, 
                                                kernel_size=synapse_conv_kernel_size, 
                                                stride=synapse_conv_stride, 
                                                padding=synapse_conv_padding, 
                                                TIME=TIME,
                                                bias=bias,
                                                sstep=single_step)]
                    else:
                        out_channels = which
                        layers += [SYNAPSE_CONV(in_channels=in_channels,
                                                out_channels=out_channels, 
                                                kernel_size=synapse_conv_kernel_size, 
                                                stride=synapse_conv_stride, 
                                                padding=synapse_conv_padding, 
                                                TIME=TIME,
                                                bias=bias,
                                                sstep=single_step)]
                    
                    img_size_var = (img_size_var - synapse_conv_kernel_size + 2*synapse_conv_padding)//synapse_conv_stride + 1
                
                    in_channels = out_channels
                    

                    # batchnorm or tdBN 추가 ##########################
                    if (tdBN_on == True):
                        assert single_step == False and DFA_on == False
                        layers += [tdBatchNorm(in_channels)] # 여기서 in_channel이 out_channel임

                    if (BN_on == True):
                        assert single_step == False and DFA_on == False
                        layers += [BatchNorm(in_channels, TIME)]
                    #################################################


                    # LIF 뉴런 추가 ##################################
                    layers += [LIF_layer(v_init=lif_layer_v_init, 
                                            v_decay=lif_layer_v_decay, 
                                            v_threshold=lif_layer_v_threshold, 
                                            v_reset=lif_layer_v_reset, 
                                            sg_width=lif_layer_sg_width,
                                            surrogate=surrogate,
                                            BPTT_on=BPTT_on,
                                            trace_const1=synapse_trace_const1,
                                            trace_const2=synapse_trace_const2,
                                            TIME=TIME,
                                            sstep=single_step,
                                            trace_on=trace_on)]
                    if DFA_on == True:
                        assert single_step == True , '일단 singlestep이랑 같이가자 dfa는'
                        layers += [Feedback_Receiver(synapse_fc_out_features)]
                    #################################################
                    

            else: # classifier_making
                layers += [SYNAPSE_FC(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
                                                out_features=which, 
                                                TIME=TIME,
                                                bias=bias,
                                                sstep=single_step)]
                in_channels = which

                # batchnorm or tdBN 추가 ##########################
                if (tdBN_on == True):
                    assert single_step == False and DFA_on == False
                    layers += [tdBatchNorm(in_channels)] # 여기서 in_channel이 out_channel임

                if (BN_on == True):
                    assert single_step == False and DFA_on == False
                    layers += [BatchNorm(in_channels, TIME)]
                #################################################
                
                # LIF 뉴런 추가 ##################################
                layers += [LIF_layer(v_init=lif_layer_v_init, 
                                        v_decay=lif_layer_v_decay, 
                                        v_threshold=lif_layer_v_threshold, 
                                        v_reset=lif_layer_v_reset, 
                                        sg_width=lif_layer_sg_width,
                                        surrogate=surrogate,
                                        BPTT_on=BPTT_on,
                                        trace_const1=synapse_trace_const1,
                                        trace_const2=synapse_trace_const2,
                                        TIME=TIME,
                                        sstep=single_step,
                                        trace_on=trace_on)]
                
                if DFA_on == True:
                    assert single_step == True , '일단 singlestep이랑 같이가자 dfa는'
                    layers += [Feedback_Receiver(synapse_fc_out_features)]
                #################################################
                    

        if classifier_making == False: # cfg에 'L'한번도 없을때
            layers += [DimChanger_for_FC()]
            in_channels = in_channels*img_size_var*img_size_var
            
        layers += [SYNAPSE_FC(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
                                        out_features=synapse_fc_out_features, 
                                        TIME=TIME,
                                        bias=bias,
                                        sstep=single_step)]

        if last_lif:
            # batchnorm or tdBN 추가 ##########################
            if (tdBN_on == True):
                assert single_step == False and DFA_on == False
                layers += [tdBatchNorm(in_channels)] # 여기서 in_channel이 out_channel임

            if (BN_on == True):
                assert single_step == False and DFA_on == False
                layers += [BatchNorm(in_channels, TIME)]
            #################################################
 
            # LIF 뉴런 추가 ##################################
            layers += [LIF_layer(v_init=lif_layer_v_init, 
                                    v_decay=lif_layer_v_decay, 
                                    v_threshold=lif_layer_v_threshold, 
                                    v_reset=lif_layer_v_reset, 
                                    sg_width=lif_layer_sg_width,
                                    surrogate=surrogate,
                                    BPTT_on=BPTT_on,
                                    trace_const1=synapse_trace_const1,
                                    trace_const2=synapse_trace_const2,
                                    TIME=TIME,
                                    sstep=single_step,
                                    trace_on=False)]
            # if DFA_on == True:
            #     assert single_step == True , '일단 singlestep이랑 같이가자 dfa는'
            #     layers += [Feedback_Receiver(synapse_fc_out_features)]
            #################################################

<<<<<<< HEAD
        return REBORN_MY_Sequential(*layers, BPTT_on=BPTT_on, DFA_on=DFA_on)
=======
        return REBORN_MY_Sequential(*layers, BPTT_on=BPTT_on, DFA_on=DFA_on, trace_on=trace_on)
>>>>>>> d579b61ea41a4c477c2770887c38dcd3e52ccdfb


class REBORN_MY_SNN_FC(nn.Module):
    def __init__(self, cfg, in_c, IMAGE_SIZE, out_c,
                    synapse_trace_const1, synapse_trace_const2, 
                    lif_layer_v_init, lif_layer_v_decay, 
                    lif_layer_v_threshold, lif_layer_v_reset,
                    lif_layer_sg_width,
                    tdBN_on,
                    BN_on, TIME,
                    surrogate,
                    BPTT_on,
                    DFA_on,
                    bias,
                    single_step,
<<<<<<< HEAD
                    last_lif):
=======
                    last_lif,
                    trace_on):
>>>>>>> d579b61ea41a4c477c2770887c38dcd3e52ccdfb
        super(REBORN_MY_SNN_FC, self).__init__()
        self.layers = self.make_layers(cfg, in_c, IMAGE_SIZE, out_c,
                    synapse_trace_const1, synapse_trace_const2, 
                    lif_layer_v_init, lif_layer_v_decay, 
                    lif_layer_v_threshold, lif_layer_v_reset,
                    lif_layer_sg_width,
                    tdBN_on,
                    BN_on, TIME,
                    surrogate,
                    BPTT_on,
                    DFA_on,
                    bias,
                    single_step,
<<<<<<< HEAD
                    last_lif)
=======
                    last_lif,
                    trace_on)
>>>>>>> d579b61ea41a4c477c2770887c38dcd3e52ccdfb
        self.single_step = single_step
    def forward(self, spike_input):
        if self.single_step == False:
            # inputs: [Batch, Time, Channel, Height, Width]   
            spike_input = spike_input.permute(1, 0, 2, 3, 4)
            # inputs: [Time, Batch, Channel, Height, Width]   
        # else: # inputs: [Batch, Channel, Height, Width]  
        
        spike_input = self.layers(spike_input)

        if self.single_step == False:
            # spike_input = spike_input.sum(axis=0)
            spike_input = spike_input.mean(axis=0)

        return spike_input
    

    @staticmethod
    def make_layers(cfg, in_c, IMAGE_SIZE, out_c,
                            synapse_trace_const1, synapse_trace_const2, 
                            lif_layer_v_init, lif_layer_v_decay, 
                            lif_layer_v_threshold, lif_layer_v_reset,
                            lif_layer_sg_width,
                            tdBN_on,
                            BN_on, TIME,
                            surrogate,
                            BPTT_on,
                            DFA_on,
                            bias,
                            single_step,
<<<<<<< HEAD
                            last_lif):
=======
                            last_lif,
                            trace_on):
>>>>>>> d579b61ea41a4c477c2770887c38dcd3e52ccdfb

        layers = []
        img_size = IMAGE_SIZE
        in_channels = in_c * img_size * img_size
        class_num = out_c
        pre_pooling_done = False
        for which in cfg:
            if which == 'P':
                assert pre_pooling_done == False, 'you must not do pooling after FC'
                if single_step == False:
                    layers += [DimChanger_for_pooling(nn.AvgPool2d(kernel_size=2, stride=2))]
                else:
                    layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
                img_size = img_size // 2
                in_channels = in_c * img_size * img_size
            elif which == 'M':
                assert pre_pooling_done == False, 'you must not do pooling after FC'
                if single_step == False:
                    layers += [DimChanger_for_pooling(nn.MaxPool2d(kernel_size=2, stride=2))]
                else:
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                img_size = img_size // 2
                in_channels = in_c * img_size * img_size
            else:
                if (pre_pooling_done == False):
                    layers += [DimChanger_for_FC()]
                    pre_pooling_done = True
                out_channels = which
                layers += [SYNAPSE_FC(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
                                            out_features=out_channels, 
                                            TIME=TIME,
                                            bias=bias,
                                            sstep=single_step)]
                in_channels = out_channels
            
            
                if (tdBN_on == True):
                    assert single_step == False and DFA_on == False
                    layers += [tdBatchNorm_FC(in_channels)] # 여기서 in_channel이 out_channel임
                if (BN_on == True):
                    assert single_step == False and DFA_on == False
                    layers += [BatchNorm_FC(in_channels, TIME)]
                    
                # LIF 뉴런 추가 ##################################
                layers += [LIF_layer(v_init=lif_layer_v_init, 
                                        v_decay=lif_layer_v_decay, 
                                        v_threshold=lif_layer_v_threshold, 
                                        v_reset=lif_layer_v_reset, 
                                        sg_width=lif_layer_sg_width,
                                        surrogate=surrogate,
                                        BPTT_on=BPTT_on,
                                        trace_const1=synapse_trace_const1,
                                        trace_const2=synapse_trace_const2,
                                        TIME=TIME,
                                        sstep=single_step,
                                        trace_on=trace_on)]
                if DFA_on == True:
                    assert single_step == True , '일단 singlestep이랑 같이가자 dfa는'
                    layers += [Feedback_Receiver(class_num)]
                #################################################

        
        out_channels = class_num
        layers += [SYNAPSE_FC(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
                                    out_features=out_channels, 
                                    TIME=TIME,
                                    bias=bias,
                                    sstep=single_step)]

        if last_lif:
            if (tdBN_on == True):
                layers += [tdBatchNorm_FC(in_channels)] # 여기서 in_channel이 out_channel임
<<<<<<< HEAD

            if (BN_on == True):
                layers += [BatchNorm_FC(in_channels, TIME)]

=======

            if (BN_on == True):
                layers += [BatchNorm_FC(in_channels, TIME)]

>>>>>>> d579b61ea41a4c477c2770887c38dcd3e52ccdfb

            # LIF 뉴런 추가 ##################################
            layers += [LIF_layer(v_init=lif_layer_v_init, 
                                    v_decay=lif_layer_v_decay, 
                                    v_threshold=lif_layer_v_threshold, 
                                    v_reset=lif_layer_v_reset, 
                                    sg_width=lif_layer_sg_width,
                                    surrogate=surrogate,
                                    BPTT_on=BPTT_on,
                                    trace_const1=synapse_trace_const1,
                                    trace_const2=synapse_trace_const2,
                                    TIME=TIME,
                                    sstep=single_step,
                                    trace_on=False)]
            # if DFA_on == True:
            #     assert single_step == True , '일단 singlestep이랑 같이가자 dfa는'
            #     layers += [Feedback_Receiver(class_num)]
            #################################################
        
        return REBORN_MY_Sequential(*layers, BPTT_on=BPTT_on, DFA_on=DFA_on, trace_on=trace_on)

########## 250423 REBORN END ##################################################################
########## 250423 REBORN END ##################################################################
########## 250423 REBORN END ##################################################################
########## 250423 REBORN END ##################################################################
































































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
        # x shape: [..., C, H, W] 형태를 [ ..., C*H*W ]로 변환
        *leading_dims, C, H, W = x.shape
        x = x.view(*leading_dims, C * H * W)
        return x
    
    
class DimChanger_for_change_0_1(nn.Module):
    def __init__(self):
        super(DimChanger_for_change_0_1, self).__init__()

    def forward(self, x):
        assert len(x.shape) == 5, 'x shape should be [Time, Batch, Channel, Height, Width]'
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



# ######### OTTT & spikingjelly functions #################################################################################
# ######### OTTT & spikingjelly functions #################################################################################
# ######### OTTT & spikingjelly functions #################################################################################
# class Scale(nn.Module):
#     def __init__(self, scale):
#         super(Scale, self).__init__()
#         self.scale = scale

#     def forward(self, x):
#         return x * self.scale
    
# class MY_Sequential(nn.Sequential):
#     def __init__(self, *args, BPTT_on, DFA_on, class_num):
#         super().__init__(*args)
#         self.BPTT_on = BPTT_on
#         self.DFA_on = DFA_on
#         self.class_num = class_num
#         if (self.DFA_on == True):
#             self.DFA_top = Top_Gradient()

#         self.just_shell = True
#     def forward(self, input):
#         if self.DFA_on == True:
#             dummies = []

#         for module in self:
#             if self.BPTT_on == True:
#                 if isinstance(module, Feedback_Receiver):
#                     output, dummy = module(input)
#                     dummies.append(dummy)
#                 else:
#                     output = module(input)
#                 if isinstance(module, ResidualBlock_fc_sstep) or isinstance(module, ResidualBlock_conv_sstep) or isinstance(module, ResidualBlock_fc) or isinstance(module, ResidualBlock_conv): # e.g., ResidualBlock_fc_sstep
#                     output = output + input

#             elif self.BPTT_on == False: #ottt with trace
                
#                 if not isinstance(input, list): # input: only spike
#                     output = module(input)
#                     if isinstance(module, ResidualBlock_fc_sstep) or isinstance(module, ResidualBlock_conv_sstep) or isinstance(module, ResidualBlock_fc) or isinstance(module, ResidualBlock_conv): # e.g., ResidualBlock_fc_sstep
#                         assert isinstance(input, list) == isinstance(output, list), 'residual input and output should have same type'
#                         output = output + input
#                 else: # input: [spike, trace]
#                     residual = False
#                     if isinstance(module, SYNAPSE_CONV_trace) or isinstance(module, SYNAPSE_FC_trace) or isinstance(module, SYNAPSE_CONV_trace_sstep) or isinstance(module, SYNAPSE_FC_trace_sstep): # e.g., Conv2d, Linear, etc.
#                     # if len(list(module.parameters())) > 0: # e.g., Conv2d, Linear, etc.
#                         module = GradwithTrace(module)
#                     elif isinstance(module, ResidualBlock_fc_sstep) or isinstance(module, ResidualBlock_conv_sstep) or isinstance(module, ResidualBlock_fc) or isinstance(module, ResidualBlock_conv): # e.g., ResidualBlock_fc_sstep
#                         residual = True
#                     elif isinstance(module, Feedback_Receiver):
#                         assert self.DFA_on == True, 'Feedback_Receiver should be used only when DFA_on is True'
#                     else: # e.g., Dropout, AvgPool, etc.
#                         module = SpikeTraceOp(module)

#                     if isinstance(module, Feedback_Receiver):
#                         output, dummy = module(input)
#                         dummies.append(dummy)
#                     else:
#                         output = module(input)

#                     if residual == True: # e.g., ResidualBlock_fc_sstep
#                         assert isinstance(input, list) == isinstance(output, list) and len(output) == len(input) and len(input) == 2, 'residual input and output should have same type'
#                         output = [a + b for a, b in zip(output, input)] #output = output + input
#             input = output

#         if self.DFA_on == True:
#             if isinstance(input, list) == True:
#                 spike, trace = input[0], input[1]
#                 output = self.DFA_top(spike, *dummies)
#                 output = [output, trace]
#             else:
#                 output = self.DFA_top(input, *dummies)
#         return output

class SpikeTraceOp(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x: Tensor):
        # x: [spike, trace], defined in OTTTLIFNode in neuron.py
        spike, trace = x[0], x[1]
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




######### ASAP DFA CODE ################################################################################################
######### ASAP DFA CODE ################################################################################################
######### ASAP DFA CODE ################################################################################################
class feedback_receiver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight_fb):
        output = input.clone()
        dummy = torch.Tensor(input.size()[0],weight_fb.size()[0]).zero_().to(input.device)
        ctx.save_for_backward(weight_fb,)
        ctx.shape = input.shape
        return output, dummy
    
    @staticmethod
    def backward(ctx, grad_output, grad_dummy):
        weight_fb, = ctx.saved_tensors
        input_size = ctx.shape
        z = torch.mm(grad_dummy.view(grad_dummy.size()[0],-1), weight_fb)
        grad_input = z.view(input_size)
        grad_weight_fb = None
        return grad_input, grad_weight_fb


class Feedback_Receiver(nn.Module):
    def __init__(self, connect_features):
        super(Feedback_Receiver, self).__init__()
        self.connect_features = connect_features
        self.weight_fb = None
    
    def forward(self, input):
        if isinstance(input, list) == True:
            spike, trace = input[0], input[1]
        else:
            spike, trace = input, input

        if self.weight_fb is None:
            self.weight_fb = nn.Parameter(torch.Tensor(self.connect_features, *spike.size()[1:]).view(self.connect_features, -1)).to(spike.device)
            # nn.init.normal_(self.weight_fb, std = math.sqrt(1./self.connect_features))
            # nn.init.kaiming_normal_(self.weight_fb, mode='fan_out', nonlinearity='relu')
            # nn.init.xavier_uniform_(self.weight_fb)
            nn.init.xavier_normal_(self.weight_fb)
        if isinstance(input, list) == True:
            output, dummy = feedback_receiver.apply(spike, self.weight_fb)
            output = [output, trace]
        else:
            output, dummy = feedback_receiver.apply(spike, self.weight_fb)
        return output, dummy

class top_gradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, *dummies):
        output = input.clone()
        ctx.save_for_backward(output ,*dummies)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        output, *dummies = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_dummies = [grad_output.clone() for dummy in dummies]
        return tuple([grad_input, *grad_dummies])

class Top_Gradient(nn.Module):
    def __init__(self):
        super(Top_Gradient, self).__init__()
    
    def forward(self, input, *dummies):
        return top_gradient.apply(input, *dummies)
######### ASAP DFA CODE ################################################################################################
######### ASAP DFA CODE ################################################################################################
######### ASAP DFA CODE ################################################################################################
    




# ######### BP_DFA_SWAP ################################################################################################
# ######### BP_DFA_SWAP ################################################################################################
# ######### BP_DFA_SWAP ###############################################################################################
# def BP_DFA_SWAP(net, convTrue_fcFalse, single_step, ddp_on, args_gpu):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     if isinstance(net, nn.DataParallel) == True:
#         net_old = net.module
#     else:
#         net_old = net

#     if net_old.params['DFA_on'] == True:
#         print('\n==================== e-transport DFA --> BP ===============================================\n==================== e-transport DFA --> BP ===============================================\n\n')
#     else:
#         print('\n==================== e-transport BP --> DFA ===============================================\n==================== e-transport BP --> DFA ===============================================\n\n')

#     net_old.params['DFA_on'] = not net_old.params['DFA_on'] # conversion

#     if (convTrue_fcFalse == False):
#         if (single_step == False):
#             net_new = MY_SNN_FC(**(net_old.params)).to(device)
#         else:
#             net_new = MY_SNN_FC_sstep(**(net_old.params)).to(device)
#     else:
#         if (single_step == False):
#             net_new = MY_SNN_CONV(**(net_old.params)).to(device)
#         else:
#             net_new = MY_SNN_CONV_sstep(**(net_old.params)).to(device)

#     net_old_layer_list = list(net_old.named_modules())
#     old_index = 0
#     net_new_layer_list = list(net_new.named_modules())
#     new_index = 0
#     while (True):
#         while (True):
#             old_name, old_current_layer = net_old_layer_list[old_index]
#             old_index = old_index + 1
#             if any(p.requires_grad for p in old_current_layer.parameters()) and not hasattr(old_current_layer, 'just_shell'):
#                 break
#         while (True):
#             new_name, new_current_layer = net_new_layer_list[new_index]
#             new_index = new_index + 1
#             if any(p.requires_grad for p in new_current_layer.parameters()) and not hasattr(new_current_layer, 'just_shell'):
#                 break

#         assert type(old_current_layer) == type(new_current_layer), 'layer type should be same'

#         new_current_layer.weight = old_current_layer.weight
#         if hasattr(new_current_layer, 'bias'):
#             new_current_layer.bias = old_current_layer.bias

#         if old_index >= len(net_old_layer_list) or new_index >= len(net_new_layer_list):
#             break
            
#     if isinstance(net, nn.DataParallel) == True:
#         assert ddp_on == False
#         net_new = torch.nn.DataParallel(net_new) 
    
#     if isinstance(net, nn.DataParallel) == False:
#         assert ddp_on == True
#         device = args_gpu
#         net_new = net_new.to(args_gpu)
#         net_new = DDP(net_new, delay_allreduce=True)

#     net_new = net_new.to(device)
#     # if isinstance(net, nn.DataParallel) == True or torch.distributed.get_rank() == 0:
#     #     print(net_new)  

#     return net_new
# ######### BP_DFA_SWAP ################################################################################################
# ######### BP_DFA_SWAP ################################################################################################
# ######### BP_DFA_SWAP ################################################################################################
    

######### Dropout ################################################################################################
######### Dropout ################################################################################################
######### Dropout ################################################################################################
class Dropout_sstep(nn.Module):
    def __init__(self, p, TIME):
        super().__init__()
        assert 0.0 < p < 1.0
        self.p = p
        self.TIME = TIME
        self.time_count = 0

    def forward(self, x, trace_input = False):
        if self.training == True:
            if trace_input == False:
                self.time_count = self.time_count + 1
                if self.time_count == 1:
                    self.mask = F.dropout(torch.ones_like(x.data), self.p, training=True)
                    # self.mask = F.dropout2d(torch.ones_like(x.data), self.p, training=True)
            x = x * self.mask
            if trace_input == False:
                if (self.time_count == self.TIME):
                    self.time_count = 0
        return x
######### Dropout ################################################################################################
######### Dropout ################################################################################################
######### Dropout ################################################################################################
    


######### UDA GRL ################################################################################################
######### UDA GRL ################################################################################################
######### UDA GRL ################################################################################################
class Gradient_Reversal_Layer(nn.Module):
    def __init__(self, alpha = 1.0):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return GRL_METHOD.apply(x, self.alpha)
    
class GRL_METHOD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        x, alpha = ctx.saved_tensors

        if ctx.needs_input_grad[0]:
            grad_input = -alpha * grad_output
            
        return grad_input, None
######### UDA GRL ################################################################################################
######### UDA GRL ################################################################################################
######### UDA GRL ################################################################################################
    
