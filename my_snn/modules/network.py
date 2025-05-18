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
# class REBORN_MY_Sequential(nn.Sequential):
#     def __init__(self, *args, BPTT_on, DFA_on, trace_on):
#         super().__init__(*args)
#         self.BPTT_on = BPTT_on
#         self.DFA_on = DFA_on
#         self.trace_on = trace_on

#         if DFA_on == True:
#             assert BPTT_on == False, 'DFA and BPTT cannot be used together'

#         if (self.DFA_on == True):
#             self.DFA_top = Top_Gradient()

#         self.activation_shape_print = False
#     def forward(self, input):
#         if self.trace_on == True:
#             dummies = []
#             cnt = 0
#             for module in self:
#                 if not isinstance(input, list): # input: only spike
#                     if self.activation_shape_print:
#                         print(cnt, input.size(),module)
#                     cnt += 1
#                     output = module(input)
#                 else: # input: [spike, trace]
#                     if self.activation_shape_print:
#                         print(cnt, input[0].shape,input[1].shape, module)
#                     cnt += 1
#                     assert input[0].shape == input[1].shape
#                     if isinstance(module, SYNAPSE_CONV) or isinstance(module, SYNAPSE_FC): # e.g., Conv2d, Linear, etc.
#                     # if len(list(module.parameters())) > 0: # e.g., Conv2d, Linear, etc.
#                         module = GradwithTrace(module)
#                     elif isinstance(module, Feedback_Receiver):
#                         assert self.DFA_on == True, 'Feedback_Receiver should be used only when DFA_on is True'
#                     else: # e.g., Dropout, AvgPool, etc.
#                         module = SpikeTraceOp(module)

#                     if isinstance(module, Feedback_Receiver):
#                         output, dummy = module(input)
#                         dummies.append(dummy)
#                     else:
#                         output = module(input)
#                 input = output

#             if isinstance(input, list) == True:
#                 spike, trace = input[0], input[1]
#                 if self.DFA_on == True:
#                     output = self.DFA_top(spike, *dummies)
#                     # output = [output, trace] # 이거 해야될수도
#             else:
#                 if self.DFA_on == True:
#                     output = self.DFA_top(input, *dummies)

#         else:
#             cnt = 0
#             for module in self:
#                 if self.activation_shape_print:
#                     print(cnt, input.size(),module)
#                 cnt += 1
#                 output = module(input)
#                 input = output

#         return output
    
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

        self.activation_shape_print = False # True False
    def forward(self, input):
        dummies = []
        cnt = 0
        for module in self:
            cnt+=1

            if isinstance(input, list):
                assert self.trace_on == True
                assert len(input) == 2 
                assert input[0].shape == input[1].shape
                if self.activation_shape_print: print(cnt, input[0].shape, input[1].shape, module)

                if isinstance(module, SYNAPSE_CONV) or isinstance(module, SYNAPSE_FC): # e.g., Conv2d, Linear, etc.
                # if len(list(module.parameters())) > 0: # e.g., Conv2d, Linear, etc.
                    module = GradwithTrace(module)
                elif isinstance(module, Feedback_Receiver):
                    assert self.DFA_on == True, 'Feedback_Receiver should be used only when DFA_on is True'
                else: # e.g., Dropout, AvgPool, etc.
                    module = SpikeTraceOp(module)
            else:
                if self.activation_shape_print: print(cnt, input.shape, module)


            if isinstance(module, Feedback_Receiver):
                assert self.DFA_on == True, 'Feedback_Receiver should be used only when DFA_on is True'
                output, dummy = module(input)
                dummies.append(dummy)
            else:
                output = module(input)

            input = output
        

        if isinstance(input, list) == True:
            output, trace = input[0], input[1]

        if self.DFA_on == True:
            output = self.DFA_top(output, *dummies)

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
                    last_lif,
                    trace_on):
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
                                    last_lif,
                                    trace_on)
        
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
                        last_lif,
                        trace_on):
        
        layers = []
        in_channels = in_c
        img_size_var = IMAGE_SIZE
        classifier_making = False
        Feedback_Receiver_count = 0
        layer_count = 0
        for which in cfg:
            layer_count += 1
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
                                                sstep=single_step,
                                                time_different_weight=False)]
                    
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
                    trace_on_temp = trace_on_temp
                    trace_on_temp = False if layer_count == len(cfg) else trace_on_temp
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
                                            trace_on=trace_on_temp)]
                    if DFA_on == True:
                        assert single_step == True , '일단 singlestep이랑 같이가자 dfa는'
                        layers += [Feedback_Receiver(synapse_fc_out_features,Feedback_Receiver_count)]
                        Feedback_Receiver_count += 1
                    #################################################
                    

            else: # classifier_making
                layers += [SYNAPSE_FC(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
                                                out_features=which, 
                                                TIME=TIME,
                                                bias=bias,
                                                sstep=single_step,
                                                time_different_weight=False)]
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
                trace_on_temp = trace_on_temp
                trace_on_temp = False if layer_count == len(cfg) else trace_on_temp
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
                                        trace_on=trace_on_temp)]
                
                if DFA_on == True:
                    assert single_step == True , '일단 singlestep이랑 같이가자 dfa는'
                    layers += [Feedback_Receiver(synapse_fc_out_features,Feedback_Receiver_count)]
                    Feedback_Receiver_count += 1
                #################################################
                    

        if classifier_making == False: # cfg에 'L'한번도 없을때
            layers += [DimChanger_for_FC()]
            in_channels = in_channels*img_size_var*img_size_var
            
        layers += [SYNAPSE_FC(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
                                        out_features=synapse_fc_out_features, 
                                        TIME=TIME,
                                        bias=bias,
                                        sstep=single_step,
                                        time_different_weight=False)]

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
            #     Feedback_Receiver_count += 1
            #################################################

        return REBORN_MY_Sequential(*layers, BPTT_on=BPTT_on, DFA_on=DFA_on, trace_on=trace_on)


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
                    last_lif,
                    trace_on):
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
                    last_lif,
                    trace_on)
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
                            last_lif,
                            trace_on):

        layers = []
        img_size = IMAGE_SIZE
        in_channels = in_c * img_size * img_size
        class_num = out_c
        pre_pooling_done = False
        Feedback_Receiver_count = 0
        layer_count = 0
        for which in cfg:
            layer_count += 1
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
                    layers += [Shaker_for_FC()]
                    layers += [Sparsity_Checker(TIME)]
                out_channels = which
                layers += [SYNAPSE_FC(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
                                            out_features=out_channels, 
                                            TIME=TIME,
                                            bias=bias,
                                            sstep=single_step,
                                            time_different_weight=False)]
                in_channels = out_channels
            
            
                if (tdBN_on == True):
                    assert single_step == False and DFA_on == False
                    layers += [tdBatchNorm_FC(in_channels)] # 여기서 in_channel이 out_channel임
                if (BN_on == True):
                    assert single_step == False and DFA_on == False
                    layers += [BatchNorm_FC(in_channels, TIME)]
                    
                # LIF 뉴런 추가 ##################################
                trace_on_temp = trace_on_temp
                trace_on_temp = False if layer_count == len(cfg) else trace_on_temp
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
                                        trace_on=trace_on_temp)]
                layers += [Sparsity_Checker(TIME)]
                if DFA_on == True:
                    assert single_step == True , '일단 singlestep이랑 같이가자 dfa는'
                    layers += [Feedback_Receiver(class_num, Feedback_Receiver_count)]
                    Feedback_Receiver_count += 1 
                #################################################

        
        out_channels = class_num
        layers += [SYNAPSE_FC(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
                                    out_features=out_channels, 
                                    TIME=TIME,
                                    bias=bias,
                                    sstep=single_step,
                                    time_different_weight=False)]

        if last_lif:
            if (tdBN_on == True):
                layers += [tdBatchNorm_FC(in_channels)] # 여기서 in_channel이 out_channel임

            if (BN_on == True):
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
                                    trace_on=False)]
            layers += [Sparsity_Checker(TIME)]
            
            # if DFA_on == True:
            #     assert single_step == True , '일단 singlestep이랑 같이가자 dfa는'
            #     layers += [Feedback_Receiver(class_num,Feedback_Receiver_count)]
            #     Feedback_Receiver_count += 1
            #
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
    
class Shaker_for_FC(nn.Module):
    def __init__(self):
        super(Shaker_for_FC, self).__init__()
        self.perm = None

    def forward(self, x):
        if isinstance(x, list) == True:
            spike, trace = x[0], x[1]
            # x shape: [..., C, H, W] 형태를 [ ..., C*H*W ]로 변환
            *leading_dims, feature_dim = spike.shape
            # feature_dim 순서 섞기 (같은 순서를 모든 샘플에 적용)
            if self.perm == None:
                self.perm = torch.randperm(feature_dim, device=spike.device)
                print('self.perm', 'fc input 처음에 한번 섞기',self.perm)
            spike = spike[..., self.perm]  # 마지막 차원만 perm으로 섞음
            trace = trace[..., self.perm]  # 마지막 차원만 perm으로 섞음
            return [spike, trace]
        else:
            # x shape: [..., C, H, W] 형태를 [ ..., C*H*W ]로 변환
            *leading_dims, feature_dim = x.shape
            # feature_dim 순서 섞기 (같은 순서를 모든 샘플에 적용)
            if self.perm == None:
                self.perm = torch.randperm(feature_dim, device=x.device)
                print('self.perm', 'fc input 처음에 한번 섞기',self.perm)
            x = x[..., self.perm]  # 마지막 차원만 perm으로 섞음
            return x
    
class Sparsity_Checker(nn.Module):
    def __init__(self, TIME):
        super(Sparsity_Checker, self).__init__()
        self.count = 0
        self.sparsity_ratio = 0
        self.TIME = TIME
        self.t = 0
        self.spike_collector = None

    def forward(self, x):

        if isinstance(x, list) == True:
            spike, trace = x[0], x[1]
            # self.spike_collector를 x의 크기로 0으로 초기화
            if self.t == 0:
                self.sparsity_ratio = 0
                self.spike_collector = torch.zeros_like(spike, device=spike.device)
            self.spike_collector = self.spike_collector + spike
            self.unique_vals, self.unique_counts = torch.unique(self.spike_collector, return_counts=True)

            num_zeros_of_collector = (self.spike_collector == 0).sum().item()
            num_zeros = (spike == 0).sum().item()
            total_elements = spike.numel()


            self.temp_sparsity_ratio_of_collector = num_zeros_of_collector / total_elements
            self.sparsity_ratio += num_zeros / total_elements
            self.t = self.t + 1
            if self.t == self.TIME:
                self.t = 0
                self.sparsity_ratio /= self.TIME

            return [spike, trace]
        else:
            # self.spike_collector를 x의 크기로 0으로 초기화
            if self.t == 0:
                self.sparsity_ratio = 0
                self.spike_collector = torch.zeros_like(x, device=x.device)
            self.spike_collector = self.spike_collector + x
            self.unique_vals, self.unique_counts = torch.unique(self.spike_collector, return_counts=True)

            num_zeros_of_collector = (self.spike_collector == 0).sum().item()
            num_zeros = (x == 0).sum().item()
            total_elements = x.numel()


            self.temp_sparsity_ratio_of_collector = num_zeros_of_collector / total_elements
            self.sparsity_ratio += num_zeros / total_elements
            self.t = self.t + 1
            if self.t == self.TIME:
                self.t = 0
                self.sparsity_ratio /= self.TIME

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

        temp = grad_dummy.view(grad_dummy.size()[0],-1)
        z = torch.mm(grad_dummy.view(grad_dummy.size()[0],-1), weight_fb)
        grad_input = z.view(input_size)
        grad_weight_fb = None
        return grad_input, grad_weight_fb


class Feedback_Receiver(nn.Module):
    def __init__(self, connect_features, count):
        super(Feedback_Receiver, self).__init__()
        self.connect_features = connect_features
        self.weight_fb = None
        self.count = count # 몇번째 Feedback_Receiver
    
    def forward(self, input):
        if isinstance(input, list) == True:
            spike, trace = input[0], input[1]
        else:
            spike, trace = input, input

        if self.weight_fb is None:
            self.weight_fb = nn.Parameter(torch.Tensor(self.connect_features, *spike.size()[1:]).view(self.connect_features, -1)).to(spike.device)
            # self.weight_fb 의 사이즈 = 10x200
            # nn.init.normal_(self.weight_fb, std = math.sqrt(1./self.connect_features))
            # nn.init.kaiming_normal_(self.weight_fb, mode='fan_out', nonlinearity='relu')
            # nn.init.xavier_uniform_(self.weight_fb)

            nn.init.xavier_normal_(self.weight_fb) # 표준!!!

            # fb_weight_init_type = 'baseline'
            # fb_weight_init_type = 'zero_p1'
            fb_weight_init_type = 'slice_and_copy_class_scale'
            # fb_weight_init_type = 'make_random_with_sparsity'

            if fb_weight_init_type == 'baseline':
                pass
            elif fb_weight_init_type == 'm1_p1':
                # -1, 1 하기
                self.weight_fb = nn.Parameter(
                    torch.randint(0, 2, (self.connect_features, torch.prod(torch.tensor(spike.size()[1:])))).float().mul_(2).sub_(1).to(spike.device)
                )
            elif fb_weight_init_type == 'zero_p1':
                # 0, 1 하기
                self.weight_fb = nn.Parameter(
                    torch.randint(0, 2, (self.connect_features, torch.prod(torch.tensor(spike.size()[1:]))))
                        .float()
                        .to(spike.device)
                )
            elif fb_weight_init_type == 'm1_zero_p1':
                # -1, 0, 1 하기
                self.weight_fb = nn.Parameter(
                    torch.randint(0, 3, (self.connect_features, torch.prod(torch.tensor(spike.size()[1:]))))
                        .float()
                        .add_(-1)  # 0 → -1, 1 → 0, 2 → 1
                        .to(spike.device)
                )
            elif fb_weight_init_type == 'p1':
                # 1만하기
                self.weight_fb.data.fill_(1.0)
            elif fb_weight_init_type == 'slice_and_copy':
                # 자르고 붙여서 주기함수로 만들어버리기
                slice_num_per_class = 10
                self.weight_fb = self.slice_and_copy(self.weight_fb, slice_num_per_class, self.connect_features, torch.prod(torch.tensor(spike.size()[1:])).item())
            elif fb_weight_init_type == 'slice_and_copy_class_scale':
                # 자르고 붙여서 주기함수로 만들어버리기
                slice_num_per_class = 10
                self.weight_fb = self.slice_and_copy_class_scale(self.weight_fb, slice_num_per_class, self.connect_features, torch.prod(torch.tensor(spike.size()[1:])).item(), self.count)
            elif fb_weight_init_type == 'make_random_with_sparsity':
                sparsity = 0.9 # 0이 10개중 9개
                self.weight_fb = self.make_random_with_sparsity(self.weight_fb, self.count, sparsity)
                
            elif fb_weight_init_type == 'slice_and_copy_half_half':
                # -1, 1 정확히 반반해서 반복
                slice_num_per_class = 10
                self.weight_fb = self.slice_and_copy_half_half(self.weight_fb, slice_num_per_class, self.connect_features, torch.prod(torch.tensor(spike.size()[1:])).item())
            elif fb_weight_init_type == 'sine':
                # 사인파형 만들기
                slice_num_per_class = 10
                self.weight_fb = nn.Parameter(
                    torch.sin(
                        torch.linspace(0, 2 * slice_num_per_class * math.pi, torch.prod(torch.tensor(spike.size()[1:]))).unsqueeze(0) +  # (1, 200)
                        torch.linspace(0, 2 * math.pi, self.connect_features).unsqueeze(1)  # (10, 1)
                    ).to(spike.device)
                )
            elif fb_weight_init_type == 'custom_copy':
                slice_num_per_class = 10
                destination_size = torch.prod(torch.tensor(spike.size()[1:])).item()
                slice_size = destination_size // slice_num_per_class
                shift_size = slice_size // slice_num_per_class
                assert destination_size % slice_num_per_class == 0, "destination_size must be divisible by slice_num_per_class"
                assert slice_size % slice_num_per_class == 0, "slice_size must be divisible by slice_num_per_class"

                one_slice = torch.linspace(-1, 1, steps=slice_size)

                # y = x ^ 3 그래프로 0에 몰리게
                # one_slice = one_slice ** 3


                # one_slice = torch.linspace(-3, 3, steps=slice_size)
                # one_slice = torch.tanh(one_slice)


                one_slice = one_slice.to(spike.device)
                self.weight_fb = self.copy_and_paste(one_slice, slice_num_per_class, self.connect_features, destination_size, slice_size, shift_size)
                
            else:
                assert False, 'fb_weight_init_type is not defined'
            
            
            # self.plot_sine_waves(self.weight_fb) # 각 클라스 것을 한 그림에 플랏
            self.plot_distribution_wb(self.weight_fb)

        if isinstance(input, list) == True:
            output, dummy = feedback_receiver.apply(spike, self.weight_fb)
            output = [output, trace]
        else:
            output, dummy = feedback_receiver.apply(spike, self.weight_fb)
        return output, dummy

    @staticmethod
    def slice_and_copy(weights, slice_num_per_class, class_num, destination_size):
        new_weights = weights.clone()
        assert destination_size % slice_num_per_class == 0, "destination_size must be divisible by slice_num_per_class"
        slice_size = destination_size // slice_num_per_class
        one_slice = new_weights[0, :slice_size]
        assert slice_size % slice_num_per_class == 0, "slice_size must be divisible by slice_num_per_class"
        shift_size = slice_size // slice_num_per_class
        class_slice = one_slice.repeat(slice_num_per_class)
        for i in range(class_num):
            new_weights[i] = torch.cat([class_slice[shift_size * i:], class_slice[:shift_size * i]])
        return nn.Parameter(new_weights).to(weights.device)
    
    @staticmethod
    def slice_and_copy_class_scale(weights, slice_num_per_class, class_num, destination_size, count):
        my_setting = 0
        new_weights = weights.clone()
        assert slice_num_per_class == class_num
        assert destination_size % class_num == 0, "slice_size must be divisible by slice_num_per_class"
        shift_size = destination_size // class_num # 20
        slice_size = destination_size // class_num # 20
        
        assert slice_num_per_class % 2 == 0, "slice_num_per_class must be even"

        if my_setting == 0:
            one_slice_fisrt = torch.full((slice_size,), 1.0).to(weights.device)
            one_slice_second = torch.full((slice_size,), 0.0).to(weights.device)
            # one_slice_second = torch.full((slice_size,), -1.0).to(weights.device)

            # class_slice = torch.cat([one_slice_fisrt.repeat(slice_num_per_class//2), one_slice_second.repeat(slice_num_per_class//2)])
            class_slice = torch.cat([one_slice_fisrt, one_slice_second.repeat(slice_num_per_class-1)])
        else:
            assert False

        for i in range(class_num):
            new_weights[i] = torch.cat([class_slice[shift_size * i:], class_slice[:shift_size * i]])
            print('new_weights[i]', new_weights[i])
        return nn.Parameter(new_weights).to(weights.device)
    
    @staticmethod
    def make_random_with_sparsity(weights, count, sparsity):
        new_weights = weights.clone()
        assert 0.0 <= sparsity <= 1.0, "sparsity must be between 0 and 1"

        total_elements = weights.numel()
        ones_count = int(total_elements * (1.0-sparsity))
        zeros_count = total_elements - ones_count

        # # 전체 binary 벡터 생성 후 무작위 섞기
        binary_vector = torch.cat([
            torch.ones(ones_count, device=weights.device),
            torch.zeros(zeros_count, device=weights.device)
        ])
        # binary_vector = torch.cat([
        #     torch.ones(ones_count, device=weights.device),
        #     -torch.ones(zeros_count, device=weights.device)
        # ])

        shuffled = binary_vector[torch.randperm(total_elements)]

        # 원래 shape로 reshape하여 weights에 복사
        new_weights.copy_(shuffled.view_as(weights))

        return nn.Parameter(new_weights).to(weights.device)
    
    @staticmethod
    def slice_and_copy_half_half(weights, slice_num_per_class, class_num, destination_size):
        new_weights = weights.clone()
        assert destination_size % slice_num_per_class == 0, "destination_size must be divisible by slice_num_per_class"
        slice_size = destination_size // slice_num_per_class


        # one_slice = new_weights[0, :slice_size]

        assert slice_size % 2 == 0, "slice_size must be even"
        num_neg = slice_size // 2
        num_pos = slice_size - num_neg  # 홀수일 경우 1이 하나 더 많음
        values = torch.tensor([-1.0] * num_neg + [1.0] * num_pos, device=weights.device)
        # one_slice = values[torch.randperm(slice_size)]  # 섞기
        one_slice = values # 안 섞기

        # print('one_slice', one_slice)

        assert slice_size % slice_num_per_class == 0, "slice_size must be divisible by slice_num_per_class"
        shift_size = slice_size // slice_num_per_class

        # class_slice를 쉬프트하기
        class_slice = one_slice.repeat(slice_num_per_class)
        for i in range(class_num):
            new_weights[i] = torch.cat([class_slice[shift_size * i:], class_slice[:shift_size * i]])

        # # one_slice를 쉬프트하기 위의것이랑 똑같은거였네 ㅋㅋ ㅈㅅ
        # for i in range(class_num):
        #     class_slice = torch.cat([one_slice[shift_size * i:], one_slice[:shift_size * i]]).repeat(slice_num_per_class)
        #     new_weights[i] = class_slice
        return nn.Parameter(new_weights).to(weights.device)
    
    @staticmethod
    def copy_and_paste(one_slice, slice_num_per_class, class_num, destination_size, slice_size, shift_size):
        # slice_size = destination_size // slice_num_per_class
        # shift_size = slice_size // slice_num_per_class
        class_slice = one_slice.repeat(slice_num_per_class)

        new_weights = torch.zeros(class_num, destination_size).to(one_slice.device)
        for i in range(class_num):
            new_weights[i] = torch.cat([class_slice[shift_size * i:], class_slice[:shift_size * i]])
        return nn.Parameter(new_weights).to(one_slice.device)
    
    @staticmethod
    def plot_distribution_wb(weights):
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().flatten().numpy()
        else:
            raise TypeError("Input must be a torch.Tensor or array-like.")

        plt.hist(weights, bins=50, color='skyblue', edgecolor='black')
        plt.title('Weight Value Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_sine_waves(weight_fb):
        """
        weight_fb (torch.Tensor): 10개의 phase를 가진 사인파 텐서 (shape: [10, N])
        """
        plt.figure(figsize=(10, 6))
        
        # 각 phase에 해당하는 사인파를 그립니다.
        for i in range(weight_fb.size(0)):
            plt.plot(weight_fb[i].detach().cpu().flatten().numpy(), label=f'Phase {i + 1}')
        
        plt.title('Sine Waves for Each Phase')
        plt.xlabel('Position')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()


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
    
