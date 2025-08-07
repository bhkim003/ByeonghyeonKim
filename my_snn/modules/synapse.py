import sys
import os
# from typing import Self
# from pydantic import PastDatetime
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
from modules.ae_network import *



class SYNAPSE_CONV(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, TIME=8, bias=True, sstep=False, time_different_weight=False, layer_count = 0, quantize_bit_list = [], scale_exp = []):
        super(SYNAPSE_CONV, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.TIME = TIME
        self.bias = bias
        self.sstep = sstep
        self.time_different_weight = time_different_weight
        self.layer_count = layer_count
        self.quantize_bit_list = quantize_bit_list
        self.scale_exp = scale_exp
        
        if len(self.quantize_bit_list) != 0:
            assert False, '아직 준비 안됨'
            if self.layer_count == 1:
                self.bit = self.quantize_bit_list[0]
            elif self.layer_count == 2:
                self.bit = self.quantize_bit_list[1]
            elif self.layer_count == 3:
                self.bit = self.quantize_bit_list[2]
            else:
                assert False, 'layer_count should be 1, 2, or 3'
        else:
            self.bit = 0

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias)


        if self.time_different_weight == True:
            self.current_time = 0
            assert self.sstep == True
            self.conv = nn.ModuleList([nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias) for _ in range(self.TIME)])
        else:
            self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias)

    def change_timesteps(self, TIME):
        self.TIME = TIME
        
    def forward(self, spike):
        if self.sstep == False:
            assert self.time_different_weight == False
            # spike: [Time, Batch, Channel, Height, Width]   
            Time = spike.shape[0]
            assert Time == self.TIME, f'Time {Time} dimension should be same as TIME {self.TIME}'

            T, B, *spatial_dims = spike.shape
            spike = self.conv(spike.reshape(T * B, *spatial_dims))

            TB, *spatial_dims = spike.shape
            spike = spike.view(T , B, *spatial_dims).contiguous() 
        else: # sstep mode
            if self.time_different_weight == True:
                assert self.sstep == True
                spike = self.conv[self.current_time](spike)
                self.current_time += 1
                if self.current_time == self.TIME:
                    self.current_time = 0
            else:
                spike =self.conv(spike)

        return spike
    
    def __repr__(self):        
        return (f"{self.__class__.__name__}("
                f"in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, "
                f"stride={self.stride}, "
                f"padding={self.padding}, "
                f"TIME={self.TIME}, "
                f"bias={self.bias}, "
                f"sstep={self.sstep}, "
                f"time_different_weight={self.time_different_weight}, "
                f"layer_count={self.layer_count}, "
                f"quantize_bit_list={self.quantize_bit_list}, "
                f"scale_exp={self.scale_exp})")
    
class SYNAPSE_FC(nn.Module):
    def __init__(self, in_features, out_features, TIME=8, bias=True, sstep=False, time_different_weight = False, layer_count = 0, quantize_bit_list = [], scale_exp = []):
        super(SYNAPSE_FC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.TIME = TIME
        self.bias = bias
        self.sstep = sstep
        self.time_different_weight = time_different_weight
        self.layer_count = layer_count
        self.quantize_bit_list = quantize_bit_list
        self.scale_exp = scale_exp
        self.weight_exp = None
        self.bias_exp = None

        # self.quantize_bit_list_for_output = [8,8,8]
        # self.quantize_bit_list_for_output = [16,16,16]
        # self.quantize_bit_list_for_output = [15,15,15]
        self.quantize_bit_list_for_output = []
        self.scale_exp_for_output = self.scale_exp
        self.exp_for_output = None


        self.current_time = 0

        if len(self.quantize_bit_list) != 0:
            if self.layer_count == 1:
                self.bit = self.quantize_bit_list[0]
                if self.scale_exp != []:
                    self.weight_exp = self.scale_exp[0][0]
                    self.bias_exp = self.scale_exp[0][1]
            elif self.layer_count == 2:
                self.bit = self.quantize_bit_list[1]
                if self.scale_exp != []:
                    self.weight_exp = self.scale_exp[1][0]
                    self.bias_exp = self.scale_exp[1][1]
            elif self.layer_count == 3:
                self.bit = self.quantize_bit_list[2]
                if self.scale_exp != []:
                    self.weight_exp = self.scale_exp[2][0]
                    self.bias_exp = self.scale_exp[2][1]
            else:
                assert False, 'layer_count should be 1, 2, or 3'
        else:
            self.bit = 0

        if len(self.quantize_bit_list_for_output) != 0:
            if self.layer_count == 1:
                self.bit_for_output = self.quantize_bit_list_for_output[0]
                if self.scale_exp_for_output != []:
                    self.exp_for_output = min(self.scale_exp_for_output[0][0], self.scale_exp_for_output[0][1])
            elif self.layer_count == 2:
                self.bit_for_output = self.quantize_bit_list_for_output[1]
                if self.scale_exp_for_output != []:
                    self.exp_for_output = min(self.scale_exp_for_output[1][0], self.scale_exp_for_output[1][1])
            elif self.layer_count == 3:
                self.bit_for_output = self.quantize_bit_list_for_output[2]
                if self.scale_exp_for_output != []:
                    self.exp_for_output = min(self.scale_exp_for_output[2][0], self.scale_exp_for_output[2][1])
            else:
                assert False, 'layer_count should be 1, 2, or 3'
        else:
            self.bit_for_output = 0

        print('\n\n\n layer_count', self.layer_count)
        print('weight bias bit', self.bit)
        print('weight exp, bias exp', self.weight_exp, self.bias_exp)
        print('bit_for_output', self.bit_for_output,'exp_for_output', self.exp_for_output,'\n\n\n')

        if self.time_different_weight == True:
            assert self.sstep == True
            self.fc = nn.ModuleList([nn.Linear(self.in_features, self.out_features, bias=self.bias) for _ in range(self.TIME)])
        else:
            self.fc = nn.Linear(self.in_features, self.out_features, bias=self.bias)
            # with torch.no_grad():
            #     self.fc.weight.mul_(4.0)
            #     if self.fc.bias is not None:
            #         self.fc.bias.zero_()

        if self.bit > 0:
            self.quantize(self.bit,percentile_print=True)

        # self.past_fc_weight = self.fc.weight.data.detach().clone().to(self.fc.weight.device)
        # # self.past_fc_bias = self.fc.bias.data.detach().clone().to(self.fc.bias.device)

        self.post_distribution_box = []

        self.total_elements = 0.00000000000000001
        self.nonzero_elements = 0
        self.total_elements2 = 0.00000000000000001
        self.and_elements = 0
        self.zero_group_num = 0
        self.feedforward_num = 0.00000000000000001
        self.total_elements3 = 0.00000000000000001
        self.under_sixteen = 0
        self.under_fifteen = 0
        self.under_fourteen = 0
        self.under_thirteen = 0
        self.under_twelve = 0
        self.under_eleven = 0

        self.step = 0

    def change_timesteps(self, TIME):
        self.TIME = TIME

    def sparsity_print_and_reset(self):
        print(f"layer   {self.layer_count}  Sparsity: {((self.total_elements-self.nonzero_elements)/self.total_elements)*100:.4f}%")
        print(f"layer   {self.layer_count}  Overlaped Sparsity: {((self.total_elements2-self.and_elements)/self.total_elements2)*100:.4f}%")
        print(f"layer   {self.layer_count}  zero_group_num/feedforward_num: {(100*self.zero_group_num/self.feedforward_num):.6f}%")
        print(f"layer   {self.layer_count}  under_sixteen/feedforward_num: {(100*self.under_sixteen/self.total_elements3):.6f}%")
        print(f"layer   {self.layer_count}  under_fifteen/feedforward_num: {(100*self.under_fifteen/self.total_elements3):.6f}%")
        print(f"layer   {self.layer_count}  under_fourteen/feedforward_num: {(100*self.under_fourteen/self.total_elements3):.6f}%")
        print(f"layer   {self.layer_count}  under_thirteen/feedforward_num: {(100*self.under_thirteen/self.total_elements3):.6f}%")
        print(f"layer   {self.layer_count}  under_twelve/feedforward_num: {(100*self.under_twelve/self.total_elements3):.6f}%")
        print(f"layer   {self.layer_count}  under_eleven/feedforward_num: {(100*self.under_eleven/self.total_elements3):.6f}%")
        self.total_elements = 0.00000000000000001
        self.nonzero_elements = 0
        self.total_elements2 = 0.00000000000000001
        self.and_elements = 0
        self.zero_group_num = 0
        self.feedforward_num = 0.00000000000000001
        self.total_elements3 = 0.00000000000000001
        self.under_sixteen = 0
        self.under_fifteen = 0
        self.under_fourteen = 0
        self.under_thirteen = 0
        self.under_twelve = 0
        self.under_eleven = 0

    def forward(self, spike):

        if self.bit > 0:
        # if self.bit > 0 and self.current_time == 0:
            self.quantize(self.bit,percentile_print=False)


        ########### test vector extraction #################
        ########### test vector extraction #################
        ########### test vector extraction #################
        if hasattr(self, 'tb_extract_scaler'):
        # if self.layer_count == 1:
            # spike 저장
            np.savetxt(f"/home/bhkim003/SNN_CHIP_Samsung_FDSOI_28nm/test_vector/zz_tb_vector_layer{self.layer_count}/tb_input_activation{self.step}.txt", spike.detach().cpu().numpy().flatten(), fmt='%d')
            # weight 저장
            weight_scaled = self.fc.weight.data.t() *(self.tb_extract_scaler)
            np.savetxt(f"/home/bhkim003/SNN_CHIP_Samsung_FDSOI_28nm/test_vector/zz_tb_vector_layer{self.layer_count}/tb_weight_matrix{self.step}.txt", weight_scaled.detach().cpu().numpy(), fmt='%d')
            self.step = self.step + 1
        ########### test vector extraction #################
        ########### test vector extraction #################
        ########### test vector extraction #################

        # # 디바이스 통일 (예: CUDA에서 연산)
        # device = self.fc.weight.device
        # # past_fc_weight와 past_fc_bias를 같은 디바이스로 옮김
        # delta_w = self.fc.weight.data - self.past_fc_weight.to(device)
        # # delta_b = self.fc.bias.data - self.past_fc_bias.to(device)
        # # epsilon = 1e-25  # 로그 안정화를 위한 작은 수
        # # delta_w = torch.sign(delta_w) * torch.log2(delta_w.abs() + epsilon)
        # delta_w = torch.sign(delta_w) * delta_w.abs() *1024
        # # delta_b = torch.sign(delta_b) * torch.log2(delta_b.abs() + epsilon)
        # # 유일한 값 출력
        # unique_delta_w = torch.unique(delta_w)
        # # unique_delta_b = torch.unique(delta_b)

        # # print(f'layer   {self.layer_count} ')
        # # print(f"delta_w - Unique Count: {unique_delta_w.numel()}")
        # # print(f"delta_w - Unique Values: {unique_delta_w.tolist()}")
        # # # print(f"delta_b - Unique Count: {unique_delta_b.numel()}")
        # # # print(f"delta_b - Unique Values: {unique_delta_b.tolist()}")

        # allowed_values = {-2, -1, 0, 1, 2}
        # unique_values = set(unique_delta_w.tolist())

        # # 허용되지 않은 값들 찾기
        # invalid_values = unique_values - allowed_values

        # if invalid_values:
        #     print(f'layer   {self.layer_count} ')
        #     print(f"delta_w - Unique Count: {unique_delta_w.numel()}")
        #     print(f"delta_w - Unique Values: {unique_delta_w.tolist()}")
        #     print(f"⚠️ delta_w contains invalid values: {sorted(invalid_values)}")

        
        # self.past_fc_weight = self.fc.weight.data.detach().clone().to(self.fc.weight.device)
        # # self.past_fc_bias = self.fc.bias.data.detach().clone().to(self.fc.bias.device)

        # # for hw design ###############################################
        # # for hw design ###############################################
        # # for hw design ###############################################
        # self.total_elements += spike.numel()
        # self.nonzero_elements += spike.count_nonzero().item()

        # # indices = torch.arange(spike.size(1)) % 10  # 980 길이, 값은 0~9 반복
        # # counts_spike_moduloten = torch.zeros(10, dtype=torch.int32)
        # # for i in range(10):
        # #     group_spike = spike[:, indices == i]
        # #     counts_spike_moduloten[i] = group_spike.count_nonzero()
        # # self.zero_group_num += (counts_spike_moduloten == 0).sum().item()
        # # self.feedforward_num += 1

        # # if self.past_spike 존재
        # if hasattr(self, 'past_spike'):
        #     # spike와 self.past_spike의 element-wise 곱
        #     self.and_elements += (spike * self.past_spike).count_nonzero().item()
        #     self.total_elements2 += spike.numel()
        # self.past_spike = spike.detach().clone()
        
        # self.total_elements3 += 1
        # if (spike.count_nonzero().item() < 16):
        #     self.under_sixteen += 1
        # if (spike.count_nonzero().item() < 15):
        #     self.under_fifteen += 1
        # if (spike.count_nonzero().item() < 14):
        #     self.under_fourteen += 1
        # if (spike.count_nonzero().item() < 13):
        #     self.under_thirteen += 1
        # if (spike.count_nonzero().item() < 12):
        #     self.under_twelve += 1
        # if (spike.count_nonzero().item() < 11):
        #     self.under_eleven += 1


        # # for hw design ###############################################
        # # for hw design ###############################################
        # # for hw design ###############################################

        if self.sstep == False:
            assert self.time_different_weight == False
            T, B, *spatial_dims = spike.shape
            assert T == self.TIME, 'Time dimension should be same as TIME'
            spike = spike.reshape(T * B, *spatial_dims)

            spike = self.fc(spike)

            TB, *spatial_dims = spike.shape
            spike = spike.view(T , B, *spatial_dims).contiguous() 
        else: # sstep mode
            if self.time_different_weight == True:
                assert self.sstep == True
                spike = self.fc[self.current_time](spike)
            else:
                spike =self.fc(spike)
                
            self.current_time = self.current_time + 1 if self.current_time != self.TIME-1 else 0

        # self.post_distribution_box.append(spike.detach().clone())

        if self.bit_for_output > 0:
            spike = QuantizeForOutput.apply(spike, self.bit_for_output, self.exp_for_output)
        
        return spike 
    
    def __repr__(self):        
        return (f"{self.__class__.__name__}("
                f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"TIME={self.TIME}, "
                f"bias={self.bias}, "
                f"sstep={self.sstep}, "
                f"time_different_weight={self.time_different_weight}, "
                f"layer_count={self.layer_count}, "
                f"quantize_bit_list={self.quantize_bit_list}, "
                f"scale_exp={self.scale_exp})")
    
    def quantize(self, bit,percentile_print=False):
        # percentile=0 
        percentile=0.999
        # percentile=0.99
        # percentile=0.95
        if percentile_print:
            print('======================================================================================') 
            print('======================================================================================') 
            print('======================================================================================') 
            print('bit',bit, 'percentile', percentile) 
            print('======================================================================================') 
            print('======================================================================================') 
            print('======================================================================================') 
        w = self.fc.weight.data
        if self.weight_exp == None:
            max_w = w.abs().max().item()
            if percentile > 0:
                max_w = torch.quantile(w.abs().flatten(), percentile).item()
            scale_w = self.nearest_power_of_two(max_w / (2**(bit-1) -1) )
        else:
            scale_w = 2**self.weight_exp
        q_weight = self.quantize_tensor(w, bit, scale_w, zero_point=0)
        self.fc.weight.data = q_weight

        if self.bias:
            b = self.fc.bias.data
            if self.bias_exp == None:
                max_b = b.abs().max().item()
                if percentile > 0:
                    max_b = torch.quantile(b.abs().flatten(), percentile).item()
                scale_b = self.nearest_power_of_two(max_b/ (2**(bit-1) -1))
            else:
                scale_b = 2**self.bias_exp
            q_bias = self.quantize_tensor(b, bit, scale_b, zero_point=0)
            self.fc.bias.data = q_bias


    @staticmethod
    def nearest_power_of_two(x):
        """x보다 크거나 같은 가장 가까운 2의 승수를 반환"""
        if x == 0:
            assert False, 'x should not be 0'
            return 2 ** -99  # 매우 작은 값으로 대체
        exp = math.ceil(math.log2(x))
        return 2 ** exp
    @staticmethod
    def quantize_tensor(tensor, bit, scale, zero_point):
        # qmin, qmax = -32767, 32767 # 16bit
        qmin, qmax = -2**(bit-1), 2**(bit-1) - 1
        # qmin, qmax = -2**(bit-1)+1, 2**(bit-1) - 1
        # q_x = torch.clamp(round_away_from_zero(tensor / scale + zero_point), qmin, qmax) * scale
        q_x = torch.clamp((tensor / scale + zero_point).round(), qmin, qmax) * scale
        # q_x = torch.clamp(torch.trunc(tensor / scale + zero_point), qmin, qmax) * scale

        return q_x



class QuantizeForOutput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bit, exp_for_output):
        # percentile=0 
        percentile=0.999
        # percentile=0.99
        # percentile=0.95
        if exp_for_output == None:
            max_x = x.abs().max().item()
            if percentile > 0:
                max_x = torch.quantile(x.abs().flatten(), percentile).item()
            assert max_x > 0, 'max_x should be greater than 0'
            scale_x = 2**math.ceil(math.log2(max_x / (2**(bit-1) -1)))
        else:
            scale_x = 2**exp_for_output

        q_x = torch.clamp((x / scale_x + 0).round(), -2**(bit-1) + 1, 2**(bit-1) - 1) * scale_x

        return q_x

    @staticmethod
    def backward(ctx, grad_output):
        # 그냥 identity gradient 전달 (straight-through estimator 방식)
        grad_input = grad_output.clone()
        return grad_input, None, None

############## Separable Conv Synapse #######################################
############## Separable Conv Synapse #######################################
############## Separable Conv Synapse #######################################
    
class SYNAPSE_SEPARABLE_CONV(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, TIME=8, bias=True, sstep=False):
        super(SYNAPSE_SEPARABLE_CONV, self).__init__()
        assert False, 'deprecated!!'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.TIME = TIME
        self.bias = bias
        self.sstep = sstep

        self.conv_depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
        self.conv_pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        if self.sstep == False:
            T, B, *spatial_dims = x.shape
            x = x.reshape(T * B, *spatial_dims)

            x = self.conv_depthwise(x)
            x = self.conv_pointwise(x)

            TB, *spatial_dims = x.shape
            x = x.view(T , B, *spatial_dims).contiguous() 
        else: # sstep mode
            x = self.conv_depthwise(x)
            x = self.conv_pointwise(x)
        return x
    
    
    def __repr__(self): 
        return (f"{self.__class__.__name__}("
                f"in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, "
                f"stride={self.stride}, "
                f"padding={self.padding}, "
                f"TIME={self.TIME}, "
                f"bias={self.bias}, "
                f"sstep={self.sstep})")



class SYNAPSE_DEPTHWISE_CONV(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, TIME=8, bias=True, sstep=False):
        super(SYNAPSE_DEPTHWISE_CONV, self).__init__()
        assert False, 'deprecated!!'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.TIME = TIME 
        self.bias = bias
        self.sstep = sstep

        assert in_channels == out_channels, 'in_channels should be same as out_channels for depthwise conv'
        self.conv_depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
        

    def forward(self, x):
        if self.sstep == False:
            T, B, *spatial_dims = x.shape
            x = x.reshape(T * B, *spatial_dims)

            x = self.conv_depthwise(x)

            TB, *spatial_dims = x.shape
            x = x.view(T , B, *spatial_dims).contiguous() 
        else:
            x = self.conv_depthwise(x)
        return x
    def __repr__(self): 
        return (f"{self.__class__.__name__}("
                f"in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, "
                f"stride={self.stride}, "
                f"padding={self.padding}, "
                f"TIME={self.TIME}, "
                f"bias={self.bias}, "
                f"sstep={self.sstep})")
############## Separable Conv Synapse #######################################
############## Separable Conv Synapse #######################################
############## Separable Conv Synapse #######################################
    





# ############## OTTT Conv trace sstep #######################################
# ############## OTTT Conv trace sstep #######################################
# ############## OTTT Conv trace sstep #######################################
# class SYNAPSE_CONV_trace_sstep(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, TIME=8, OTTT_sWS_on = False, first_conv = False, bias=True):
#         super(SYNAPSE_CONV_trace_sstep, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         # self.weight = torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, requires_grad=True)
#         # self.bias = torch.randn(self.out_channels, requires_grad=True)
#         self.weight = nn.Parameter(torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
#         self.bias = nn.Parameter(torch.randn(self.out_channels)) if bias else None
#         # Kaiming 초기화
#         nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

#         if self.bias is not None:
#             nn.init.constant_(self.bias, 0)
        
#         self.TIME = TIME

#         self.OTTT_sWS_on = OTTT_sWS_on
#         # self.first_conv = first_conv # no using in this module

#         self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1)) if self.OTTT_sWS_on == True else None

#     def forward(self, spike):
#         if (self.OTTT_sWS_on == True):
#             weight = self.get_weight()
#         else:
#             weight = self.weight

#         spike = F.conv2d(spike, weight, bias=self.bias, stride=self.stride, padding=self.padding)
#         output_current = spike
#         return output_current

#     def get_weight(self):
#         fan_in = np.prod(self.weight.shape[1:])
#         mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
#         var = torch.var(self.weight, axis=[1, 2, 3], keepdims=True)
#         weight = (self.weight - mean) / ((var * fan_in + 1e-4) ** 0.5)
#         if self.gain is not None:
#             weight = weight * self.gain
#         return weight
    

    
# class SYNAPSE_FC_trace_sstep(nn.Module):
#     def __init__(self, in_features, out_features, TIME=8, OTTT_sWS_on = False, bias=True):
#         super(SYNAPSE_FC_trace_sstep, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = nn.Parameter(torch.randn(self.out_features, self.in_features))
#         self.bias = nn.Parameter(torch.randn(self.out_features)) if bias else None
#         # Xavier 균등 분포 초기화
#         nn.init.xavier_uniform_(self.weight)
#         # 바이어스는 0으로 초기화
#         if self.bias is not None:
#             nn.init.constant_(self.bias, 0)

#         # # Xavier 정규 분포 초기화
#         # nn.init.xavier_normal_(self.weight)
#         # # 바이어스는 0으로 초기화
#         # nn.init.constant_(self.bias, 0)

#         # # ottt
#         # nn.init.normal_(self.weight, 0, 0.01)
#         # nn.init.constant_(self.bias, 0)

#         self.TIME = TIME

#         self.OTTT_sWS_on = OTTT_sWS_on

#         self.gain = nn.Parameter(torch.ones(self.out_features, 1)) if self.OTTT_sWS_on == True else None

#     def forward(self, spike):
#         # print('spike sparsity', self.in_features, self.out_features, torch.count_nonzero(spike.detach()) / (spike.detach()).numel())

#         weight = self.weight if self.OTTT_sWS_on == False else self.get_weight()
#         output_current = F.linear(spike, weight, self.bias)
#         return output_current 
    

#     def get_weight(self):
#         fan_in = np.prod(self.weight.shape[1:])
#         mean = torch.mean(self.weight, axis=[1], keepdims=True)
#         var = torch.var(self.weight, axis=[1], keepdims=True)

#         # # 계산된 평균과 분산으로 weight 정규화
#         # mean = torch.mean(self.weight, dim=1, keepdim=True)
#         # var = torch.var(self.weight, dim=1, keepdim=True)
#         # fan_in = self.in_features  # Fully connected layer의 경우 fan_in은 in_features와 동일
        
#         # 정규화
#         weight = (self.weight - mean) / ((var * fan_in + 1e-4) ** 0.5)
        
#         if self.gain is not None:
#             weight = weight * self.gain
        
#         return weight
        
    
# ############## OTTT Conv trace sstep #######################################
# ############## OTTT Conv trace sstep #######################################
# ############## OTTT Conv trace sstep #######################################

