import sys
import os
from cv2 import NONE_POLISHER
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
from modules.old_fashioned import *
from modules.ae_network import *



    
class LIF_layer(nn.Module):
    def __init__ (self, v_init , v_decay , v_threshold , v_reset , sg_width, surrogate, BPTT_on, trace_const1=1, trace_const2=0.7, TIME=6, sstep=False, trace_on=False, layer_count = 0, scale_exp = []):
        super(LIF_layer, self).__init__()
        self.v_init = v_init
        self.v_decay = v_decay
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.sg_width = sg_width
        self.surrogate = surrogate
        self.BPTT_on = BPTT_on
        self.trace_const1 = trace_const1
        self.trace_const2 = trace_const2
        self.TIME = TIME
        self.sstep = sstep
        self.trace_on = trace_on # sstep일때만 통함
        self.past_post_spike = None
        self.layer_count = layer_count
        # self.quantize_bit_list = [16,16,16]
        self.quantize_bit_list = [15,15,15]
        # self.quantize_bit_list = []
        self.scale_exp = scale_exp
        self.v_exp = None

        
        self.sg_bit = 4
        # self.sg_bit = 0
        print(f'\n\n\nLIF {self.layer_count} sg_bit {self.sg_bit}\n\n')

        
        if len(self.quantize_bit_list) != 0:
            if self.layer_count == 1:
                self.v_bit = self.quantize_bit_list[0]
                if self.scale_exp != []:
                    self.v_exp = min(self.scale_exp[0][0], self.scale_exp[0][1])
            elif self.layer_count == 2:
                self.v_bit = self.quantize_bit_list[1]
                if self.scale_exp != []:
                    self.v_exp = min(self.scale_exp[1][0], self.scale_exp[1][1])
            elif self.layer_count == 3:
                assert False
                self.v_bit = self.quantize_bit_list[2]
                if self.scale_exp != []:
                    self.v_exp = min(self.scale_exp[2][0], self.scale_exp[2][1])
            else:
                assert False, 'layer_count should be 1, 2, or 3'
        else:
            self.v_bit = 0

        print(f"++++++++++++++++++++++++\n\n lif layer {self.layer_count} v_bit: {self.v_bit}, v_exp: {self.v_exp}\n\n++++++++++++++++++++++++++++++++++++++++++++++")

        
        if self.sstep == True:
            self.time_count = 0

        self.v_distribution_box = []
        for i in range(self.TIME):
            self.v_distribution_box.append([])

            

    def forward(self, input_current):
        if self.sstep == False:
            Time = input_current.shape[0]
            v = torch.full_like(input_current[0], fill_value = self.v_init, dtype = torch.float, requires_grad=False)
            post_spike = torch.full_like(input_current, fill_value = 0.0, device=input_current.device, dtype = torch.float, requires_grad=False) 
            
            for t in range(Time):
                if (self.BPTT_on == True):
                    v = v * self.v_decay + input_current[t]
                    # v = V_DECAY.apply(v, self.v_decay, self.BPTT_on) + input_current[t]
                else:
                    v = v.detach() * self.v_decay + input_current[t]
                    # v = V_DECAY.apply(v, self.v_decay, self.BPTT_on) + input_current[t]

                post_spike[t] = FIRE.apply(v - self.v_threshold, self.surrogate, self.sg_width, self.sg_bit) 

                if (self.v_reset >= 0 and self.v_reset < 10000): # soft reset
                    v = v - post_spike[t].detach() * self.v_threshold
                elif (self.v_reset >= 10000 and self.v_reset < 20000): # hard reset 
                    v = v*(1-post_spike[t].detach()) + (self.v_reset - 10000)*post_spike[t].detach()
            return post_spike
        
        else: #singlestep mode
            self.time_count = self.time_count + 1

            if self.time_count == 1:
                if self.trace_on == True:
                    self.trace = torch.full_like(input_current, fill_value = 0.0, dtype = torch.float, requires_grad=False) 
                    self.past_post_spike= torch.full_like(input_current, fill_value = 0.0, dtype = torch.float, requires_grad=False) 
                self.v = torch.full_like(input_current, fill_value = self.v_init, dtype = torch.float, requires_grad=False) # v (membrane potential) init
                

            self.v = self.v.detach() * self.v_decay + input_current 
            # 여기서 v_decay 0.5가 곱해지면서 밑의 quantization에서 0.5는 +1, -0.5는 -1된다. 이거 잘짜셈.
            # 그러니까 rtl짤때 2'complement에서 0.5 decay처리할 때 lsb가 양수일때는 1더해줘야되고, 음수일때는 걍 버리면되는거임.

            if self.v_bit > 0:
                self.v = V_Quantize.apply(self.v, self.v_bit, self.v_exp)
            # print(f"Unique elements in v: {self.v.unique().numel()}: {self.v.unique().tolist()}")

            # self.v_distribution_box[self.time_count-1].append(self.v.detach().clone())

            post_spike = FIRE.apply(self.v - self.v_threshold, self.surrogate, self.sg_width, self.sg_bit) 
            
            if (self.v_reset >= 0 and self.v_reset < 10000): # soft reset
                self.v = self.v - post_spike.detach() * self.v_threshold
                if self.trace_on == True:
                    self.trace = self.trace*self.trace_const2 + post_spike*self.trace_const1

            elif (self.v_reset >= 10000 and self.v_reset < 20000): # hard reset 
                self.v = self.v*(1-post_spike.detach()) + (self.v_reset - 10000)*post_spike.detach()
                if self.trace_on == True:
                    # self.trace에다가  self.past_post_spike가 1인 곳은 0으로 만들기
                    self.trace = self.trace*(1-self.past_post_spike)*self.trace_const2 + post_spike*self.trace_const1
                    self.past_post_spike = post_spike.detach().clone()

            if (self.time_count == self.TIME):
                self.time_count = 0


            if self.trace_on == True:
                self.trace = self.trace.detach()
                return [post_spike, self.trace] 
            else: 
                return post_spike


    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"v_init={self.v_init}, "
                f"v_decay={self.v_decay}, "
                f"v_threshold={self.v_threshold}, "
                f"v_reset={self.v_reset}, "
                f"sg_width={self.sg_width}, "
                f"surrogate={self.surrogate}, "
                f"BPTT_on={self.BPTT_on}, "
                f"trace_const1={self.trace_const1}, "
                f"trace_const2={self.trace_const2}, "
                f"TIME={self.TIME}, "
                f"sstep={self.sstep}, "
                f"trace_on={self.trace_on}, "
                f"layer_count={self.layer_count}, "
                f"scale_exp={self.scale_exp})")

                

class FIRE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_minus_threshold, surrogate, sg_width, sg_bit):
        if surrogate == 'sigmoid':
            surrogate = 1
        elif surrogate == 'rectangle':
            surrogate = 2
        elif surrogate == 'rough_rectangle':
            surrogate = 3
        elif surrogate == 'hard_sigmoid':
            surrogate = 4
        elif surrogate == 'one':
            surrogate = 5
        elif surrogate == 'one_if_over_threshold':
            surrogate = 6
        else:
            assert False, 'surrogate doesn\'t exist'
        ctx.save_for_backward(v_minus_threshold,
                            torch.tensor([surrogate], requires_grad=False),
                            torch.tensor([sg_width], requires_grad=False),
                            torch.tensor([sg_bit], requires_grad=False)) # save before reset
        return (v_minus_threshold >= 0.0).to(torch.float)
    @staticmethod
    def backward(ctx, grad_output):
        v_minus_threshold, surrogate, sg_width, sg_bit = ctx.saved_tensors
        # v_minus_threshold=v_minus_threshold.item() #ValueError: only one element tensors can be converted to Python scalars
        surrogate=surrogate.item()
        sg_width=sg_width.item()
        sg_bit=sg_bit.item()

        if (surrogate == 1):
            #===========surrogate gradient function (sigmoid)
            alpha = sg_width 
            sig = torch.sigmoid(alpha*v_minus_threshold)
            grad_input = alpha*sig*(1-sig)*grad_output
        elif (surrogate == 2):
            # ===========surrogate gradient function (rectangle)
            grad_input = grad_output * (v_minus_threshold.abs() <= sg_width/2).float() / sg_width
        elif (surrogate == 3):
            #===========surrogate gradient function (rough rectangle)
            grad_output[v_minus_threshold.abs() > sg_width/2] = 0
            grad_input = grad_output / sg_width
        elif (surrogate == 4):
            # #===========surrogate gradient function (hard sigmoid)
            # alpha = sg_width 
            # sig = torch.clamp(alpha*v_minus_threshold * 0.2 + 0.5, min=0, max=1)
            # sg_temp = alpha*sig*(1-sig) # max 1.0 여기까지는

            # if sg_bit > 0:
            #     sg_temp_max = 1.0
            #     sg_temp_bit = 4 # 이렇게하면 4비트로 하면 000 001 010 011 100 까지만 표기 가능
            #     sg_temp_max -= 2 ** (-(sg_temp_bit - 1))  # 최대 표현 가능한 값
            #     sg_temp *= sg_temp_max
            #     scale_sg_temp = 2**math.ceil(math.log2(sg_temp_max / (2**(sg_temp_bit-1) -1))) 
            #     sg_temp = torch.clamp((sg_temp / scale_sg_temp + 0).round(), -2**(sg_temp_bit-1) + 1, 2**(sg_temp_bit-1) - 1) * scale_sg_temp

            # grad_input = sg_temp*grad_output


            # ===========surrogate gradient function (hard sigmoid)
            alpha = sg_width  #alpha클수록 좁아짐
            sig = torch.clamp(alpha*v_minus_threshold * 0.2 + 0.5, min=0, max=1)
            # sig = torch.sigmoid(alpha*v_minus_threshold)
            sg_temp = 4.0*sig*(1-sig) # max 1.0 여기까지는

            if sg_bit > 0:
                sg_temp_max = 1.0
                sg_bit = 4 # 이렇게하면 4비트로 하면 000 001 010 011 100 까지만 표기 가능
                sg_temp_max -= 2 ** (-(sg_bit - 1))  # 최대 표현 가능한 값
                sg_temp *= sg_temp_max
                scale_sg_temp = 2**math.ceil(math.log2(sg_temp_max / (2**(sg_bit-1) -1))) 
                sg_temp = torch.clamp((sg_temp / scale_sg_temp).round(), -2**(sg_bit-1) + 1, 2**(sg_bit-1) - 1) * scale_sg_temp

            grad_input = sg_temp*grad_output
        elif (surrogate == 5):
            #===========surrogate gradient function (just one)
            grad_input = grad_output / sg_width
        elif (surrogate == 6):
            #===========surrogate gradient function (just one_if_over_threshold) # v_minus_threshold>=0.0
            grad_output[v_minus_threshold < 0.0] = 0
            grad_input = grad_output / sg_width
        else:
            assert False, 'surrogate doesn\'t exist'


        return grad_input, None, None, None


class V_Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, v_bit, v_exp):
        # percentile=0 
        percentile=0.999
        # percentile=0.99
        # percentile=0.95
        
        if v_exp == None:
            max_v = v.abs().max().item()
            if percentile > 0:
                max_v = torch.quantile(v.abs().flatten(), percentile).item()
            assert max_v > 0, 'max_v should be greater than 0'
            scale_v = 2**math.ceil(math.log2(max_v / (2**(v_bit-1) -1))) 
        else:
            scale_v = 2**v_exp

        q_v = torch.clamp((v / scale_v).round(), -2**(v_bit-1)+1, 2**(v_bit-1) - 1) * scale_v
        # q_v = torch.clamp(round_hardware_good(v / scale_v), -2**(v_bit-1), 2**(v_bit-1) - 1) * scale_v
        # q_v = torch.clamp(round_away_from_zero(v / scale_v), -2**(v_bit-1), 2**(v_bit-1) - 1) * scale_v
        # q_v = torch.clamp((v / scale_v).round(), -2**(v_bit-1), 2**(v_bit-1) - 1) * scale_v
        # q_v = torch.clamp(torch.trunc(v / scale_v), -2**(v_bit-1), 2**(v_bit-1) - 1) * scale_v

        return q_v

    @staticmethod
    def backward(ctx, grad_output):
        # 그냥 identity gradient 전달 (straight-through estimator 방식)
        grad_input = grad_output.clone()
        return grad_input, None, None
    

# class V_DECAY(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, v, v_decay, BPTT_on):
#         ctx.save_for_backward(torch.tensor([v_decay], requires_grad=False),
#                               torch.tensor([BPTT_on], requires_grad=False)) # save before reset
#         return v*v_decay

#     @staticmethod
#     def backward(ctx, grad_output):
#         v_decay, BPTT_on = ctx.saved_tensors
#         v_decay=v_decay.item()
#         BPTT_on=BPTT_on.item()
        
#         v_decay = v_decay if BPTT_on else 0.0
#         grad_input = grad_output * v_decay

#         return grad_input, None, None
    

    

# ####### NDA Neuron #####################################################
# ####### NDA Neuron #####################################################
# ####### NDA Neuron #####################################################
# ####### NDA Neuron #####################################################
# def fire_function(gamma):
#     class ZIF(torch.autograd.Function): # zero is firing
#         @staticmethod
#         def forward(ctx, input):
#             out = (input >= 0).float()
#             # gradient를 위해 input을 저장하는 코드인듯 ㅇㅇ
#             # 예의주시해봐
#             ctx.save_for_backward(input)
#             return out


#         # 걍 근데 이건 1/2보다 작으면 1
#         # 1/2보다 크면 0인데?
#         @staticmethod
#         def backward(ctx, grad_output):
#             # forward에서 저장해놨던 input가져오는거임
#             (input, ) = ctx.saved_tensors
#             grad_input = grad_output.clone()
#             tmp = (input.abs() < gamma/2).float() / gamma
#             # 사각형 형태의 surrogate gradient임.
#             # 1/2 0    ----
#             # -1/2 0   |  |
#             # 1/2 1    ----
#             # -1/2 1
#             grad_input = grad_input * tmp
#             return grad_input, None

#     return ZIF.apply


# class LIFSpike(nn.Module):
#     def __init__(self, thresh=0.5, tau=0.25, gamma=1.0):
#         super(LIFSpike, self).__init__()
#         self.thresh = thresh
#         self.tau = tau
#         self.gamma = gamma

#     def forward(self, x):
#         mem = torch.zeros_like(x[:, 0])
#         # print('\n\nmem size', mem.size())
#         # print(x)
#         # print(x[0])
#         # print(x[0][0])
#         # print(x[0][0][0])
#         # print('xsize!!',x.size())

#         # mem size torch.Size([64, 512, 6, 6])
#         # xsize!! torch.Size([64, 10, 512, 6, 6])

#         spikes = []
#         T = x.shape[1]
#         for t in range(T):
#             mem = mem * self.tau + x[:, t, ...] #걍 인덱스별로 각각 덧셈
#             spike = fire_function(self.gamma)(mem - self.thresh)
#             # mem = (1 - spike.detach()) * mem.detach() #spike나감과 동시에 reset
#             mem = (1 - spike) * mem #spike나감과 동시에 reset
#             spikes.append(spike)

#         # print('spikes size',spikes.size())
#         # print('torch.stack(spikes,dim=1)', torch.stack(spikes, dim=1).size())
            
#         # print('xsize out',torch.stack(spikes, dim=1).size())
        
#         return torch.stack(spikes, dim=1)
# ####### NDA Neuron #####################################################
# ####### NDA Neuron #####################################################
# ####### NDA Neuron #####################################################
# ####### NDA Neuron #####################################################



######## LIF Neuron trace single step #####################################################
######## LIF Neuron trace single step #####################################################
######## LIF Neuron trace single step #####################################################
# class LIF_layer_trace_sstep(nn.Module):
#     def __init__ (self, v_init , v_decay , v_threshold , v_reset , sg_width, surrogate, BPTT_on, trace_const1=1, trace_const2=0.7, TIME=6):
#         super(LIF_layer_trace_sstep, self).__init__()
#         self.v_init = v_init
#         self.v_decay = v_decay
#         self.v_threshold = v_threshold
#         self.v_reset = v_reset
#         self.sg_width = sg_width
#         self.surrogate = surrogate
#         self.BPTT_on = BPTT_on
#         self.trace_const1 = trace_const1
#         self.trace_const2 = trace_const2

#         self.TIME = TIME
#         self.time_count = 0

#     def forward(self, input_current):
#         self.time_count = self.time_count + 1

#         if self.time_count == 1:
#             self.trace = torch.full_like(input_current, fill_value = 0.0, dtype = torch.float, requires_grad=False) # v (membrane potential) init
#             self.v = torch.full_like(input_current, fill_value = self.v_init, dtype = torch.float, requires_grad=False) # v (membrane potential) init

#         self.v = self.v.detach() * self.v_decay + input_current 
#         post_spike = FIRE.apply(self.v - self.v_threshold, self.surrogate, self.sg_width) 
#         if (self.v_reset >= 0 and self.v_reset < 10000): # soft reset
#             self.v = self.v - post_spike.detach() * self.v_threshold
#         elif (self.v_reset >= 10000 and self.v_reset < 20000): # hard reset 
#             self.v = self.v*(1-post_spike.detach()) + (self.v_reset - 10000)*post_spike.detach()
#         out_trace = self.trace*self.trace_const2 + post_spike*self.trace_const1

#         if (self.time_count == self.TIME):
#             self.v = self.v.detach()
#             self.trace = self.trace.detach()
#             self.time_count = 0
        
#         return [post_spike, out_trace] 
######## LIF Neuron trace single step #####################################################
######## LIF Neuron trace single step #####################################################
######## LIF Neuron trace single step #####################################################
    







######## QCFS Neuron ######################################################################
######## QCFS Neuron ######################################################################
######## QCFS Neuron ######################################################################
######## QCFS Neuron ######################################################################

class QCFS_GradFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
class QCFS_IF(nn.Module):
    def __init__(self,  L=8, thresh=8.0):
        super(QCFS_IF, self).__init__()
        self.thresh = nn.Parameter(torch.tensor([thresh]), requires_grad=True)
        self.L = L
        self.shift = 0.5
        print('thresh', self.thresh.item(), 'L', self.L)

    def forward(self, x):
        x = x / self.thresh
        x = torch.clamp(x, 0, 1) # 이게 한 줄 밑에 가야되는 거 아님? 논문상으론 그런데??? 나중에 실험해보자
        x = QCFS_GradFloor.apply(x*self.L+self.shift)/self.L
        x = x * self.thresh
        return x
######## QCFS Neuron ######################################################################
######## QCFS Neuron ######################################################################
######## QCFS Neuron ######################################################################
######## QCFS Neuron ######################################################################
