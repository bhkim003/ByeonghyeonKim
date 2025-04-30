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
    def __init__ (self, v_init , v_decay , v_threshold , v_reset , sg_width, surrogate, BPTT_on, trace_const1=1, trace_const2=0.7, TIME=6, sstep=False, trace_on=False):
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
        
        if self.sstep == True:
            self.time_count = 0

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

                post_spike[t] = FIRE.apply(v - self.v_threshold, self.surrogate, self.sg_width) 

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
                self.v = torch.full_like(input_current, fill_value = self.v_init, dtype = torch.float, requires_grad=False) # v (membrane potential) init
                

            self.v = self.v.detach() * self.v_decay + input_current 
            post_spike = FIRE.apply(self.v - self.v_threshold, self.surrogate, self.sg_width) 
            
            if (self.v_reset >= 0 and self.v_reset < 10000): # soft reset
                self.v = self.v - post_spike.detach() * self.v_threshold
            elif (self.v_reset >= 10000 and self.v_reset < 20000): # hard reset 
                self.v = self.v*(1-post_spike.detach()) + (self.v_reset - 10000)*post_spike.detach()
            
            if self.trace_on == True:
                self.trace = self.trace*self.trace_const2 + post_spike*self.trace_const1

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
                f"trace_on={self.trace_on})")

class FIRE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_minus_threshold, surrogate, sg_width):
        if surrogate == 'sigmoid':
            surrogate = 1
        elif surrogate == 'rectangle':
            surrogate = 2
        elif surrogate == 'rough_rectangle':
            surrogate = 3
        elif surrogate == 'hard_sigmoid':
            surrogate = 4
        else:
            assert False, 'surrogate doesn\'t exist'
        ctx.save_for_backward(v_minus_threshold,
                            torch.tensor([surrogate], requires_grad=False),
                            torch.tensor([sg_width], requires_grad=False)) # save before reset
        return (v_minus_threshold >= 0.0).to(torch.float)
    @staticmethod
    def backward(ctx, grad_output):
        v_minus_threshold, surrogate, sg_width = ctx.saved_tensors
        # v_minus_threshold=v_minus_threshold.item() #ValueError: only one element tensors can be converted to Python scalars
        surrogate=surrogate.item()
        sg_width=sg_width.item()

        if (surrogate == 1):
            #===========surrogate gradient function (sigmoid)
            alpha = sg_width 
            sig = torch.sigmoid(alpha*v_minus_threshold)
            grad_input = alpha*sig*(1-sig)*grad_output
        elif (surrogate == 2):
            # ===========surrogate gradient function (rectangle)
            grad_input = grad_output * (v_minus_threshold.abs() < sg_width/2).float() / sg_width
        elif (surrogate == 3):
            #===========surrogate gradient function (rough rectangle)
            grad_output[v_minus_threshold.abs() > sg_width/2] = 0
            grad_input = grad_output / sg_width
        elif (surrogate == 4):
            #===========surrogate gradient function (hard sigmoid)
            alpha = sg_width 
            sig = torch.clamp(alpha*v_minus_threshold * 0.2 + 0.5, min=0, max=1)
            grad_input = alpha*sig*(1-sig)*grad_output
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
