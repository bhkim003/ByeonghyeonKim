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

######## LIF Neuron #####################################################
######## LIF Neuron #####################################################
######## LIF Neuron #####################################################
class LIF_layer(nn.Module):
    def __init__ (self, v_init , v_decay , v_threshold , v_reset , sg_width, surrogate, BPTT_on):
        super(LIF_layer, self).__init__()
        self.v_init = v_init
        self.v_decay = v_decay
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.sg_width = sg_width
        self.surrogate = surrogate
        self.BPTT_on = BPTT_on

    def forward(self, input_current):
        v = torch.full_like(input_current, fill_value = self.v_init, dtype = torch.float) # v (membrane potential) init
        post_spike = torch.full_like(input_current, fill_value = self.v_init, device=input_current.device, dtype = torch.float) # v (membrane potential) init
        # i와 v와 post_spike size는 여기서 다 같음: [Time, Batch, Channel, Height, Width] 
        Time = v.shape[0]
        for t in range(Time):
            # leaky하고 input_current 더하고 fire하고 reset까지 (backward직접처리)
            post_spike[t], v[t] = LIF_METHOD.apply(input_current[t], v[t], 
                                            self.v_decay, self.v_threshold, self.v_reset, self.sg_width, self.surrogate, self.BPTT_on) 
        return post_spike 



class LIF_METHOD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_current_one_time, v_one_time, v_decay, v_threshold, v_reset, sg_width, surrogate, BPTT_on):
        v_one_time = v_one_time * v_decay + input_current_one_time # leak + pre-synaptic current integrate
        spike = (v_one_time >= v_threshold).float() #fire
        if surrogate == 'sigmoid':
            surrogate = 1
        elif surrogate == 'rectangle':
            surrogate = 2
        elif surrogate == 'rough_rectangle':
            surrogate = 3
        else:
            pass

        if (BPTT_on == True):
            BPTT_on = 1
        else:
            BPTT_on = 0

        ctx.save_for_backward(v_one_time, torch.tensor([v_decay], requires_grad=False), 
                            torch.tensor([v_threshold], requires_grad=False), 
                            torch.tensor([v_reset], requires_grad=False), 
                            torch.tensor([sg_width], requires_grad=False),
                            torch.tensor([surrogate], requires_grad=False),
                            torch.tensor([BPTT_on], requires_grad=False)) # save before reset
        
        v_one_time = (v_one_time - spike * v_threshold).clamp_min(0) # reset
        return spike, v_one_time

    @staticmethod
    def backward(ctx, grad_output_spike, grad_output_v):
        v_one_time, v_decay, v_threshold, v_reset, sg_width, surrogate, BPTT_on = ctx.saved_tensors
        v_decay=v_decay.item()
        v_threshold=v_threshold.item()
        v_reset=v_reset.item()
        sg_width=sg_width.item()
        surrogate=surrogate.item()
        BPTT_on=BPTT_on.item()

        grad_input_current = grad_output_spike.clone()
        # grad_temp_v = grad_output_v.clone() # not used

        ################ select one of the following surrogate gradient functions ################
        if (surrogate == 1):
            #===========surrogate gradient function (sigmoid)
            sig = torch.sigmoid((v_one_time - v_threshold))
            grad_input_current *= 4*sig*(1-sig)
            # grad_x = grad_output * (1. - sgax) * sgax * ctx.alpha

        elif (surrogate == 2):
            # ===========surrogate gradient function (rectangle)
            grad_input_current *= ((v_one_time - v_threshold).abs() < sg_width/2).float() / sg_width

        elif (surrogate == 3):
            #===========surrogate gradient function (rough rectangle)
            grad_input_current[(v_one_time - v_threshold).abs() > sg_width/2] = 0
        else: 
            assert False, 'surrogate doesn\'t exist'
        ###########################################################################################

        ## if BPTT_on == 1, then second return value is not None
        if (BPTT_on == 1):
            grad_output_v = v_decay * grad_output_v 
        else:
            grad_output_v = None 
        
        return grad_input_current, grad_output_v, None, None, None, None, None, None
######## LIF Neuron #####################################################
######## LIF Neuron #####################################################
######## LIF Neuron #####################################################
