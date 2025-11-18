import sys
import os


import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

print('1')


import argparse
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
print('2')

import torchvision
import torchvision.datasets
import torchvision.transforms as transforms

import numpy as np

import matplotlib
matplotlib.use('Agg') # GUI 백엔드가 아닌 파일 출력용 백엔드로 설정

import matplotlib.pyplot as plt

import time

from snntorch import spikegen
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML

from tqdm import tqdm
print('3')

from apex.parallel import DistributedDataParallel as DDP
print('11')

import random
import datetime
print('12')

import json

from sklearn.utils import shuffle
print('13')

''' 레퍼런스
https://spikingjelly.readthedocs.io/zh-cn/0.0.0.0.4/spikingjelly.datasets.html#module-spikingjelly.datasets
https://github.com/GorkaAbad/Sneaky-Spikes/blob/main/datasets.py
https://github.com/GorkaAbad/Sneaky-Spikes/blob/main/how_to.md
https://github.com/nmi-lab/torchneuromorphic
https://snntorch.readthedocs.io/en/latest/snntorch.spikevision.spikedata.html#shd
'''

import snntorch
from snntorch.spikevision import spikedata
print('14')

import modules.spikingjelly;
print('15')
from modules.spikingjelly.datasets.dvs128_gesture import DVS128Gesture
print('16')
from modules.spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from modules.spikingjelly.datasets.n_mnist import NMNIST
# from modules.spikingjelly.datasets.es_imagenet import ESImageNet
from modules.spikingjelly.datasets import split_to_train_test_set
from modules.spikingjelly.datasets.n_caltech101 import NCaltech101
from modules.spikingjelly.datasets import pad_sequence_collate, padded_sequence_mask

import modules.torchneuromorphic as torchneuromorphic
print('4')

import wandb

from torchviz import make_dot
import graphviz
from turtle import shape





print('5')





















































# my module import
from modules import *

# modules 폴더에 새모듈.py 만들면
# modules/__init__py 파일에 form .새모듈 import * 하셈
# 그리고 새모듈.py에서 from modules.새모듈 import * 하셈






print('6')





































































from matplotlib.ft2font import EXTERNAL_STREAM


def my_snn_system(devices = "0,1,2,3",
                    single_step = False, # True # False
                    unique_name = 'main',
                    my_seed = 42,
                    TIME = 10,
                    BATCH = 256,
                    IMAGE_SIZE = 32,
                    which_data = 'CIFAR10',
                    # CLASS_NUM = 10,
                    data_path = '/data2',
                    rate_coding = True,
    
                    lif_layer_v_init = 0.0,
                    lif_layer_v_decay = 0.6,
                    lif_layer_v_threshold = 1.2,
                    lif_layer_v_reset = 0.0,
                    lif_layer_sg_width = 1,

                    # synapse_conv_in_channels = IMAGE_PIXEL_CHANNEL,
                    synapse_conv_kernel_size = 3,
                    synapse_conv_stride = 1,
                    synapse_conv_padding = 1,

                    synapse_trace_const1 = 1,
                    synapse_trace_const2 = 0.6,

                    # synapse_fc_out_features = CLASS_NUM,

                    pre_trained = False,
                    convTrue_fcFalse = True,

                    cfg = [64, 64],
                    net_print = False, # True # False
                    
                    pre_trained_path = "net_save/save_now_net.pth",
                    learning_rate = 0.0001,
                    epoch_num = 200,
                    tdBN_on = False,
                    BN_on = False,

                    surrogate = 'sigmoid',

                    BPTT_on = False,

                    optimizer_what = 'SGD', # 'SGD' 'Adam', 'RMSprop'
                    scheduler_name = 'no',
                    
                    ddp_on = False, # DECREPATED # fALSE

                    dvs_clipping = 1, 
                    dvs_duration = 25_000,


                    DFA_on = False, # True # False
                    trace_on = False, 
                    OTTT_input_trace_on = False, # True # False
                    
                    exclude_class = True, # True # False # gesture에서 10번째 클래스 제외

                    merge_polarities = False, # True # False # tonic dvs dataset 에서 polarities 합치기
                    denoise_on = True, 

                    extra_train_dataset = 0, # DECREPATED # data_loader에서 train dataset을 몇개 더 쓸건지 

                    num_workers = 2,
                    chaching_on = True,
                    pin_memory = True, # True # False
                    
                    UDA_on = False,  # DECREPATED # uda
                    alpha_uda = 1.0, # DECREPATED # uda

                    bias = True,

                    last_lif = False,
                        
                    temporal_filter = 1, 
                    initial_pooling = 1,

                    temporal_filter_accumulation = False,

                    quantize_bit_list=[],
                    scale_exp=[],
                    ):
    ## 함수 내 모든 로컬 변수 저장 ########################################################
    hyperparameters = locals()
    print('param', hyperparameters,'\n')
    hyperparameters['current epoch'] = 0
    ######################################################################################

    ## hyperparameter check #############################################################
    if single_step == True:
        assert BPTT_on == False and tdBN_on == False 
    if tdBN_on == True:
        assert BPTT_on == True
    if pre_trained == True:
        print('\n\n')
        print("Caution! pre_trained is True\n\n"*3)    
    if DFA_on == True:
        assert single_step == True and BPTT_on == False 
    # assert single_step == DFA_on, 'DFA랑 single_step공존하게해라'
    if trace_on:
        assert BPTT_on == False and single_step == True
    if OTTT_input_trace_on == True:
        assert BPTT_on == False and single_step == True #and trace_on == True
    if temporal_filter > 1:
        assert convTrue_fcFalse == False
    ######################################################################################


    

    ## wandb 세팅 ###################################################################
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.config.update(hyperparameters)
    wandb.run.name = f'lr_{learning_rate}_{unique_name}_{which_data}_tstep{TIME}'
    wandb.define_metric("summary_val_acc", summary="max")
    # wandb.run.log_code(".", 
    #                     include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"),
    #                     exclude_fn=lambda path: 'logs/' in path or 'net_save/' in path or 'result_save/' in path or 'trying/' in path or 'wandb/' in path or 'private/' in path or '.git/' in path or 'tonic' in path or 'torchneuromorphic' in path or 'spikingjelly' in path 
    #                     )
    ###################################################################################



    ## gpu setting ##################################################################################################################
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"]= devices
    ###################################################################################################################################


    ## seed setting ##################################################################################################################
    seed_assign(my_seed)
    ###################################################################################################################################
    

    ## data_loader 가져오기 ##################################################################################################################
    # data loader, pixel channel, class num
    train_data_split_indices = []
    train_loader, test_loader, synapse_conv_in_channels, CLASS_NUM, train_data_count = data_loader(
            which_data,
            data_path, 
            rate_coding, 
            BATCH, 
            IMAGE_SIZE,
            ddp_on,
            TIME*temporal_filter, 
            dvs_clipping,
            dvs_duration,
            exclude_class,
            merge_polarities,
            denoise_on,
            my_seed,
            extra_train_dataset,
            num_workers,
            chaching_on,
            pin_memory,
            train_data_split_indices,) 
    synapse_fc_out_features = CLASS_NUM

    print('\nlen(train_loader):', len(train_loader), 'BATCH:', BATCH, 'train_data_count:', train_data_count) 
    print('len(test_loader):', len(test_loader), 'BATCH:', BATCH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\ndevice ==> {device}\n")
    if device == "cpu":
        print("="*50,"\n[WARNING]\n[WARNING]\n[WARNING]\n: cpu mode\n\n","="*50)

    ### network setting #######################################################################################################################
    if (convTrue_fcFalse == False):
        net = REBORN_MY_SNN_FC(cfg, synapse_conv_in_channels*temporal_filter, IMAGE_SIZE//initial_pooling, synapse_fc_out_features,
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
                    trace_on,
                    quantize_bit_list,
                    scale_exp).to(device)
    else:
        net = REBORN_MY_SNN_CONV(cfg, synapse_conv_in_channels, IMAGE_SIZE//initial_pooling,
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
                    trace_on,
                    quantize_bit_list,
                    scale_exp).to(device)

    net = torch.nn.DataParallel(net) 
    
    if pre_trained == True:
        # 1. 전체 state_dict 로드
        checkpoint = torch.load(pre_trained_path)

        # 2. 현재 모델의 state_dict 가져오기
        model_dict = net.state_dict()

        # 3. 'SYNAPSE'가 포함된 key만 필터링 (현재 모델에도 존재하는 key만)
        filtered_dict = {k: v for k, v in checkpoint.items() if ('weight' in k or 'bias' in k) and k in model_dict}

        # 4. 업데이트된 키 출력
        print("🔄 업데이트된 SYNAPSE 관련 레이어들:")
        for k in filtered_dict.keys():
            print(f" - {k}")

        # 5. 모델 dict 업데이트 및 로딩
        model_dict.update(filtered_dict)
        net.load_state_dict(model_dict)
    
    net = net.to(device)
    if (net_print == True):
        print(net)    

    print(f"\n========================================================\nTrainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}\n========================================================\n")
    ####################################################################################################################################
    

    ## wandb logging ###########################################
    # wandb.watch(net, log="all", log_freq = 10) #gradient, parameter logging해줌
    ############################################################

    ## criterion ########################################## # loss 구해주는 친구
    def my_cross_entropy_loss(logits, targets):
        # logits: (batch_size, num_classes)
        # targets: (batch_size,) -> 클래스 인덱스
        log_probs = F.log_softmax(logits, dim=1)  # log(p_i)
        loss = F.nll_loss(log_probs, targets)
        # print(loss.shape)
        return loss
    
    class CustomLossFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, target):
            ctx.save_for_backward(input, target)
            return F.cross_entropy(input, target)

        @staticmethod
        def backward(ctx, grad_output):
            # MAE 스타일의 gradient를 흉내냄
            input, target = ctx.saved_tensors
            input_argmax = input.argmax(dim=1)
            input_one_hot = torch.zeros_like(input).scatter_(1, input_argmax.unsqueeze(1), 1.0)
            target_one_hot = torch.zeros_like(input).scatter_(1, target.unsqueeze(1), 1.0)
            # print('grad_output', grad_output) # 이거 걍 1.0임
            return input_one_hot - target_one_hot, None  # target에는 gradient 없음

    # Wrapper module
    class CustomCriterion(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input, target):
            return CustomLossFunction.apply(input, target)

    # criterion = nn.CrossEntropyLoss().to(device)
    criterion = CustomCriterion().to(device)
    
    # if (OTTT_sWS_on == True):
    #     # criterion = nn.CrossEntropyLoss().to(device)
        # criterion = lambda y_t, target_t: ((1 - 0.05) * F.cross_entropy(y_t, target_t) + 0.05 * F.mse_loss(y_t, F.one_hot(target_t, CLASS_NUM).float())) / TIME 
    #     if which_data == 'DVS_GESTURE':
    #         criterion = lambda y_t, target_t: ((1 - 0.001) * F.cross_entropy(y_t, target_t) + 0.001 * F.mse_loss(y_t, F.one_hot(target_t, CLASS_NUM).float())) / TIME 
    ####################################################

    ## optimizer, scheduler ########################################################################
    class MySGD(torch.optim.Optimizer):
        def __init__(self, params, lr=0.01, momentum=0.0, quantize_bit_list=[], scale_exp=[], net=None):
            if momentum < 0.0 or momentum >= 1.0:
                raise ValueError(f"Invalid momentum value: {momentum}")
            
            defaults = {'lr': lr, 'momentum': momentum}
            super(MySGD, self).__init__(params, defaults)
            self.step_count = 0
            self.quantize_bit_list = quantize_bit_list
            # self.quantize_bit_list = []
            self.scale_exp = scale_exp
            self.param_to_name = {param: name for name, param in net.module.named_parameters()} if net else {}

        @torch.no_grad()
        def step(self):
            """모든 파라미터에 대해 gradient descent 수행"""
            loss = None
            for group in self.param_groups:
                lr = group['lr']
                momentum = group['momentum']
                for param in group['params']:
                    if param.grad is None:
                        continue
                    name = self.param_to_name.get(param, 'unknown')
                    # gradient를 이용해 파라미터 업데이트
                    d_p = param.grad

                    if momentum > 0.0:
                        param_state = self.state[param]
                        if 'momentum_buffer' not in param_state:
                            # momentum buffer 초기화
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(d_p)
                            # buf *= momentum 
                            # buf += d_p
                        d_p = buf

                    dw = -lr*d_p
                                        
                    # if 'layers.7.fc.weight' in name or 'layers.7.fc.bias' in name:
                    #     dw = dw * 0.5

                    if len(self.quantize_bit_list) != 0:
                        if 'layers.1.fc.weight' in name:
                            dw_bit = self.quantize_bit_list[0]
                            if self.scale_exp != []:
                                exp = self.scale_exp[0][0]
                                scale_dw = 2**exp
                            else:
                                max_dw = dw.abs().max().item()
                                assert max_dw > 0, f"max_dw is zero for parameter {param.name if hasattr(param, 'name') else 'unknown'}"
                                scale_dw = 2**math.ceil(math.log2(max_dw / (2**(dw_bit-1) -1)))
                        elif 'layers.1.fc.bias' in name:
                            dw_bit = self.quantize_bit_list[0]
                            if self.scale_exp != []:
                                exp = self.scale_exp[0][1]
                                scale_dw = 2**exp
                            else:
                                max_dw = dw.abs().max().item()
                                assert max_dw > 0, f"max_dw is zero for parameter {param.name if hasattr(param, 'name') else 'unknown'}"
                                scale_dw = 2**math.ceil(math.log2(max_dw / (2**(dw_bit-1) -1)))
                        elif 'layers.4.fc.weight' in name:
                            dw_bit = self.quantize_bit_list[1]
                            if self.scale_exp != []:
                                exp = self.scale_exp[1][0]
                                scale_dw = 2**exp
                            else:
                                max_dw = dw.abs().max().item()
                                assert max_dw > 0, f"max_dw is zero for parameter {param.name if hasattr(param, 'name') else 'unknown'}"
                                scale_dw = 2**math.ceil(math.log2(max_dw / (2**(dw_bit-1) -1)))
                        elif 'layers.4.fc.bias' in name:
                            dw_bit = self.quantize_bit_list[1]
                            if self.scale_exp != []:
                                exp = self.scale_exp[1][1]
                                scale_dw = 2**exp
                            else:
                                max_dw = dw.abs().max().item()
                                assert max_dw > 0, f"max_dw is zero for parameter {param.name if hasattr(param, 'name') else 'unknown'}"
                                scale_dw = 2**math.ceil(math.log2(max_dw / (2**(dw_bit-1) -1)))
                        elif 'layers.7.fc.weight' in name:
                            dw_bit = self.quantize_bit_list[2]
                            if self.scale_exp != []:
                                exp = self.scale_exp[2][0]
                                scale_dw = 2**exp
                            else:
                                max_dw = dw.abs().max().item()
                                assert max_dw > 0, f"max_dw is zero for parameter {param.name if hasattr(param, 'name') else 'unknown'}"
                                scale_dw = 2**math.ceil(math.log2(max_dw / (2**(dw_bit-1) -1)))
                        elif 'layers.7.fc.bias' in name:
                            dw_bit = self.quantize_bit_list[2]
                            if self.scale_exp != []:
                                exp = self.scale_exp[2][1]
                                scale_dw = 2**exp
                                
                            else:
                                max_dw = dw.abs().max().item()
                                assert max_dw > 0, f"max_dw is zero for parameter {param.name if hasattr(param, 'name') else 'unknown'}"
                                scale_dw = 2**math.ceil(math.log2(max_dw / (2**(dw_bit-1) -1)))
                        else:
                            assert False, f"Unknown parameter name: {name}"


                        # print(f'dw_bit{dw_bit}, exp{exp}')
                        # print(f'name {name}, d_p: {d_p.shape}, unique elements: {d_p.unique().numel()}, values: {d_p.unique().tolist()}')
                        # print(f'name {name}, dw: {dw.shape}, unique elements: {dw.unique().numel()}, values: {dw.unique().tolist()}')
                        # dw = torch.clamp((dw / scale_dw + 0).round(), -2**(dw_bit-1) + 1, 2**(dw_bit-1) - 1) * scale_dw
                        dw = torch.clamp(round_away_from_zero(dw / scale_dw + 0), -2**(dw_bit-1) + 1, 2**(dw_bit-1) - 1) * scale_dw
                        # print(f'name {name}, dw_post: {dw.shape}, unique elements: {dw.unique().numel()}, values: {dw.unique().tolist()}')

                    if 'layers.1.fc.weight' in name:
                        ooo_fifo = 2
                    elif 'layers.4.fc.weight' in name:
                        ooo_fifo = 1
                    elif 'layers.7.fc.weight' in name:
                        ooo_fifo = 0
                    else:
                        assert False

                    if ooo_fifo > 0:
                        # ====== FIFO 처리 ======
                        param_state = self.state[param]
                        if 'fifo_buffer' not in param_state:
                            param_state['fifo_buffer'] = []

                        fifo = param_state['fifo_buffer']
                        fifo.append(dw.clone())  # clone() to detach from current graph

                        if len(fifo) == ooo_fifo+1:
                            oldest_dw = fifo.pop(0)
                            param.add_(oldest_dw)
                    else: 
                        param.add_(dw)
                        # param -= dw 위 연산이랑 다름. inmemory연산이라 좀 다른 듯
 
            return loss
    
    if(optimizer_what == 'SGD'):
        optimizer = MySGD(net.parameters(), lr=learning_rate, momentum=0.0, quantize_bit_list=quantize_bit_list, scale_exp=scale_exp, net=net)
        # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.0)
        print(optimizer)
    elif(optimizer_what == 'Adam'):
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        # optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)
        # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate/256 * BATCH, weight_decay=1e-4)
        # optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0, betas=(0.9, 0.999))
    elif(optimizer_what == 'RMSprop'):
        pass


    if (scheduler_name == 'StepLR'):
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif (scheduler_name == 'ExponentialLR'):
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif (scheduler_name == 'ReduceLROnPlateau'):
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    elif (scheduler_name == 'CosineAnnealingLR'):
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=50)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=epoch_num)
    elif (scheduler_name == 'OneCycleLR'):
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(train_loader), epochs=epoch_num)
    else:
        pass # 'no' scheduler
    ## optimizer, scheduler ########################################################################


    tr_acc = 0
    tr_correct = 0
    tr_total = 0
    tr_acc_best = 0
    tr_epoch_loss_temp = 0
    tr_epoch_loss = 0
    val_acc_best = 0
    val_acc_now = 0
    val_loss = 0
    iter_of_val = False
    total_backward_count = 0
    real_backward_count = 0
    #======== EPOCH START ==========================================================================================
    for epoch in range(epoch_num):
        epoch_start_time = time.time()
        print('total_backward_count', total_backward_count, 'real_backward_count',real_backward_count, f'{100*real_backward_count/(total_backward_count+0.00000001):7.3f}%')
        if epoch == 1:
            for name, module in net.named_modules():
                if isinstance(module, Feedback_Receiver):
                    print(f"[{name}] weight_fb parameter count: {module.weight_fb.numel():,}")

        max_val_box = []
        max_val_scale_exp_8bit_box = []
        max_val_scale_exp_16bit_box = []
        perc_95_box = []
        perc_95_scale_exp_8bit_box = []
        perc_95_scale_exp_16bit_box = []
        perc_99_box = []
        perc_99_scale_exp_8bit_box = []
        perc_99_scale_exp_16bit_box = []
        perc_999_box = []
        perc_999_scale_exp_8bit_box = []
        perc_999_scale_exp_16bit_box = []
        ##### weight 프린트 ######################################################################
        for name, param in net.module.named_parameters():
            if ('weight' in name or 'bias' in name) and ('1' in name or '4' in name or '7' in name):
                
                data = param.detach().cpu().numpy().flatten()
                abs_data = np.abs(data)

                # 통계량 계산
                mean = np.mean(data)
                std = np.std(data)
                abs_mean = np.mean(abs_data)
                abs_std = np.std(abs_data)
                eps = 1e-15

                # 절대값 기반 max, percentiles
                max_val = abs_data.max()
                max_val_scale_exp_8bit = math.ceil(math.log2((eps+max_val)/ (2**(8-1) -1)))
                max_val_scale_exp_16bit = math.ceil(math.log2((eps+max_val)/ (2**(16-1) -1)))
                perc_95 = np.percentile(abs_data, 95)
                perc_95_scale_exp_8bit = math.ceil(math.log2((eps+perc_95)/ (2**(8-1) -1)))
                perc_95_scale_exp_16bit = math.ceil(math.log2((eps+perc_95)/ (2**(16-1) -1)))
                perc_99 = np.percentile(abs_data, 99)
                perc_99_scale_exp_8bit = math.ceil(math.log2((eps+perc_99)/ (2**(8-1) -1)))
                perc_99_scale_exp_16bit = math.ceil(math.log2((eps+perc_99)/ (2**(16-1) -1)))
                perc_999 = np.percentile(abs_data, 99.9)
                perc_999_scale_exp_8bit = math.ceil(math.log2((eps+perc_999)/ (2**(8-1) -1)))
                perc_999_scale_exp_16bit = math.ceil(math.log2((eps+perc_999)/ (2**(16-1) -1)))
                
                max_val_box.append(max_val)
                max_val_scale_exp_8bit_box.append(max_val_scale_exp_8bit)
                max_val_scale_exp_16bit_box.append(max_val_scale_exp_16bit)
                perc_95_box.append(perc_95)
                perc_95_scale_exp_8bit_box.append(perc_95_scale_exp_8bit)
                perc_95_scale_exp_16bit_box.append(perc_95_scale_exp_16bit)
                perc_99_box.append(perc_99)
                perc_99_scale_exp_8bit_box.append(perc_99_scale_exp_8bit)
                perc_99_scale_exp_16bit_box.append(perc_99_scale_exp_16bit)
                perc_999_box.append(perc_999)
                perc_999_scale_exp_8bit_box.append(perc_999_scale_exp_8bit)
                perc_999_scale_exp_16bit_box.append(perc_999_scale_exp_16bit)




                # if epoch % 5 == 0 or epoch < 3:
                #     print("=> Plotting weight and bias distributions...")
                #     # 그래프 그리기
                #     plt.figure(figsize=(6, 4))
                #     plt.hist(data, bins=100, alpha=0.7, color='skyblue')
                #     plt.axvline(x=max_val, color='red', linestyle='--', label=f'Max: {max_val:.4f}')
                #     plt.axvline(x=-max_val, color='red', linestyle='--')
                #     plt.axvline(x=perc_95, color='green', linestyle='--', label=f'95%: {perc_95:.4f}')
                #     plt.axvline(x=-perc_95, color='green', linestyle='--')
                #     plt.axvline(x=perc_99, color='orange', linestyle='--', label=f'99%: {perc_99:.4f}')
                #     plt.axvline(x=-perc_99, color='orange', linestyle='--')
                #     plt.axvline(x=perc_999, color='purple', linestyle='--', label=f'99.9%: {perc_999:.4f}')
                #     plt.axvline(x=-perc_999, color='purple', linestyle='--')
                    
                #     # 제목에 통계값 포함
                #     title = (
                #         f"{name}, Epoch {epoch}\n"
                #         f"mean={mean:.4f}, std={std:.4f}, "
                #         f"|mean|={abs_mean:.4f}, |std|={abs_std:.4f}\n"
                #         f"Scale 8bit max = { max_val_scale_exp_8bit}, "
                #         f"Scale 16bit max = {max_val_scale_exp_16bit}\n"
                #         f"Scale 8bit p999 = {perc_999_scale_exp_8bit }, "
                #         f"Scale 16bit p999 = {perc_999_scale_exp_16bit }\n"
                #         f"Scale 8bit p99 = {perc_99_scale_exp_8bit }, "
                #         f"Scale 16bit p99 = { perc_99_scale_exp_16bit}\n"
                #         f"Scale 8bit p95 = { perc_95_scale_exp_8bit}, "
                #         f"Scale 16bit p95 = { perc_95_scale_exp_16bit}"
                #     )
                #     plt.title(title)
                #     plt.xlabel('Value')
                #     plt.ylabel('Frequency')
                #     plt.grid(True)
                #     plt.legend()
                #     plt.tight_layout()
                #     plt.show()
        ##### weight 프린트 ######################################################################

        ####### iterator : input_loading & tqdm을 통한 progress_bar 생성###################
        iterator = enumerate(train_loader, 0)
        # iterator = tqdm(iterator, total=len(train_loader), desc='train', dynamic_ncols=True, position=0, leave=True)
        ##################################################################################   

        ###### ITERATION START ##########################################################################################################
        for i, data in iterator:
            net.train() # train 모드로 바꿔줘야함
            ### data loading & semi-pre-processing ################################################################################
            if len(data) == 2:
                inputs, labels = data
                # 처리 로직 작성
            elif len(data) == 3:
                inputs, labels, x_len = data
            else:
                assert False, 'data length is not 2 or 3'
            #######################################################################################################################
                
            ## batch 크기 ######################################
            real_batch = labels.size(0)
            ###########################################################

            # 차원 전처리
            ###########################################################################################################################        
            if (which_data == 'DVS_CIFAR10' or which_data == 'DVS_GESTURE' or which_data == 'DVS_GESTURE_TONIC' or which_data == 'DVS_CIFAR10_2' or which_data == 'NMNIST' or which_data == 'NMNIST_TONIC' or which_data == 'N_CALTECH101' or which_data == 'n_tidigits' or which_data == 'heidelberg'):
                inputs = inputs.permute(1, 0, 2, 3, 4)
            elif rate_coding == True :
                inputs = spikegen.rate(inputs, num_steps=TIME)
            else :
                inputs = inputs.repeat(TIME, 1, 1, 1, 1)
            # inputs: [Time, Batch, Channel, Height, Width]  
            ####################################################################################################################### 
                


            # if i % 1000 == 999:
            #     # SYNAPSE_FC에 있는 sparsity_print_and_reset() 실행
            #     for name, module in net.module.named_modules():
            #         if isinstance(module, SYNAPSE_FC):
            #             module.sparsity_print_and_reset()

                            
            ## initial pooling #######################################################################
            if (initial_pooling > 1):
                pool = nn.MaxPool2d(kernel_size=2)
                num_pooling_layers = int(math.log2(initial_pooling))
                # Time, Batch, Channel 차원은 그대로 두고, Height, Width 차원에 대해서만 pooling 적용
                shape_temp = inputs.shape
                inputs = inputs.reshape(shape_temp[0]*shape_temp[1], shape_temp[2], shape_temp[3], shape_temp[4])
                for _ in range(num_pooling_layers):
                    inputs = pool(inputs)
                inputs = inputs.reshape(shape_temp[0], shape_temp[1], shape_temp[2], shape_temp[3]//initial_pooling, shape_temp[4]//initial_pooling)
            ## initial pooling #######################################################################
            ## temporal filtering ####################################################################
            shape_temp = inputs.shape
            if (temporal_filter > 1):
                slice_bucket = []
                for t_temp in range(TIME):
                    start = t_temp * temporal_filter
                    end = start + temporal_filter
                    slice_concat = torch.movedim(inputs[start:end], 0, -2).reshape(shape_temp[1],shape_temp[2],shape_temp[3],-1)
                    
                    if temporal_filter_accumulation == True:
                        if t_temp == 0:
                            slice_bucket.append(slice_concat)
                        else:
                            slice_bucket.append(slice_concat+slice_bucket[t_temp-1])
                    else:
                        slice_bucket.append(slice_concat)

                inputs = torch.stack(slice_bucket, dim=0)
                if temporal_filter_accumulation == True and dvs_clipping > 0:
                    inputs = (inputs != 0.0).float()
            ## temporal filtering ####################################################################
            ####################################################################################################################### 
                

            # # dvs 데이터 시각화 코드 (확인 필요할 시 써라)
            # ##############################################################################################
            # dvs_visualization(inputs, labels, TIME, BATCH, my_seed)
            # #####################################################################################################

            ## to (device) #######################################
            inputs = inputs.to(device)
            labels = labels.to(device)
            ###########################################################

            # ## gradient 초기화 #######################################
            # optimizer.zero_grad()
            # ###########################################################
                            
            if merge_polarities == True:
                inputs = inputs[:,:,0:1,:,:]

            if single_step == False:
                # net에 넣어줄때는 batch가 젤 앞 차원으로 와야함. # dataparallel때매##############################
                # inputs: [Time, Batch, Channel, Height, Width]   
                inputs = inputs.permute(1, 0, 2, 3, 4) # net에 넣어줄때는 batch가 젤 앞 차원으로 와야함. # dataparallel때매
                # inputs: [Batch, Time, Channel, Height, Width] 
                #################################################################################################
            else:
                labels = labels.repeat(TIME, 1)
                ## first input도 ottt trace 적용하기 위한 코드 (validation 시에는 필요X) ##########################
                if trace_on == True and OTTT_input_trace_on == True:
                    spike = inputs
                    trace = torch.full_like(spike, fill_value = 0.0, dtype = torch.float, requires_grad=False)
                    inputs = []
                    for t in range(TIME):
                        trace[t] = trace[t-1]*synapse_trace_const2 + spike[t]*synapse_trace_const1
                        inputs += [[spike[t], trace[t]]]
                ##################################################################################################


            if single_step == False:
                ### input --> net --> output #####################################################
                outputs = net(inputs)
                ##################################################################################
                ## loss, backward ##########################################
                iter_loss = criterion(outputs, labels)
                iter_loss.backward()
                ############################################################
                ## weight 업데이트!! ##################################
                optimizer.step()
                ################################################################
            else:
                outputs_all = []
                iter_loss = 0.0
                for t in range(TIME):
                    optimizer.step() # full step time update
                    optimizer.zero_grad()
                    ### input[t] --> net --> output_one_time #########################################
                    outputs_one_time = net(inputs[t])
                    ##################################################################################
                    one_time_loss = criterion(outputs_one_time, labels[t].contiguous())
                    one_time_loss.backward() # one_time backward
                    iter_loss += one_time_loss.data
                    outputs_all.append(outputs_one_time.detach())

                    total_backward_count = total_backward_count + 1
                    outputs_one_time_argmax = (outputs_one_time.detach()).argmax(dim=1)
                    real_backward_count = real_backward_count + (outputs_one_time_argmax != labels[t]).sum().item()

                outputs_all = torch.stack(outputs_all, dim=1)
                outputs = outputs_all.mean(1) # ottt꺼 쓸때
                labels = labels[0]
                iter_loss /= TIME

            tr_epoch_loss_temp += iter_loss.data/len(train_loader)

            ## net 그림 출력해보기 #################################################################
            # print('시각화')
            # make_dot(outputs, params=dict(list(net.named_parameters()))).render("net_torchviz", format="png")
            # return 0
            ##################################################################################

            #### batch 어긋남 방지 ###############################################
            assert real_batch == outputs.size(0), f'batch size is not same. real_batch: {real_batch}, outputs.size(0): {outputs.size(0)}'
            #######################################################################
            

            ####### training accruacy save for print ###############################
            _, predicted = torch.max(outputs.data, 1)
            total = real_batch
            correct = (predicted == labels).sum().item()
            iter_acc = correct / total
            tr_total += total
            tr_correct += correct
            iter_acc_string = f'epoch-{epoch:<3} iter_acc:{100 * iter_acc:7.2f}%, lr={[f"{lr:9.7f}" for lr in (param_group["lr"] for param_group in optimizer.param_groups)]}'
            iter_acc_string2 = f'epoch-{epoch:<3} lr={[f"{lr:9.7f}" for lr in (param_group["lr"] for param_group in optimizer.param_groups)]}'
            ################################################################
            

            ##### validation ##################################################################################################################################
            if i == len(train_loader)-1 :
                iter_of_val = True

                tr_acc = tr_correct/tr_total
                tr_correct = 0
                tr_total = 0

                val_loss = 0
                correct_val = 0
                total_val = 0

                with torch.no_grad():
                    net.eval() # eval 모드로 바꿔줘야함 
                    for data_val in test_loader:
                        ## data_val loading & semi-pre-processing ##########################################################
                        if len(data_val) == 2:
                            inputs_val, labels_val = data_val
                        elif len(data_val) == 3:
                            inputs_val, labels_val, x_len = data_val
                        else:
                            assert False, 'data_val length is not 2 or 3'

                        if (which_data == 'DVS_CIFAR10' or which_data == 'DVS_GESTURE' or which_data == 'DVS_GESTURE_TONIC' or which_data == 'DVS_CIFAR10_2' or which_data == 'NMNIST' or which_data == 'NMNIST_TONIC' or which_data == 'N_CALTECH101' or which_data == 'n_tidigits' or which_data == 'heidelberg'):
                            inputs_val = inputs_val.permute(1, 0, 2, 3, 4)
                        elif rate_coding == True :
                            inputs_val = spikegen.rate(inputs_val, num_steps=TIME)
                        else :
                            inputs_val = inputs_val.repeat(TIME, 1, 1, 1, 1)
                        # inputs_val: [Time, Batch, Channel, Height, Width]  
                        ###################################################################################################

                        
                        ## initial pooling #######################################################################
                        if (initial_pooling > 1):
                            pool = nn.MaxPool2d(kernel_size=2)
                            num_pooling_layers = int(math.log2(initial_pooling))
                            # Time, Batch, Channel 차원은 그대로 두고, Height, Width 차원에 대해서만 pooling 적용
                            shape_temp = inputs_val.shape
                            inputs_val = inputs_val.reshape(shape_temp[0]*shape_temp[1], shape_temp[2], shape_temp[3], shape_temp[4])
                            for _ in range(num_pooling_layers):
                                inputs_val = pool(inputs_val)
                            inputs_val = inputs_val.reshape(shape_temp[0], shape_temp[1], shape_temp[2], shape_temp[3]//initial_pooling, shape_temp[4]//initial_pooling)
                        ## initial pooling #######################################################################

                        ## temporal filtering ####################################################################
                        shape_temp = inputs_val.shape
                        if (temporal_filter > 1):
                            slice_bucket = []
                            for t_temp in range(TIME):
                                start = t_temp * temporal_filter
                                end = start + temporal_filter
                                slice_concat = torch.movedim(inputs_val[start:end], 0, -2).reshape(shape_temp[1],shape_temp[2],shape_temp[3],-1)
                                
                                if temporal_filter_accumulation == True:
                                    if t_temp == 0:
                                        slice_bucket.append(slice_concat)
                                    else:
                                        slice_bucket.append(slice_concat+slice_bucket[t_temp-1])
                                else:
                                    slice_bucket.append(slice_concat)

                            inputs_val = torch.stack(slice_bucket, dim=0)
                            if temporal_filter_accumulation == True and dvs_clipping > 0:
                                inputs = (inputs != 0.0).float()
                        ## temporal filtering ####################################################################
                            
                        inputs_val = inputs_val.to(device)
                        labels_val = labels_val.to(device)
                        real_batch = labels_val.size(0)
                        
                        if merge_polarities == True:
                            inputs_val = inputs_val[:,:,0:1,:,:]

                        ## network 연산 시작 ############################################################################################################
                        if single_step == False:
                            outputs = net(inputs_val.permute(1, 0, 2, 3, 4)) #inputs_val: [Batch, Time, Channel, Height, Width]  
                            val_loss += criterion(outputs, labels_val)/len(test_loader)
                        else:
                            outputs_all = []
                            for t in range(TIME):
                                outputs = net(inputs_val[t])
                                val_loss_temp = criterion(outputs, labels_val)
                                outputs_all.append(outputs.detach())
                                val_loss += (val_loss_temp.data/TIME)/len(test_loader)
                            outputs_all = torch.stack(outputs_all, dim=1)
                            outputs = outputs_all.mean(1)
                        #################################################################################################################################

                        _, predicted = torch.max(outputs.data, 1)
                        total_val += real_batch
                        assert real_batch == outputs.size(0), f'batch size is not same. real_batch: {real_batch}, outputs.size(0): {outputs.size(0)}'
                        correct_val += (predicted == labels_val).sum().item()

                    val_acc_now = correct_val / total_val

                if val_acc_best < val_acc_now:
                    val_acc_best = val_acc_now
                    # wandb 키면 state_dict아닌거는 저장 안됨
                    # network save
                    torch.save(net.state_dict(), f"net_save/save_now_net_weights_{unique_name}.pth")

                if tr_acc_best < tr_acc:
                    tr_acc_best = tr_acc

                tr_epoch_loss = tr_epoch_loss_temp
                tr_epoch_loss_temp = 0

            ####################################################################################################################################################
            
            ## progress bar update ############################################################################################################
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            if iter_of_val == False:
                # iterator.set_description(f"{iter_acc_string}, iter_loss:{iter_loss:10.6f}") 
                pass 
            else:
                # iterator.set_description(f"{iter_acc_string2}, tr/val_loss:{tr_epoch_loss:10.6f}/{val_loss:10.6f}, tr:{100 * tr_acc:7.2f}%, tr_best:{100 * tr_acc_best:7.2f}%, val:{100 * val_acc_now:7.2f}%, val_best:{100 * val_acc_best:7.2f}%")  
                print(f"{iter_acc_string2}, tr/val_loss:{tr_epoch_loss:10.6f}/{val_loss:10.6f}, val:{100 * val_acc_now:7.2f}%, val_best:{100 * val_acc_best:7.2f}%, tr:{100 * tr_acc:7.2f}%, tr_best:{100 * tr_acc_best:7.2f}%, epoch time: {epoch_time:.2f} seconds, {epoch_time/60:.2f} minutes")
                iter_of_val = False
            ####################################################################################################################################
            
            ## wandb logging ############################################################################################################
            if i == len(train_loader)-1 :
                wandb.log({"iter_acc": iter_acc})
                wandb.log({"tr_acc": tr_acc})
                wandb.log({"val_acc_now": val_acc_now})
                wandb.log({"val_acc_best": val_acc_best})
                wandb.log({"summary_val_acc": val_acc_now})
                wandb.log({"epoch": epoch})
                wandb.log({"val_loss": val_loss}) 
                wandb.log({"tr_epoch_loss": tr_epoch_loss}) 
                # wandb.log({"max_val_scale_exp_8bit_1w": max_val_scale_exp_8bit_box[0]}) 
                # wandb.log({"max_val_scale_exp_8bit_1b": max_val_scale_exp_8bit_box[1]})
                # wandb.log({"max_val_scale_exp_8bit_2w": max_val_scale_exp_8bit_box[2]}) 
                # wandb.log({"max_val_scale_exp_8bit_2b": max_val_scale_exp_8bit_box[3]}) 
                # wandb.log({"max_val_scale_exp_8bit_3w": max_val_scale_exp_8bit_box[4]}) 
                # wandb.log({"max_val_scale_exp_8bit_3b": max_val_scale_exp_8bit_box[5]})

                # wandb.log({"perc_999_scale_exp_8bit_1w": perc_999_scale_exp_8bit_box[0]}) 
                # wandb.log({"perc_999_scale_exp_8bit_1b": perc_999_scale_exp_8bit_box[1]})
                # wandb.log({"perc_999_scale_exp_8bit_2w": perc_999_scale_exp_8bit_box[2]}) 
                # wandb.log({"perc_999_scale_exp_8bit_2b": perc_999_scale_exp_8bit_box[3]}) 
                # wandb.log({"perc_999_scale_exp_8bit_3w": perc_999_scale_exp_8bit_box[4]}) 
                # wandb.log({"perc_999_scale_exp_8bit_3b": perc_999_scale_exp_8bit_box[5]}) 

            ####################################################################################################################################
            
        ###### ITERATION END ##########################################################################################################

        ## scheduler update #############################################################################
        if (scheduler_name != 'no'):
            if (scheduler_name == 'ReduceLROnPlateau'):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        #################################################################################################
        
    #======== EPOCH END ==========================================================================================
























































































# unique_name = 'main' ## 이거 설정하면 새로운 경로에 모두 save
# wandb.init(project= f'my_snn {unique_name}',save_code=False, dir='/data2/bh_wandb', tags=["common"])
# ## wandb 과거 하이퍼파라미터 가져와서 붙여넣기 (devices unique_name은 니가 할당해라)#################################
# param = {'devices': '3', 'single_step': True, 'unique_name': 'main', 'my_seed': 42, 'TIME': 10, 'BATCH': 16, 'IMAGE_SIZE': 128, 'which_data': 'DVS_GESTURE_TONIC', 'data_path': '/data2', 'rate_coding': False, 'lif_layer_v_init': 0, 'lif_layer_v_decay': 0.25, 'lif_layer_v_threshold': 0.75, 'lif_layer_v_reset': 0, 'lif_layer_sg_width': 4, 'synapse_conv_kernel_size': 3, 'synapse_conv_stride': 1, 'synapse_conv_padding': 1, 'synapse_trace_const1': 1, 'synapse_trace_const2': 0, 'pre_trained': False, 'convTrue_fcFalse': False, 'cfg': [200, 200], 'net_print': True, 'pre_trained_path': 'net_save/save_now_net_weights_{unique_name}.pth', 'learning_rate': 0.001, 'epoch_num': 100, 'tdBN_on': False, 'BN_on': False, 'surrogate': 'hard_sigmoid', 'BPTT_on': False, 'optimizer_what': 'SGD', 'scheduler_name': 'no', 'ddp_on': False, 'dvs_clipping': 2, 'dvs_duration': 25000, 'DFA_on': True, 'trace_on': True, 'OTTT_input_trace_on': False, 'exclude_class': True, 'merge_polarities': False, 'denoise_on': True, 'extra_train_dataset': 0, 'num_workers': 2, 'chaching_on': True, 'pin_memory': True, 'UDA_on': False, 'alpha_uda': 1, 'bias': True, 'last_lif': False, 'temporal_filter': 5, 'initial_pooling': 8}
# my_snn_system(devices = '0',single_step = param['single_step'],unique_name = unique_name,my_seed = param['my_seed'],TIME = param['TIME'],BATCH = param['BATCH'],IMAGE_SIZE = param['IMAGE_SIZE'],which_data = param['which_data'],data_path = param['data_path'],rate_coding = param['rate_coding'],lif_layer_v_init = param['lif_layer_v_init'],lif_layer_v_decay = param['lif_layer_v_decay'],lif_layer_v_threshold = param['lif_layer_v_threshold'],lif_layer_v_reset = param['lif_layer_v_reset'],lif_layer_sg_width = param['lif_layer_sg_width'],synapse_conv_kernel_size = param['synapse_conv_kernel_size'],synapse_conv_stride = param['synapse_conv_stride'],synapse_conv_padding = param['synapse_conv_padding'],synapse_trace_const1 = param['synapse_trace_const1'],synapse_trace_const2 = param['synapse_trace_const2'],pre_trained = param['pre_trained'],convTrue_fcFalse = param['convTrue_fcFalse'],cfg = param['cfg'],net_print = param['net_print'],pre_trained_path = param['pre_trained_path'],learning_rate = param['learning_rate'],epoch_num = param['epoch_num'],tdBN_on = param['tdBN_on'],BN_on = param['BN_on'],surrogate = param['surrogate'],BPTT_on = param['BPTT_on'],optimizer_what = param['optimizer_what'],scheduler_name = param['scheduler_name'],ddp_on = param['ddp_on'],dvs_clipping = param['dvs_clipping'],dvs_duration = param['dvs_duration'],DFA_on = param['DFA_on'],trace_on = param['trace_on'],OTTT_input_trace_on = param['OTTT_input_trace_on'],exclude_class = param['exclude_class'],merge_polarities = param['merge_polarities'],denoise_on = param['denoise_on'],extra_train_dataset = param['extra_train_dataset'],num_workers = param['num_workers'],chaching_on = param['chaching_on'],pin_memory = param['pin_memory'],UDA_on = param['UDA_on'],alpha_uda = param['alpha_uda'],bias = param['bias'],last_lif = param['last_lif'],temporal_filter = param['temporal_filter'],initial_pooling = param['initial_pooling'],temporal_filter_accumulation= param['temporal_filter_accumulation'])
# #############################################################################























































































### my_snn control board (Gesture) ########################
decay = 0.5 # 0.0 # 0.875 0.25 0.125 0.75 0.5
# nda 0.25 # ottt 0.5

unique_name = 'main'
run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + f"{datetime.datetime.now().microsecond // 1000:03d}"


wandb.init(project= f'my_snn {unique_name}',save_code=False, dir='/data2/bh_wandb', tags=["common"])

my_snn_system(  devices = "5",
                single_step = True, # True # False # DFA_on이랑 같이 가라
                unique_name = run_name,
                my_seed = 7777,
                TIME = 4, # dvscifar 10 # ottt 6 or 10 # nda 10  # 제작하는 dvs에서 TIME넘거나 적으면 자르거나 PADDING함
                BATCH = 1, # batch norm 할거면 2이상으로 해야함   # nda 256   #  ottt 128
                IMAGE_SIZE = 17, # dvscifar 48 # MNIST 28 # CIFAR10 32 # PMNIST 28 #NMNIST 34 # GESTURE 128
                # dvsgesture 128, dvs_cifar2 128, nmnist 34, n_caltech101 180,240, n_tidigits 64, heidelberg 700, 

                # DVS_CIFAR10 할거면 time 10으로 해라
                which_data = 'NMNIST_TONIC',
# 'CIFAR100' 'CIFAR10' 'MNIST' 'FASHION_MNIST' 'DVS_CIFAR10' 'PMNIST'아직
# 'DVS_GESTURE', 'DVS_GESTURE_TONIC','DVS_CIFAR10_2','NMNIST','NMNIST_TONIC','CIFAR10','N_CALTECH101','n_tidigits','heidelberg'
                # CLASS_NUM = 10,
                data_path = '/data2', # YOU NEED TO CHANGE THIS
                rate_coding = False, # True # False

                lif_layer_v_init = 0.0,
                lif_layer_v_decay = decay,
                lif_layer_v_threshold = 0.5,   #nda 0.5  #ottt 1.0
                lif_layer_v_reset = 10000.0, # 10000이상은 hardreset (내 LIF쓰기는 함 ㅇㅇ)
                lif_layer_sg_width = 6.0, # 2.570969004857107 # sigmoid류에서는 alpha값 4.0, rectangle류에서는 width값 0.5

                # synapse_conv_in_channels = IMAGE_PIXEL_CHANNEL,
                synapse_conv_kernel_size = 3,
                synapse_conv_stride = 1,
                synapse_conv_padding = 1,

                synapse_trace_const1 = 1, # 현재 trace구할 때 현재 spike에 곱해지는 상수. 걍 1로 두셈.
                synapse_trace_const2 = decay, # 현재 trace구할 때 직전 trace에 곱해지는 상수. lif_layer_v_decay와 같게 할 것을 추천

                # synapse_fc_out_features = CLASS_NUM,

                pre_trained = False, # True # False
                convTrue_fcFalse = False, # True # False

                # 'P' for average pooling, 'D' for (1,1) aver pooling, 'M' for maxpooling, 'L' for linear classifier, [  ] for residual block
                # conv에서 10000 이상은 depth-wise separable (BPTT만 지원), 20000이상은 depth-wise (BPTT만 지원)
                # cfg = ['M', 'M', 32, 'P', 32, 'P', 32, 'P'], 
                # cfg = ['M', 'M', 64, 'P', 64, 'P', 64, 'P'], 
                # cfg = ['M', 'M', 64, 'M', 96, 'M', 128, 'M'], 
                cfg = [200, 200], 
                # cfg = ['M', 'M', 64, 'M', 96], 
                # cfg = ['M', 'M', 64, 'M', 96, 'L', 512, 512], 
                # cfg = ['M', 'M', 64], 
                # cfg = [64, 124, 64, 124],
                # cfg = ['M','M',512], 
                # cfg = [512], 
                # cfg = ['M', 'M', 64, 128, 'P', 128, 'P'], 
                # cfg = ['M','M',512],
                # cfg = ['M',200],
                # cfg = [200,200],
                # cfg = ['M','M',200,200],
                # cfg = ([200],[200],[200],[2]), # (feature extractor, classifier, domain adapter, # of domain)
                # cfg = (['M','M',200],[200],[200],[2]), # (feature extractor, classifier, domain adapter, # of domain)
                # cfg = ['M',200,200],
                # cfg = ['M','M',1024,512,256,128,64],
                # cfg = [200,200],
                # cfg = [12], #fc
                # cfg = [12, 'M', 48, 'M', 12], 
                # cfg = [64,[64,64],64], # 끝에 linear classifier 하나 자동으로 붙습니다
                # cfg = [64, 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512, 'D'], #ottt
                # cfg = [64, 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512], 
                # cfg = [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512], 
                # cfg = [64, 'P', 128, 'P', 256, 256, 'P', 512, 512, 512, 512, 'D'], # nda
                # cfg = [64, 'P', 128, 'P', 256, 256, 'P', 512, 512, 512, 512], # nda 128pixel
                # cfg = [64, 'P', 128, 'P', 256, 256, 'P', 512, 512, 512, 512, 'L', 4096, 4096],
                # cfg = [20001,10001], # depthwise, separable
                # cfg = [64,20064,10001], # vanilla conv, depthwise, separable
                # cfg = [8, 'P', 8, 'P', 8, 'P', 8,'P', 8, 'P'],
                # cfg = [],        
                
                net_print = True, # True # False # True로 하길 추천
                
                pre_trained_path = f"net_save/save_now_net_weights_20250906_001313_333.pth",
                # learning_rate = 0.001, #0.1 bptt, #0.01 ottt, # default 0.001  # ottt 0.1 # nda 0.001 # 0.00936191669529645
                learning_rate = 1/256, #0.1 bptt, #0.01 ottt, # default 0.001  # ottt 0.1 # nda 0.001 # 0.00936191669529645
                epoch_num = 300,
                tdBN_on = False,  # True # False
                BN_on = False,  # True # False
                
                surrogate = 'hard_sigmoid', # 'sigmoid' 'rectangle' 'rough_rectangle' 'hard_sigmoid'
                
                BPTT_on = False,  # True # False # True이면 BPTT, False이면 OTTT  # depthwise, separable은 BPTT만 가능
                
                optimizer_what = 'SGD', # 'SGD' 'Adam', 'RMSprop'
                scheduler_name = 'no', # 'no' 'StepLR' 'ExponentialLR' 'ReduceLROnPlateau' 'CosineAnnealingLR' 'OneCycleLR'
                
                ddp_on = False, # DECREPATED # fALSE

                dvs_clipping = 1, #일반적으로 1 또는 2 # 100ms때는 5 # 숫자만큼 크면 spike 아니면 걍 0
                # gesture, cifar-dvs2, nmnist, ncaltech101
                # gesture: 100_000c1-5, 25_000c5, 10_000c5, 1_000c5, 1_000_000c5

                dvs_duration = 5_000, # 0 아니면 time sampling # dvs number sampling OR time sampling # gesture, cifar-dvs2, nmnist, ncaltech101
                # 있는 데이터들 #gesture 100_000 25_000 10_000 1_000 1_000_000 #nmnist 10000 #nmnist_tonic 10_000 25_000
                # 한 숫자가 1us인듯 (spikingjelly코드에서)
                # 한 장에 50 timestep만 생산함. 싫으면 my_snn/trying/spikingjelly_dvsgesture의__init__.py 를 참고해봐
                # nmnist 5_000us, gesture는 100_000us, 25_000us

                DFA_on = True, # True # False # single_step이랑 같이 켜야 됨.

                trace_on = False,   # True # False
                OTTT_input_trace_on = False, # True # False # 맨 처음 input에 trace 적용 # trace_on False면 의미없음.

                exclude_class = True, # True # False # gesture에서 10번째 클래스 제외

                merge_polarities = False, # True # False # tonic dvs dataset 에서 polarities 합치기
                denoise_on = False, # True # False # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

                extra_train_dataset = 0, 

                num_workers = 2, # local wsl에서는 2가 맞고, 서버에서는 4가 좋더라.
                chaching_on = True, # True # False # only for certain datasets (gesture_tonic, nmnist_tonic)
                pin_memory = True, # True # False 

                UDA_on = False,  # DECREPATED # uda
                alpha_uda = 1.0, # DECREPATED # uda

                bias = False, # True # False 

                last_lif = False, # True # False 

                temporal_filter = 1, 
                initial_pooling = 1,

                temporal_filter_accumulation = False, # True # False 

                quantize_bit_list=[8,8,8],
                scale_exp=[[-9,-9],[-9,-9],[-8,-8]], 
# 1w -11~-9
# 1b -11~ -7
# 2w -10~-8
# 2b -10~-8
# 3w -10
# 3b -10
                ) 

# num_workers = 4 * num_GPU (or 8, 16, 2 * num_GPU)
# entry * batch_size * num_worker = num_GPU * GPU_throughtput
# num_workers = batch_size / num_GPU
# num_workers = batch_size / num_CPU

# sigmoid와 BN이 있어야 잘된다.
# average pooling  
# 이 낫다. 

# nda에서는 decay = 0.25, threshold = 0.5, width =1, surrogate = rectangle, batch = 256, tdBN = True
## OTTT 에서는 decay = 0.5, threshold = 1.0, surrogate = sigmoid, batch = 128, BN = True

wandb.finish()


# 46까지했었긴했음
# och-42  lr=['0.0039062'], tr/val_loss:  0.791181/  0.859875, val:  94.82%, val_best:  96.05%, tr:  99.85%, tr_best:  99.88%, epoch time: 2653.91 seconds, 44.23 minutes
# epoch-43  lr=['0.0039062'], tr/val_loss:  0.797550/  0.848301, val:  95.38%, val_best:  96.05%, tr:  99.87%, tr_best:  99.88%, epoch time: 2648.23 seconds, 44.14 minutes
# epoch-44  lr=['0.0039062'], tr/val_loss:  0.801503/  0.850083, val:  95.41%, val_best:  96.05%, tr:  99.87%, tr_best:  99.88%, epoch time: 2644.07 seconds, 44.07 minutes
# epoch-45  lr=['0.0039062'], tr/val_loss:  0.802839/  0.866149, val:  94.36%, val_best:  96.05%, tr:  99.87%, tr_best:  99.88%, epoch time: 2633.83 seconds, 43.90 minutes
# Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...







# total_backward_count 94560000 real_backward_count 4378277   4.630%
# epoch-197 lr=['0.0039062'], tr/val_loss:  0.711656/  0.795913, val:  96.51%, val_best:  96.58%, tr:  99.91%, tr_best:  99.94%, epoch time: 2973.94 seconds, 49.57 minutes
# total_backward_count 95040000 real_backward_count 4398112   4.628%
# epoch-198 lr=['0.0039062'], tr/val_loss:  0.713885/  0.791970, val:  95.18%, val_best:  96.58%, tr:  99.91%, tr_best:  99.94%, epoch time: 2974.66 seconds, 49.58 minutes
# total_backward_count 95520000 real_backward_count 4417944   4.625%
# epoch-199 lr=['0.0039062'], tr/val_loss:  0.717439/  0.813721, val:  94.56%, val_best:  96.58%, tr:  99.90%, tr_best:  99.94%, epoch time: 2974.68 seconds, 49.58 minutes
# total_backward_count 96000000 real_backward_count 4437843   4.623%
# ^CTraceback (most recent call last):
