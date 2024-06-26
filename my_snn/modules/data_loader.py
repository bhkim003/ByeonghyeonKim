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

from apex.parallel import DistributedDataParallel as DDP



from modules.data_loader import *
from modules.network import *
from modules.neuron import *
from modules.synapse import *

def data_loader(which_data, data_path, rate_coding, BATCH, IMAGE_SIZE, ddp_on):

    if (which_data == 'MNIST'):

        if rate_coding :
            transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0,), (1,))])
        else : 
            transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,),(0.5))])

        trainset = torchvision.datasets.MNIST(root=data_path,
                                            train=True,
                                            download=True,
                                            transform=transform)


        testset = torchvision.datasets.MNIST(root=data_path,
                                            train=False,
                                            download=True,
                                            transform=transform)

        if (ddp_on == True):
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)

            test_sampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False)

            train_loader = DataLoader(trainset,
                                    batch_size =BATCH,
                                    num_workers =2,
                                    sampler=train_sampler)
            test_loader = DataLoader(testset,
                                    batch_size =BATCH,
                                    num_workers =2,
                                    sampler=test_sampler)
        else: 
            train_loader = DataLoader(trainset,
                                    batch_size =BATCH,
                                    shuffle = True,
                                    num_workers =2)
            test_loader = DataLoader(testset,
                                    batch_size =BATCH,
                                    shuffle = False,
                                    num_workers =2)


    if (which_data == 'CIFAR10'):

        if rate_coding :
            transform_train = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor()])

            transform_test = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                                transforms.ToTensor()])
        
        else :
            transform_train = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
                                            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

            transform_test = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),])
                                            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))


        trainset = torchvision.datasets.CIFAR10(root=data_path,
                                            train=True,
                                            download=True,
                                            transform=transform_train)


        testset = torchvision.datasets.CIFAR10(root=data_path,
                                            train=False,
                                            download=True,
                                            transform=transform_test)
        
        
        if (ddp_on == True):
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)

            test_sampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False)

            train_loader = DataLoader(trainset,
                                    batch_size =BATCH,
                                    num_workers =2,
                                    sampler=train_sampler)
            test_loader = DataLoader(testset,
                                    batch_size =BATCH,
                                    num_workers =2,
                                    sampler=test_sampler)
        else: 
            train_loader = DataLoader(trainset,
                                    batch_size =BATCH,
                                    shuffle = True,
                                    num_workers =2)
            test_loader = DataLoader(testset,
                                    batch_size =BATCH,
                                    shuffle = False,
                                    num_workers =2)


        '''
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck') 
        '''

    if (which_data == 'FASHION_MNIST'):

        if rate_coding :
            transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                        transforms.ToTensor()])
        else : 
            transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,),(0.5))])

        trainset = torchvision.datasets.FashionMNIST(root=data_path,
                                            train=True,
                                            download=True,
                                            transform=transform)


        testset = torchvision.datasets.FashionMNIST(root=data_path,
                                            train=False,
                                            download=True,
                                            transform=transform)

        if (ddp_on == True):
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)

            test_sampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False)

            train_loader = DataLoader(trainset,
                                    batch_size =BATCH,
                                    num_workers =2,
                                    sampler=train_sampler)
            test_loader = DataLoader(testset,
                                    batch_size =BATCH,
                                    num_workers =2,
                                    sampler=test_sampler)
        else: 
            train_loader = DataLoader(trainset,
                                    batch_size =BATCH,
                                    shuffle = True,
                                    num_workers =2)
            test_loader = DataLoader(testset,
                                    batch_size =BATCH,
                                    shuffle = False,
                                    num_workers =2)

        

    data_iter = IMAGE_PIXEL_CHANNEL = iter(train_loader)
    images, labels = data_iter.next()

    # 채널 수와 클래스 개수를 확인합니다.
    synapse_conv_in_channels = images.shape[1]

    return train_loader, test_loader, synapse_conv_in_channels


