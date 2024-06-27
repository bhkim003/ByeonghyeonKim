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

import random


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
        synapse_conv_in_channels = 1
        


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
        synapse_conv_in_channels = 3
        


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
        synapse_conv_in_channels = 1



    # reference: H Zheng, Y Wu, L Deng, Y Hu, G Li. "Tdbn: Going Deeper with Directly-Trained Larger Spiking Neural Networks." Proceedings of the AAAI conference on artificial intelligence  (2021). Print.
    if (which_data == 'DVS-CIFAR10'):
        data_path = data_path + '/cifar-dvs'
        train_path = data_path + '/train'
        val_path = data_path + '/test'
        train_dataset = DVSCifar10(path=train_path, transform=True, img_size = IMAGE_SIZE)
        val_dataset = DVSCifar10(path=val_path, transform=False, img_size = IMAGE_SIZE)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH, shuffle=True,
                                                    num_workers=2, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH, shuffle=False,
                                                    num_workers=2, pin_memory=True)
        synapse_conv_in_channels = 2

    # data_iter = iter(train_loader)
    # images, labels = data_iter.next()


    return train_loader, test_loader, synapse_conv_in_channels



# reference: H Zheng, Y Wu, L Deng, Y Hu, G Li. "Tdbn: Going Deeper with Directly-Trained Larger Spiking Neural Networks." Proceedings of the AAAI conference on artificial intelligence  (2021). Print.
class DVSCifar10(Dataset):
    def __init__(self, path, transform=None, img_size = 48):
        self.path = path
        self.transform = transform
        self.img_size = img_size

        # 일단 기본적인 resize는 무조건 해주고
        # rotate하고 shear(x방향)하고 roll은 무작위로함.
        self.resize = transforms.Resize(size=(self.img_size, self.img_size), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        self.rotate = transforms.RandomRotation(degrees=30)
        self.shearx = transforms.RandomAffine(degrees=0, shear=(-30, 30))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.path + '/{}.pt'.format(index))
        data = self.resize(data.permute([3, 0, 1, 2]))

        if self.transform == True:
            choices = ['roll', 'rotate', 'shear']
            aug = np.random.choice(choices)
            if aug == 'roll':
                off1 = random.randint(-5, 5)
                off2 = random.randint(-5, 5)
                data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))
            if aug == 'rotate':
                data = self.rotate(data)
            if aug == 'shear':
                data = self.shearx(data)

        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(os.listdir(self.path))