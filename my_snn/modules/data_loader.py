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

from torchvision import datasets, transforms
from sklearn.utils import shuffle

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
        CLASS_NUM = 10
        


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
        CLASS_NUM = 10
        


        '''
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck') 
        '''



    # CIFAR100의 설정
    elif which_data == 'CIFAR100':
        if rate_coding:
            transform_train = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])

            transform_test = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor()
            ])
        else:
            transform_train = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])

            transform_test = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])

        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)

        if ddp_on:
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
            test_sampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False)

            train_loader = DataLoader(trainset, batch_size=BATCH, num_workers=2, sampler=train_sampler)
            test_loader = DataLoader(testset, batch_size=BATCH, num_workers=2, sampler=test_sampler)
        else:
            train_loader = DataLoader(trainset, batch_size=BATCH, shuffle=True, num_workers=2)
            test_loader = DataLoader(testset, batch_size=BATCH, shuffle=False, num_workers=2)

        synapse_conv_in_channels = 3
        CLASS_NUM = 100

        '''
        classes = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
                'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
                'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
                'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin',
                'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo',
                'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
                'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange',
                'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain',
                'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
                'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail',
                'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
                'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm')
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
        CLASS_NUM = 10




    # reference: H Zheng, Y Wu, L Deng, Y Hu, G Li. "Tdbn: Going Deeper with Directly-Trained Larger Spiking Neural Networks." Proceedings of the AAAI conference on artificial intelligence  (2021). Print.
    if (which_data == 'DVS_CIFAR10'):
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
        CLASS_NUM = 10



    # reference: Xiao, Mingqing, et al. "Hebbian learning based orthogonal projection for continual learning of spiking neural networks." arXiv preprint arXiv:2402.11984 (2024).
    if (which_data == 'PMNIST'):
        data, taskcla, size = pmnist_get(data_dir=data_path, seed=0, fixed_order=False)


        train_loader = data  # 다른 dataset과 다르니 외부에서 리빌딩해라
        test_loader = taskcla # 다른 dataset과 다르니 외부에서 리빌딩해라
        synapse_conv_in_channels = 1
        CLASS_NUM = 10




    # data_iter = iter(train_loader)
    # images, labels = data_iter.next()
    return train_loader, test_loader, synapse_conv_in_channels, CLASS_NUM



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
        self.cutout = Cutout(length=16)  # Cutout length (patch size) set to 16 pixels


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.path + '/{}.pt'.format(index))
        # print('원래사이즈', data.size())
        data = self.resize(data.permute([3, 0, 1, 2]))

        if self.transform == True:
            choices = ['roll', 'rotate', 'shear', 'cutout']
            aug = np.random.choice(choices)
            if aug == 'roll':
                off1 = random.randint(-5, 5)
                off2 = random.randint(-5, 5)
                data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))
            elif aug == 'rotate':
                data = self.rotate(data)
            elif aug == 'shear':
                data = self.shearx(data)
            elif aug == 'cutout':
                data = self.cutout(data)

        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(os.listdir(self.path))
    
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h = img.size(2)
        w = img.size(3)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img


# reference: Xiao, Mingqing, et al. "Hebbian learning based orthogonal projection for continual learning of spiking neural networks." arXiv preprint arXiv:2402.11984 (2024).
def pmnist_get(data_dir='/data2', seed=0, fixed_order=False):
    data = {}
    taskcla = []
    size = [1, 28, 28]

    mnist_dir = data_dir
    pmnist_dir = os.path.join(mnist_dir, 'binary_pmnist')

    nperm = 10  # 10 tasks
    seeds = np.array(list(range(nperm)), dtype=int)
    if not fixed_order:
        seeds = shuffle(seeds, random_state=seed)

    if not os.path.isdir(pmnist_dir):
        os.makedirs(pmnist_dir)
        # Pre-load
        # MNIST
        mean = (0.1307,)
        std = (0.3081,)
        dat = {}
        dat['train'] = datasets.MNIST(mnist_dir, train=True, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        dat['test'] = datasets.MNIST(mnist_dir, train=False, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        for i, r in enumerate(seeds):
            print(i, end=',')
            sys.stdout.flush()
            data[i] = {}
            data[i]['name'] = 'pmnist-{:d}'.format(i)
            data[i]['ncla'] = 10
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[i][s] = {'x': [], 'y': []}
                for image, target in loader:
                    aux = image.view(-1).numpy()
                    aux = shuffle(aux, random_state=r * 100 + i)
                    image = torch.FloatTensor(aux).view(size)
                    data[i][s]['x'].append(image)
                    data[i][s]['y'].append(target.numpy()[0])

            # "Unify" and save
            for s in ['train', 'test']: # test, train 각각 x,y를 저장
                data[i][s]['x'] = torch.stack(data[i][s]['x']).view(-1, size[0], size[1], size[2])
                data[i][s]['y'] = torch.LongTensor(np.array(data[i][s]['y'], dtype=int)).view(-1)
                torch.save(data[i][s]['x'],os.path.join(os.path.expanduser(pmnist_dir), 'data' + str(r) + s + 'x.bin'))
                torch.save(data[i][s]['y'],os.path.join(os.path.expanduser(pmnist_dir), 'data' + str(r) + s + 'y.bin'))
        print()

    else:

        # Load binary files
        for i, r in enumerate(seeds):
            data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
            data[i]['ncla'] = 10
            data[i]['name'] = 'pmnist-{:d}'.format(i)

            # Load
            for s in ['train', 'test']:
                data[i][s] = {'x': [], 'y': []}
                data[i][s]['x'] = torch.load(os.path.join(os.path.expanduser(pmnist_dir), 'data' + str(r) + s + 'x.bin'))
                data[i][s]['y'] = torch.load(os.path.join(os.path.expanduser(pmnist_dir), 'data' + str(r) + s + 'y.bin'))

    # Validation
    #for t in data.keys():
    #    r=np.arange(data[t]['train']['x'].size(0))
    #    # r=np.array(shuffle(r,random_state=seed),dtype=int)
    #    r=np.array(r,dtype=int)
    #    nvalid=int(pc_valid*len(r))
    #    ivalid=torch.LongTensor(r[:nvalid])
    #    itrain=torch.LongTensor(r[nvalid:])
    #    data[t]['valid'] = {}
    #    data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
    #    data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
    #    data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
    #    data[t]['train']['y']=data[t]['train']['y'][itrain].clone()

    # Others
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, size
