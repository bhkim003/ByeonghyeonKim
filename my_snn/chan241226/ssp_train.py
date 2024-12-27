import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir("./../data/")

filename = ["C_Easy1_noise005.mat", "C_Easy1_noise01.mat", "C_Easy1_noise015.mat", "C_Easy1_noise02.mat",
            "C_Easy2_noise005.mat", "C_Easy2_noise01.mat", "C_Easy2_noise015.mat", "C_Easy2_noise02.mat",
            "C_Difficult1_noise005.mat", "C_Difficult1_noise01.mat", "C_Difficult1_noise015.mat", "C_Difficult1_noise02.mat",
            "C_Difficult2_noise005.mat", "C_Difficult2_noise01.mat", "C_Difficult2_noise015.mat", "C_Difficult2_noise02.mat"]

template =  ["Spike_TEMPLATE_e1n005_new.npy", "Spike_TEMPLATE_e1n010_new.npy", "Spike_TEMPLATE_e1n015_new.npy", "Spike_TEMPLATE_e1n020_new.npy",
             "Spike_TEMPLATE_e2n005_new.npy", "Spike_TEMPLATE_e2n010_new.npy", "Spike_TEMPLATE_e2n015_new.npy", "Spike_TEMPLATE_e2n020_new.npy",
             "Spike_TEMPLATE_d1n005_new.npy", "Spike_TEMPLATE_d1n010_new.npy", "Spike_TEMPLATE_d1n015_new.npy", "Spike_TEMPLATE_d1n020_new.npy",
             "Spike_TEMPLATE_d2n005_new.npy", "Spike_TEMPLATE_d2n010_new.npy", "Spike_TEMPLATE_d2n015_new.npy", "Spike_TEMPLATE_d2n020_new.npy"]

spike_tot = ["Spike_e1n005.npy", "Spike_e1n010.npy", "Spike_e1n015.npy", "Spike_e1n020.npy",
            "Spike_e2n005.npy", "Spike_e2n010.npy", "Spike_e2n015.npy", "Spike_e2n020.npy",
            "Spike_d1n005.npy", "Spike_d1n010.npy", "Spike_d1n015.npy", "Spike_d1n020.npy",
            "Spike_d2n005.npy", "Spike_d2n010.npy", "Spike_d2n015.npy", "Spike_d2n020.npy"]

times_tot = ['Spike_e1n005_times.npy', 'Spike_e1n010_times.npy', 'Spike_e1n015_times.npy', 'Spike_e1n020_times.npy',
             'Spike_e2n005_times.npy', 'Spike_e2n010_times.npy', 'Spike_e2n015_times.npy', 'Spike_e2n020_times.npy',
             'Spike_d1n005_times.npy', 'Spike_d1n010_times.npy', 'Spike_d1n015_times.npy', 'Spike_d1n020_times.npy',
             'Spike_d2n005_times.npy', 'Spike_d2n010_times.npy', 'Spike_d2n015_times.npy', 'Spike_d2n020_times.npy']

thr_tot = np.array([0.5, 0.5, 0.55, 0.7, 0.5, 0.5, 0.55, 0.7, 0.5, 0.5, 0.55, 0.7, 0.5, 0.5, 0.55, 0.7])
cos_thr = np.array([0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.85, 0.95, 0.9, 0.8, 0.95, 0.95, 0.95, 0.95, 0.8])
#%%

from scipy import io

cluster_ans = np.zeros((16, 4000))

for i in range(16):
    mat1 = io.loadmat(filename[i])        
    cluster_ans[i, :len(mat1['spike_class'][0][0][0])] = mat1['spike_class'][0][0][0]


from torch.utils.data import DataLoader, Dataset

class spikedataset(Dataset):

    def __init__(self, path, transform = None):    
        
        self.transform = transform
        spike_h = np.load(path)
        self.spike = spike_h
        self.len = len(self.spike)
        
    def __getitem__(self, index):
        spike = self.spike[index]            
        if self.transform is not None:
            spike = self.transform(spike)
        return spike
    
    def __len__(self):
        return self.len

train_dataset = spikedataset('training_dataset_20.npy')
train_loader = DataLoader(dataset = train_dataset, batch_size = 32, shuffle = True)
batch_size = 32
#%%

import torch 
import torch.nn as nn
import torch.nn.functional as F

class AE(nn.Module):

    def __init__(self):
        super(AE, self).__init__()

        # encoder
        self.conv1 = nn.Conv1d(1, 32, 3, stride = 2, bias = False) # 24
        self.conv2 = nn.Conv1d(32, 64, 3, stride = 2, bias = False) # 11
        self.conv3 = nn.Conv1d(64, 96, 3, stride = 2, bias = False) # 4 # 병현: 여기 5인데?
        self.fc1 = nn.Linear(96 * 5, 4, bias = False)
        
        # decoder
        self.fc4 = nn.Linear(4, 5 * 96, bias = False)
        self.deconv3 = nn.ConvTranspose1d(96, 64, 3, stride = 2, bias = False) #6 + 2 + 1= 9
        self.deconv1 = nn.ConvTranspose1d(64, 32, 3, stride = 2, output_padding=1, bias = False) #16(9-1)*stride + 4(kernel-1) + 1 = 21
        self.deconv2 = nn.ConvTranspose1d(32, 1, 3, stride = 2, output_padding=1, bias = False) #40 + 4 + 1 = 45
       

    def forward(self, x):

        # encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 96 * 5)
        mid = self.fc1(x)
        norm = torch.sqrt(torch.sum(torch.pow(mid, 2), dim = 1))
        h = (mid.t()/norm).t()

        # decoder
        z = F.relu(self.fc4(h))
        z = z.view(-1, 96, 5)
        z = F.relu(self.deconv3(z))
        z = F.relu(self.deconv1(z))
        z = self.deconv2(z)

        return h, z

net = AE()
#device = torch.device('cpu')

#net.load_state_dict(torch.load('allnobiasv2_7000.pth' , map_location = device))
#net.eval()
device_num = 0
net.cuda(device_num)
#%%
import torch.optim as optim
from scipy import io

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

training_cycle = 2400


num_cluster = 3
distance_size = 0
for i in range(num_cluster):
    distance_size += i+1
tau = np.zeros(3)


acc_y = np.zeros(50000)
acc_y2 = np.zeros(50000)
acc_y_test = np.zeros(50000)

acc_update = 0
#%%
max_epoch = 7000
for epoch in range(max_epoch):

    
    
    cluster_acc_data = np.zeros(16)
    cluster_acc_data2 = np.zeros(16)
    cluster_test = np.zeros(16)    
    
    
    
    if(epoch == 0 or epoch == 1 or epoch % 1000 == 999): 
    
        for ds in range(16):
            thr = thr_tot[ds]
            spike_template = np.load(template[ds])
            spike = np.load(spike_tot[ds])
            ans_cluster = cluster_ans[ds]
            times = np.load(times_tot[ds])
            print(ds+1)
            
            Cluster = np.zeros((3, 4))
            
            for i in range(3):
                spike_torch = torch.from_numpy(spike_template[i, :])
                spike_torch = spike_torch.view(1, 1, 50)
                inner_inf, spike_class = net(spike_torch.float().cuda(device_num))
                
                Cluster[i, :] = inner_inf.cpu().detach().numpy()
                
            b = np.zeros((len(spike), 4))
            
            
            for i in range(len(spike)):
                spike_torch = torch.from_numpy(spike[i, :])
                spike_torch = spike_torch.view(1, 1, 50)
                inner_inf, spike_class = net(spike_torch.float().cuda(device_num))
            
                b[i, :] = inner_inf.cpu().detach().numpy()
                
            spike_id = np.zeros(len(spike))
            distance = np.zeros(distance_size)
            tau = np.zeros(3)
            
            cos = cos_thr[ds]
            
            for spike_index in range(2400):
                ft1, ft2, ft3, ft4 = b[spike_index, :]
                for q in range(3):
                    tau[q] = np.sum(b[spike_index, :] * Cluster[q, :])
                for i in range(num_cluster):
                    distance[i] = pow(abs(Cluster[i, 0] - ft1), 2) + pow(abs(Cluster[i, 1] - ft2), 2) + pow(abs(Cluster[i, 2] - ft3), 2) + pow(abs(Cluster[i, 3] - ft4), 2)
            
                if(np.max(tau)>=cos):
                    k = 0
                    for j in range(1, num_cluster):
                        k = k + j
                        for i in range(j, num_cluster):
                            distance[i + j * num_cluster - k] = np.sum(pow(Cluster[j-1, :] - Cluster[i, :], 2))*1500
                            
                    m = np.argmin(distance)
                    if(m < num_cluster):
                        Cluster[m, 0] = (Cluster[m, 0] * 15 + ft1)/16
                        Cluster[m, 1] = (Cluster[m, 1] * 15 + ft2)/16
                        Cluster[m, 2] = (Cluster[m, 2] * 15 + ft3)/16
                        Cluster[m, 3] = (Cluster[m, 3] * 15 + ft4)/16
                        
                    else:
                        x = num_cluster
                                        
                        for i in range(1, num_cluster):
                            y = x + num_cluster - i
                            if(x <= m and m < y):
                                Cluster[i - 1, :] = (Cluster[i - 1, :] * 15 + Cluster[m - x + i, :])/16
                                Cluster[m - x + i, :] = (ft1, ft2, ft3, ft4)
                            x = y
                            
            for spike_index in range(len(spike)):
                ft1, ft2, ft3, ft4 = b[spike_index, :]
                for q in range(3):
                    tau[q] = np.sum(b[spike_index, :] * Cluster[q, :])
                for i in range(num_cluster):
                    distance[i] = pow(abs(Cluster[i, 0] - ft1), 2) + pow(abs(Cluster[i, 1] - ft2), 2) + pow(abs(Cluster[i, 2] - ft3), 2) + pow(abs(Cluster[i, 3] - ft4), 2)
                
                if(np.max(tau)< -1):
                    spike_id[spike_index] = 4
                else:
                    m = np.argmin(distance[0:num_cluster])
                    spike_id[spike_index] = m + 1
            
            cluster_accuracy = np.zeros(6)
            cluster_accuracy2 = np.zeros(6)
            cluster_accuracy3 = np.zeros(6)
            
            for ep in range(6):
                
                if(ep == 1 or ep == 4):
                    for i in range(len(spike)):
                        if(spike_id[i] == 3):
                            spike_id[i] = 2
                        elif(spike_id[i] == 2):
                            spike_id[i] = 3
                            
                elif(ep == 2 or ep == 5):
                    for i in range(len(spike)):
                        if(spike_id[i] == 1):
                            spike_id[i] = 2
                        elif(spike_id[i] == 2):
                            spike_id[i] = 1
                elif(ep == 3):
                    for i in range(len(spike)):
                        if(spike_id[i] == 1):
                            spike_id[i] = 3
                        elif(spike_id[i] == 3):
                            spike_id[i] = 1
                
                true_cluster = 0
                for i in range(int(times[0, 2])):
                    if(times[i, 0] == 0) and (times[i + 1, 0] == 0):
                        print("break")
                        break
                    
                    else:
                        if(spike_id[int(times[i, 0])]==ans_cluster[int(times[i, 1])]):
                            true_cluster += 1
                        
                cluster_accuracy[ep] = true_cluster*100/times[0, 2]
                            
                true_cluster2 = 0
                for i in range(int(times[0, 2]), int(times[1, 2])):
                    if(times[i, 0] == 0) and (times[i + 1, 0] == 0):
                        print("break")
                        break
                    else:
                        if(spike_id[int(times[i, 0])]==ans_cluster[int(times[i, 1])]):
                            true_cluster2 += 1
                cluster_accuracy2[ep] = true_cluster2*100/(times[1, 2]-times[0, 2])
                cluster_accuracy3[ep] = (true_cluster + true_cluster2)/times[1, 2]
                
            cluster_acc_data[ds] = max(cluster_accuracy)
            print("training acc : ", max(cluster_accuracy))
            print('test acc : ', cluster_accuracy2[np.argmax(cluster_accuracy)])
            cluster_acc_data2[ds] = cluster_accuracy3[np.argmax(cluster_accuracy)]
            cluster_test[ds] = cluster_accuracy2[np.argmax(cluster_accuracy)]
        print("training acc : ", np.sum(cluster_acc_data)/16)
        print("acc : ", np.sum(cluster_acc_data2)/16)
        
        acc_y[epoch] = np.sum(cluster_acc_data)/16
        acc_y2[epoch] = np.sum(cluster_acc_data2)/16
        acc_y_test[epoch] = np.sum(cluster_test)/16
        
        torch.save(net.state_dict(), './../result_net/original/new_data_net_%depoch_%0.3f.pth' % (epoch, np.sum(cluster_acc_data2)/16))

    running_loss = 0.0
   # for i in range(len(spike_normed)):
    for i in range(int(2400*20/batch_size)):
        optimizer.zero_grad()
        spike_torch = next(iter(train_loader)).cuda(device_num)
        spike_torch = spike_torch.view(-1, 1,  50)
        spike_torch = spike_torch.float()
        inner_inf, spike_class = net(spike_torch)
        
        loss1 = criterion(spike_class[:, 0, 5:25], spike_torch[:, 0, 5:25])
        loss2 = criterion(spike_class[:, 0, 0:5], spike_torch[:, 0, 0:5])
        loss3 = criterion(spike_class[:, 0, 25:50], spike_torch[:, 0, 25:50])
        loss = loss1 * 2.125 + (loss2 + loss3)/4
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
#%%
x = np.arange(0, 9, 1)
y = np.zeros(9)
z = np.zeros(9)
w = np.zeros(9)

y[0] = acc_y[0]
y[1] = acc_y[1]
y[2] = acc_y[1000-1]
y[3] = acc_y[2000-1]
y[4] = acc_y[3000-1]
y[5] = acc_y[4000-1]
y[6] = acc_y[5000-1]
y[7] = acc_y[6000-1]
y[8] = acc_y[7000-1]

z[0] = acc_y2[0]
z[1] = acc_y2[1]
z[2] = acc_y2[1000-1]
z[3] = acc_y2[2000-1]
z[4] = acc_y2[3000-1]
z[5] = acc_y2[4000-1]
z[6] = acc_y2[5000-1]
z[7] = acc_y2[6000-1]
z[8] = acc_y2[7000-1]

print(np.max(z), np.argmax(z))

w[0] = acc_y_test[0]
w[1] = acc_y_test[1]
w[2] = acc_y_test[1000-1]
w[3] = acc_y_test[2000-1]
w[4] = acc_y_test[3000-1]
w[5] = acc_y_test[4000-1]
w[6] = acc_y_test[5000-1]
w[7] = acc_y_test[6000-1]
w[8] = acc_y_test[7000-1]

plt.figure()
plt.plot(x, y)
plt.figure()
plt.plot(x, z)
#for i, v in enumerate(x):
#    plt.text(v, z[i], str(z[i]), fontsize = 10, weight = 'bold', color = "red", horizontalalignment='center', verticalalignment='bottom')
plt.savefig('./../result_net/original/accuracy.svg')

plt.figure()
plt.plot(x, w)

