import numpy as np
import matplotlib.pyplot as plt
import math
import os

os.chdir('./../data')

from scipy import io

filename = ["C_Easy1_noise005.mat", "C_Easy1_noise01.mat", "C_Easy1_noise015.mat", "C_Easy1_noise02.mat",
            "C_Easy1_noise025.mat", "C_Easy1_noise03.mat", "C_Easy1_noise035.mat", "C_Easy1_noise04.mat",
            "C_Easy2_noise005.mat", "C_Easy2_noise01.mat", "C_Easy2_noise015.mat", "C_Easy2_noise02.mat",
            "C_Difficult1_noise005.mat", "C_Difficult1_noise01.mat", "C_Difficult1_noise015.mat", "C_Difficult1_noise02.mat",
            "C_Difficult2_noise005.mat", "C_Difficult2_noise01.mat", "C_Difficult2_noise015.mat", "C_Difficult2_noise02.mat"]

template =  ["Spike_TEMPLATE_e1n005.npy", "Spike_TEMPLATE_e1n010.npy", "Spike_TEMPLATE_e1n015.npy", "Spike_TEMPLATE_e1n020.npy",
             "Spike_TEMPLATE_e1n025.npy", "Spike_TEMPLATE_e1n030.npy", "Spike_TEMPLATE_e1n035.npy", "Spike_TEMPLATE_e1n040.npy",
             "Spike_TEMPLATE_e2n005.npy", "Spike_TEMPLATE_e2n010.npy", "Spike_TEMPLATE_e2n015.npy", "Spike_TEMPLATE_e2n020.npy",
             "Spike_TEMPLATE_d1n005.npy", "Spike_TEMPLATE_d1n010.npy", "Spike_TEMPLATE_d1n015.npy", "Spike_TEMPLATE_d1n020.npy",
             "Spike_TEMPLATE_d2n005.npy", "Spike_TEMPLATE_d2n010.npy", "Spike_TEMPLATE_d2n015.npy", "Spike_TEMPLATE_d2n020.npy"]


def making_naive_template():
    for ds in range(20):
        print("")
        print("data", ds)
        mat1 = io.loadmat(filename[ds])
        raw = mat1['data'][0]
        thr = 0.9
        
        wait = 0
        spike_index = 0
        spike = np.zeros((10000, 50))
        spike_times = np.zeros(10000)
        training_cycle = 100
        slope = np.zeros(len(raw)-2)
        for i in range(len(raw)-2):
            slope[i]=raw[i+1]-raw[i]
            
        for i in range(len(raw)-2):
            wait += 1
            if(21 <= wait):
                if(raw[i+1] < raw[i+2] and raw[i+1] <= raw[i] and raw[i+1] < -thr) or (raw[i+1] > raw[i+2] and raw[i] <= raw[i+1] and raw[i+1] > thr):
                    
                    max_slope_index = i + np.argmax(slope[i - 8 : i + 5]) - 8
                    spike[spike_index, :] = raw[max_slope_index - 10 : max_slope_index + 40]
                    spike_times[spike_index] = i-20
                    spike_index += 1        
                    wait = 0
        
        num_cluster = 3
        Cluster = np.zeros((num_cluster, 50))
        distance_size = 0
        cluster_num = np.zeros(num_cluster)
        
        for i in range(num_cluster):
            distance_size += i+1    
        distance = np.zeros(distance_size)
        
        for spike_index in range(training_cycle):
            spike_n = spike[spike_index, :]
            
            if(spike_index == 0):
                Cluster[0, :] = spike_n
                Cl_num = 1
                cluster_num[0] += 1
                        
            else:
                for i in range(num_cluster):
                    distance[i] = np.sum(abs(Cluster[i, 5:25] - spike_n[5:25])) * 17 + np.sum(abs(Cluster[i, 0:5] - spike_n[0:5])) * 2 + np.sum(abs(Cluster[i, 25:50] - spike_n[25:50])) * 2
                #if(spike_index == 4):
                   # print(distance)
                k = 0
                for j in range(1, num_cluster):
                    k = k + j
                    for i in range(j, num_cluster):

                        if(spike_index < 30):
                            mer_thr = 1.5
                        else:
                            mer_thr = 2.5
                            
                        if(cluster_num[j-1]>10) or (cluster_num[i]>10):
                            distance[i + j * num_cluster - k] = 1500000000000
                        else:
                            distance[i + j * num_cluster - k] = np.sum(abs(Cluster[j - 1, 5:25] - Cluster[i, 5:25])) * 17 + np.sum(abs(Cluster[j - 1, 0:5] - Cluster[i, 0:5])) * 2 + np.sum(abs(Cluster[j - 1, 25:50] - Cluster[i, 25:50])) * 2
                            distance[i + j * num_cluster - k] = distance[i + j * num_cluster - k] * mer_thr
                            
                m = np.argmin(distance)
                if(m < num_cluster):
                    
                    Cluster[m, :] = (Cluster[m, :] * 15 + spike_n)/16
                    cluster_num[m] += 1
                    
                    
                else:
                    x = num_cluster
                    i
                    for i in range(1, num_cluster):
                        y = x + num_cluster - i
                        if(x <= m and m < y):
                            Cluster[i - 1, :] = (Cluster[i - 1, :] + Cluster[m - x + i, :])/2
                            cluster_num[i-1] = cluster_num[i-1] + cluster_num[m-x+i]
                            Cluster[m - x + i, :] = spike_n
                            cluster_num[m-x+i] = 1
                        x = y
        np.save(template[ds], Cluster)
        
making_naive_template()