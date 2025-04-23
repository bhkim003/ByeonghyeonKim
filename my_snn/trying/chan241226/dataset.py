import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import os

os.chdir("./../data/")
'''
filename = ["C_Easy1_noise005.mat", "C_Easy1_noise01.mat", "C_Easy1_noise015.mat", "C_Easy1_noise02.mat",
            "C_Easy2_noise005.mat", "C_Easy2_noise01.mat", "C_Easy2_noise015.mat", "C_Easy2_noise02.mat",
            "C_Difficult1_noise005.mat", "C_Difficult1_noise01.mat", "C_Difficult1_noise015.mat", "C_Difficult1_noise02.mat",
            "C_Difficult2_noise005.mat", "C_Difficult2_noise01.mat", "C_Difficult2_noise015.mat", "C_Difficult2_noise02.mat"]

'''
filename = ["C_Easy1_noise005.mat", "C_Easy1_noise01.mat", "C_Easy1_noise015.mat", "C_Easy1_noise02.mat",
            "C_Easy1_noise025.mat", "C_Easy1_noise03.mat", "C_Easy1_noise035.mat", "C_Easy1_noise04.mat",
            "C_Easy2_noise005.mat", "C_Easy2_noise01.mat", "C_Easy2_noise015.mat", "C_Easy2_noise02.mat",
            "C_Difficult1_noise005.mat", "C_Difficult1_noise01.mat", "C_Difficult1_noise015.mat", "C_Difficult1_noise02.mat",
            "C_Difficult2_noise005.mat", "C_Difficult2_noise01.mat", "C_Difficult2_noise015.mat", "C_Difficult2_noise02.mat"]

dataset_num = 20
training_num = 2400
spike_length = 50

training_spike_group = np.zeros((dataset_num, training_num, spike_length))

for ds in range(20):
    mat1 = io.loadmat(filename[ds])
    raw = mat1['data'][0]
    ans_times = mat1['spike_times'][0][0][0]
    ans_cluster = mat1['spike_class'][0][0][0]

    slope = np.zeros(len(raw)-2)
    for i in range(len(raw)-2):
        slope[i] = raw[i+1] - raw[i]

    spike_group = np.zeros((len(ans_times), spike_length))

    for i in range(len(ans_times)):
        max_slope_index = ans_times[i] + np.argmax(slope[ans_times[i] : ans_times[i] + 25])
        spike_group[i, :] = raw[max_slope_index - 10 : max_slope_index + 40]

    '''#max_slope_idx_check
    max_slope_index = np.zeros(len(ans_times))
    
    for i in range(len(ans_times)):
        slope_value = np.zeros(49)
        for j in range(49):
            slope_value[j] = abs(spike_group[i, j+1] - spike_group[i, j])
        max_slope_index[i] = np.argmax(slope_value)
    print(max_slope_index)

    x = np.arange(0, 50, 1)
    plt.figure()
    plt.plot(x, spike_group[0, :])
    plt.savefig('./../result_net/etc/spike.svg')

    '''
    training_spike_group[ds, :, :] = spike_group[:training_num, :]

training_spike_group_reshape = training_spike_group.reshape(-1, spike_length)

np.random.shuffle(training_spike_group_reshape)

# check dataset
x = np.arange(0, 50, 1)
plt.figure()
plt.plot(x, training_spike_group_reshape[1, :])
plt.savefig('./../result_net/etc/spike.svg')
print(np.shape(training_spike_group_reshape))


np.save('./../data/training_dataset_20', training_spike_group_reshape)