import numpy as np

def acc_det(spike_index, spike_times, ans_times):
        k = 0
        FN = 0
        TP = 0
        FP = 0
        spike_true_index = np.zeros(10000)
        spike_false_index = np.zeros(10000)
        ans_index = np.zeros(10000)
        spike_times[spike_times == 0] = 1500000
        det_win = 20

        '''
        f = open('./../result/TP_index.txt', 'w')
        g = open('./../result/FP_index.txt', 'w')
        h = open('./../result/FN_index.txt', 'w')
        '''
        for j in range(len(ans_times)):
                if(ans_times[j] + det_win >= spike_times[k] and spike_times[k] >= ans_times[j] - det_win):
                        spike_true_index[TP] = k
                        ans_index[TP] = j
                        #f.write('%7d %7d'%(ans_times[j], spike_times[k]) +"\n")
                        TP = TP + 1
                        k = k + 1
                elif(ans_times[j] + det_win < spike_times[k]):
                        FN = FN + 1
                        #h.write('%7d'%(ans_times[j]) +"\n")
                else:
                        while(1):
                                spike_false_index[FP] = k
                                FP = FP + 1
                                #g.write('%7d'%(spike_times[k]) + "\n")
                                k = k + 1
                                if(ans_times[j] - det_win <= spike_times[k]):
                                        break
                        if(ans_times[j] + det_win >= spike_times[k]):
                                spike_true_index[TP] = k
                                ans_index[TP] = j
                                #f.write('%7d %7d'%(ans_times[j], spike_times[k]) +"\n")
                                TP = TP + 1
                                k = k + 1
                        else:
                                FN = FN + 1
				#h.write('%7d'%(ans_times[j]) +"\n")
        print("# of ans : ", len(ans_times))
        print("# of TP ; ", TP)
        print("# of FP ; ", FP)
        print("# of FN : ", FN)
        print("Det acc : ", TP/len(ans_times))
        return spike_true_index, spike_false_index, ans_index, TP, TP/len(ans_times)
''''
def acc_clu(numspike, spike_id, TP, spike_true_index, ans_index, ans_cluster):
	cluster_accuracy = np.zeros(6)
	for ep in range(6):
		if(ep == 1 or ep == 4):
			for i in range(numspike):
				if(spike_id[i] == 3):
					spike_id[i] = 2
				elif(spike_id[i] == 2):
					spike_id[i] = 3
		elif(ep == 2 or ep == 5):
			for i in range(numspike):
				if(spike_id[i] == 1):
					spike_id[i] = 2
				elif(spike_id[i] == 2):
					spike_id[i] = 1
		elif(ep == 3):
			for i in range(numspike):
				if(spike_id[i] == 1):
					spike_id[i] = 3
				elif(spike_id[i] == 3):
					spike_id[i] = 1
		true_cluster = 0
		for i in range(TP):
			if(spike_true_index[i] == 0) and (spike_true_index[i+1] == 0):
				print("break")
				break
			else:
				if(spike_id[int(spike_true_index[i])] == ans_cluster[int(ans_index[i])]):
					true_cluster += 1
		cluster_accuracy[ep] = true_cluster*100/TP
	print('Clu acc : ', max(cluster_accuracy))
	return max(cluster_accuracy)
'''

def acc(spike_index, spike_times, ans_times, spike_id, ans_cluster, training = 0):
        k = 0
        FN = 0
        TP = 0
        FP = 0
        spike_times[spike_times == 0] = 1500000
        det_win = 20
        id_ssp = np.zeros(10000)
        id_ans = np.zeros(10000)
        id_false = np.zeros(10000)
        training_TP = 0
        training_ans = 0
        training_cycle = 100
        for j in range(len(ans_times)):
                if(ans_times[j] + det_win >= spike_times[k] and spike_times[k] >= ans_times[j] - det_win):
                        id_ssp[TP] = spike_id[k]
                        id_ans[TP] = ans_cluster[j]		
                        TP = TP + 1
                        k = k + 1
                        if(k == training_cycle and training == 1):
                                training_TP = TP
                                training_ans = j
                elif(ans_times[j] + det_win < spike_times[k]):
                        FN = FN + 1
                else:
                        while(1):
                                id_false[FP] = spike_id[k]
                                FP = FP + 1

                                k = k + 1
                                if(k == training_cycle and training == 1):
                                        training_TP = TP
                                        training_ans = j

                                if(ans_times[j] - det_win <= spike_times[k]):
                                        break
                        if(ans_times[j] + det_win >= spike_times[k]):
                                id_ssp[TP] = spike_id[k]
                                id_ans[TP] = ans_cluster[j]
                                TP = TP + 1
                                k = k + 1
                                if(k == training_cycle and training == 1):
                                        training_TP = TP
                                        training_ans = j
                        else:
                                FN = FN + 1
        #print(training_TP, training_ans)
        print("# of ans : ", len(ans_times))
        print("# of TP ; ", TP)
        print('training miss : ', training_TP)
        print('# of Error : ', len(ans_times)-(TP-training_TP))
        
        #print("# of FP ; ", FP)
        print("# of FN : ", FN)
        print("Det acc : ", (TP-training_TP)/(len(ans_times)-training_ans))

        filtered_spike = 0
        filtered_noise = 0
        cluster_accuracy = np.zeros(6)
        true_clusters = np.zeros(6)
        noise = 0
        for i in range(TP):
                if(id_ssp[i] == 4):
                        filtered_spike += 1
        for i in range(FP):
                if(id_false[i] == 4):
                        filtered_noise += 1
        for ep in range(6):
                if(ep == 1 or ep == 4):
                        for i in range(spike_index):
                                if(id_ssp[i] == 3):
                                        id_ssp[i] = 2
                                elif(id_ssp[i] == 2):
                                        id_ssp[i] = 3
                elif(ep == 2 or ep == 5):
                        for i in range(spike_index):
                                if(id_ssp[i] == 1):
                                        id_ssp[i] = 2
                                elif(id_ssp[i] == 2):
                                        id_ssp[i] = 1
                elif(ep == 3):
                        for i in range(spike_index):
                                if(id_ssp[i] == 1):
                                        id_ssp[i] = 3
                                elif(id_ssp[i] == 3):
                                        id_ssp[i] = 1
                true_cluster = 0
                for i in range(training_TP, TP):
                        if(id_ssp[i] == id_ans[i]):
                                true_cluster += 1
                
                cluster_accuracy[ep] = true_cluster*100/(TP-filtered_spike-training_TP)
                true_clusters[ep] = true_cluster
        #print('filtered noise : ', filtered_noise)
        print('filtered spike : ', filtered_spike)
        print("true cluster : ", max(true_clusters))
        print('filtered FP : ', FP-filtered_noise)
        print('Final det acc : ', (TP-filtered_spike-training_TP)/(len(ans_times)-training_ans))

        print('Clu acc : ', max(cluster_accuracy))
	
        return (TP-training_TP-filtered_spike)/(len(ans_times)-training_ans), max(cluster_accuracy), max(true_clusters)

