
import random
import os

from jinja2 import pass_environment

from modules.data_loader import *
from modules.network import *
from modules.neuron import *
from modules.synapse import *
from modules.old_fashioned import *
from modules.ae_network import *

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE
import itertools

def round_away_from_zero(x):
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)

def round_hardware_good(x):
    return torch.where(x >= 0, torch.trunc(x), torch.floor(x))

def seed_assign(seed):
    random.seed(seed)                          # Python random 시드 고정
    np.random.seed(seed)                       # NumPy 시드 고정
    torch.manual_seed(seed)                    # PyTorch CPU 시드 고정
    torch.cuda.manual_seed(seed)               # PyTorch GPU 시드 고정
    torch.cuda.manual_seed_all(seed)           # PyTorch 멀티 GPU 시드 고정
    torch.backends.cudnn.deterministic = True  # 연산의 결정론적 동작 보장
    # torch.backends.cudnn.benchmark = False     # 성능 최적화 비활성화 (결정론적 보장)
    # torch.use_deterministic_algorithms(True)
    # torch.set_default_dtype(torch.float32)
    # torch.backends.cudnn.allow_tf32 = False  # TF32 연산 비활성화
    # torch.backends.cuda.matmul.allow_tf32 = False  # matmul 연산에서도 TF32 사용 금지


def plot_distributions(ds, plot_tau, plot_denominator, plot_m, plot_max_tau, cos_thr_ds,
                       tr_cycle_acc, post_tr_cycle_acc, total_cycle_acc):
    """
    plot_tau, plot_denominator, plot_m, plot_max_tau 값을 시각화하는 함수.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. Tau 값의 히스토그램
    plt.figure(figsize=(10, 5))
    plt.hist(plot_tau, bins=50, alpha=0.7, color='blue', label='tau')
    plt.axvline(np.mean(plot_tau), color='red', linestyle='dashed', linewidth=1, label='Mean Tau')
    plt.xlabel('Tau Values')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Tau Values, dataset: {ds}\n'
              f'tr_cycle_acc: {tr_cycle_acc:.2f}, post_tr_cycle_acc: {post_tr_cycle_acc:.2f}, '
              f'total_cycle_acc: {total_cycle_acc:.2f}')
    plt.legend()
    plt.show()

    # 2. Denominator 값의 히스토그램
    plt.figure(figsize=(10, 5))
    plt.hist(plot_denominator, bins=50, alpha=0.7, color='orange', label='denominator')
    plt.axvline(np.mean(plot_denominator), color='red', linestyle='dashed', linewidth=1, label='Mean Denominator')
    plt.xlabel('Denominator Values')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Denominator Values, dataset: {ds}\n'
              f'tr_cycle_acc: {tr_cycle_acc:.2f}, post_tr_cycle_acc: {post_tr_cycle_acc:.2f}, '
              f'total_cycle_acc: {total_cycle_acc:.2f}')
    plt.legend()
    plt.show()

    # 3. M 값의 히스토그램 (클러스터 인덱스 분포)
    plt.figure(figsize=(10, 5))
    unique_m, counts_m = np.unique(plot_m, return_counts=True)
    plt.bar(unique_m, counts_m, alpha=0.7, color='purple')
    plt.xlabel('Cluster Index (M)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Cluster Indices (M), dataset: {ds}\n'
              f'tr_cycle_acc: {tr_cycle_acc:.2f}, post_tr_cycle_acc: {post_tr_cycle_acc:.2f}, '
              f'total_cycle_acc: {total_cycle_acc:.2f}')
    plt.xticks(unique_m)  # 클러스터 인덱스 표시
    plt.grid(axis='y')
    plt.show()

    # 4. Max Tau 값의 히스토그램
    cos_thr_percentile = 100 - (np.sum(np.array(plot_max_tau) < cos_thr_ds) / len(plot_max_tau) * 100)
    plt.figure(figsize=(10, 5))
    plt.hist(plot_max_tau, bins=50, alpha=0.7, color='green', label='max_tau')
    plt.axvline(cos_thr_ds, color='red', linestyle='dashed', linewidth=1, label=f'cos_thr: {cos_thr_ds:.2f}')
    plt.xlabel('Max Tau Values')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Max Tau Values, dataset: {ds}\n'
              f'cos_thr_ds is in the top {cos_thr_percentile:.2f}% of max_tau\n'
              f'tr_cycle_acc: {tr_cycle_acc:.2f}, post_tr_cycle_acc: {post_tr_cycle_acc:.2f}, '
              f'total_cycle_acc: {total_cycle_acc:.2f}')
    plt.legend()
    plt.text(
        cos_thr_ds, plt.gca().get_ylim()[1] * 0.9,
        f'{cos_thr_percentile:.2f}% (cos_thr_ds: {cos_thr_ds:.2f})',
        color='red', fontsize=12, verticalalignment='top'
    )
    plt.show()

    # 5. Max Tau vs. Cosine Threshold 비교 (산점도)
    plt.figure(figsize=(10, 5))
    plt.plot(plot_max_tau, marker='o', linestyle='', color='blue', alpha=0.5, label='Max Tau')
    plt.axhline(cos_thr_ds, color='red', linestyle='dashed', linewidth=1, label=f'cos_thr: {cos_thr_ds:.2f}')
    plt.xlabel('Spike Index')
    plt.ylabel('Max Tau Value')
    plt.title(f'Max Tau Values vs. Cosine Threshold, dataset: {ds}\n'
              f'cos_thr_ds is in the top {cos_thr_percentile:.2f}% of max_tau\n'
              f'tr_cycle_acc: {tr_cycle_acc:.2f}, post_tr_cycle_acc: {post_tr_cycle_acc:.2f}, '
              f'total_cycle_acc: {total_cycle_acc:.2f}')
    plt.text(
        len(plot_max_tau) * 0.7, cos_thr_ds, f'{cos_thr_percentile:.2f}%', color='red', fontsize=12, verticalalignment='bottom'
    )
    plt.legend()
    plt.grid()
    plt.show()

def plot_spike(spike, title="Spike Visualization (Black & White)"):
    """
    Spike 데이터를 검은색으로 시각화하는 함수.
    가로축: Timestep
    세로축: Feature
    각 요소를 구분하는 격자가 추가되며, x축과 y축에 일정한 간격으로 숫자 눈금을 표시함.
    """
    spike = spike.squeeze()
    spike[:, :] = spike[:, ::-1]  # Flip horizontally

    plt.figure(figsize=(10, 6))
    plt.imshow(spike.T, aspect='auto', cmap='Greys', interpolation='nearest')

    # x축 눈금 설정 (Timestep)
    x_ticks = np.arange(spike.shape[0])
    plt.xticks(x_ticks)  # X축 눈금 설정

    # y축 눈금을 feature 개수만큼 설정
    y_ticks = np.arange(spike.shape[1])
    plt.yticks(y_ticks)  # Y축 눈금 설정

    # 격자 추가
    plt.grid(visible=True, which='minor', color='gray', linestyle='--', linewidth=0.5)
    plt.gca().set_xticks(np.arange(-0.5, spike.shape[0], 1), minor=True)
    plt.gca().set_yticks(np.arange(-0.5, spike.shape[1], 1), minor=True)
    plt.grid(which='minor', color='gray', linestyle='--', linewidth=0.5)

    # 색상 막대와 라벨 추가
    plt.colorbar(label='Spike Value')
    plt.xlabel('Timestep')
    plt.ylabel('Feature')
    plt.title(title)

    plt.show()


def plot_origin_spike(spike, min_max_y_on = False):
    # 최소값과 최대값 계산
    if min_max_y_on == True:
        min_val = np.min(spike)
        max_val = np.max(spike)
    else:
        min_val = -2
        max_val = 2
    # min_val = 0
    # max_val = 1

    # 플로팅
    plt.figure(figsize=(8, 4))
    plt.plot(spike, marker='o', linestyle='-', color='b', label='Spike Feature')
    plt.title('Original Spike')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.ylim(min_val, max_val)  # y축 범위를 min_val에서 max_val로 설정
    plt.legend()
    plt.grid(True)
    plt.show()



def cluster_spikes_with_accuracy_torch(features, true_labels, n_clusters, init_point=None, reset_num = 10):
    """
    Perform k-means clustering and calculate clustering accuracy using PyTorch.

    Parameters:
        features (torch.Tensor): 2D tensor with shape [spike_num, feature].
        true_labels (torch.Tensor): 1D tensor with the true labels for each spike.
        n_clusters (int): The number of clusters for k-means.
        init_point (torch.Tensor or None): Initial centroids for k-means, if provided.

    Returns:
        tuple: (cluster_labels, accuracy)
            - cluster_labels (torch.Tensor): Cluster labels for each spike.
            - accuracy (float): Clustering accuracy.
    """
    if isinstance(true_labels, np.ndarray):
        true_labels = torch.from_numpy(true_labels).to(features.device)
    else:
        true_labels = true_labels.to(features.device)

    def kmeans(features, n_clusters, init_point, max_iter=1000, tol=1e-4):
        # Initialize centroids
        if init_point is None:
            centroids = features[torch.randperm(features.size(0))[:n_clusters]]
        else:
            centroids = init_point
        
        for _ in range(max_iter):
            # Compute distances and assign clusters
            distances = torch.cdist(features, centroids)
            labels = torch.argmin(distances, dim=1)
            
            # Update centroids
            new_centroids = torch.stack([features[labels == i].mean(dim=0) for i in range(n_clusters)])
            
            # Check for convergence
            if torch.allclose(centroids, new_centroids, atol=tol):
                break
            centroids = new_centroids

        return labels, centroids

    max_acc_best = 0
    for i in range(reset_num):
        # Perform k-means clustering
        cluster_labels, _ = kmeans(features, n_clusters, init_point)
        
        # Convert cluster labels for accuracy calculation
        cluster_labels_one_start = cluster_labels + 1  # [0, 1, 2] -> [1, 2, 3]
        label_converter_ground = list(range(1, n_clusters + 1))  # [1, 2, 3]
        label_converter_permutations = list(itertools.permutations(label_converter_ground))  # All permutations
        
        acc_bin = []
        for perm in label_converter_permutations:
            label_converter = torch.tensor(perm, dtype=torch.int32, device=features.device)
            acc = torch.sum(label_converter[cluster_labels_one_start - 1] == (true_labels +1)).item()
            acc_bin.append(acc)
        
        max_acc = max(acc_bin) / len(true_labels)
        if max_acc_best < max_acc:
            max_acc_best = max_acc
            cluster_labels_best = cluster_labels

    return max_acc_best

def cluster_spikes_with_accuracy(features, true_labels, n_clusters, init_point):
    """
    Perform k-means clustering and calculate clustering accuracy.

    Parameters:
        features (numpy.ndarray): 2D array with shape [spike_num, feature].
        true_labels (numpy.ndarray): 1D array with the true labels for each spike.
        n_clusters (int): The number of clusters for k-means.

    Returns:
        tuple: (cluster_labels, accuracy)
            - cluster_labels (numpy.ndarray): Cluster labels for each spike.
            - accuracy (float): Clustering accuracy.
    """
    # Perform k-means clustering
    if init_point is None:
        kmeans = KMeans(n_clusters=n_clusters,n_init='auto', random_state=42)
    else:
        init_point = init_point[:n_clusters]
        kmeans = KMeans(n_clusters=n_clusters, init=init_point, n_init=1, random_state=42)
    cluster_labels = kmeans.fit_predict(features)

    cluster_labels_one_start = cluster_labels + 1 # [0, 1, 2] -> [1, 2, 3]로 변환
    label_converter_ground = list(range(1, n_clusters + 1)) # [1, 2, 3] 생성
    label_converter_permutations = list(itertools.permutations(label_converter_ground)) # 모든 순열 구하기
    acc_bin = []
    for perm in label_converter_permutations:
        label_converter = list(perm)
        acc = 0    
        for i in range(len(features)):
            if(label_converter[int(cluster_labels_one_start[i]-1)] == true_labels[i]+1):
                acc += 1
        acc_bin.append(acc)
    max_acc = max(acc_bin)/len(true_labels)

    return max_acc

# Example usage:
# features = np.random.rand(100, 4)  # Replace with your spike feature array
# true_labels = np.random.randint(0, 3, 100)  # Replace with your true labels
# cluster_labels, accuracy = cluster_spikes_with_accuracy(features, true_labels, n_clusters=3)
# print("Cluster Labels:", cluster_labels)
# print("Accuracy:", accuracy)


def zero_to_one_normalize_features(spike, level_num, coarse_com_config, scaling=1.0, norm01=True):
    if norm01 == False:
        # print('level_num', level_num)
        spike = spike * scaling
        levels = torch.linspace(coarse_com_config[1], coarse_com_config[0], level_num, device=spike.device)  # 0에서 1까지 균등한 level_num 개의 값
        # print('levels', levels, len(levels))
        bucketized = torch.bucketize(spike, levels)
        # print('bucketized', bucketized[0])
        bucketized[bucketized == 0] = 1 # bucketize 결과에서 0인 요소를 1로 변경
        # print('bucketized2', bucketized[0])
        spike_normalized = levels[bucketized - 1] # 최종 양자화된 값 매핑
        # print('spike_normalized[0]', spike_normalized[0])
        # print('spike_ori', spike[0])
        # plot_origin_spike(spike_normalized[0].cpu().detach().numpy(), min_max_y_on=False)
    else:
        min_val = torch.min(spike, dim=1, keepdim=True)[0]
        max_val = torch.max(spike, dim=1, keepdim=True)[0]
        
        # Min-Max Normalization
        spike_normalized = (spike - min_val) / (max_val - min_val + 1e-12)

        spike_normalized = spike_normalized * scaling

        ## 실험용 ### ### ### ###
        # -0.5부터 0.5로 정규화
        # spike_normalized = spike_normalized
        ## 실험용 ###############

        # Quantization: level_num 단계로 매핑
        if level_num > 0:
            levels = torch.linspace(0, 1, level_num, device=spike.device)  # 0에서 1까지 균등한 level_num 개의 값
            bucketized = torch.bucketize(spike_normalized.contiguous(), levels)
            bucketized[bucketized == 0] = 1 # bucketize 결과에서 0인 요소를 1로 변경
            spike_normalized = levels[bucketized - 1] # 최종 양자화된 값 매핑
    # plot_origin_spike(spike_normalized[0].cpu().detach().numpy(), min_max_y_on=True)
    return spike_normalized



def copy_weights(source_encoder, target_encoder):
    """
    Copy weights and bias from source encoder to target encoder.
    Matches layers based on compatible weight shapes.
    """
    source_layers = list(source_encoder.named_modules())
    target_layers = list(target_encoder.named_modules())

    # Iterate through source and target layers
    src_i = 0
    tgt_i = 0
    # for src_i in range(len(source_layers)):
        # for tgt_i in range(len(target_layers)):
    
    while(src_i < len(source_layers) and tgt_i < len(target_layers)):
        src_name, src_layer = source_layers[src_i]
        tgt_name, tgt_layer = target_layers[tgt_i]

        if isinstance(src_layer, (nn.Conv1d, nn.Linear, nn.ConvTranspose1d)):
            if isinstance(tgt_layer, (nn.Conv1d, nn.Linear, nn.ConvTranspose1d)):
                src_i += 1
                tgt_i += 1
                # print(f"Copy layer: {src_name}, {src_layer} -> {tgt_name}, {tgt_layer}")
            else:
                tgt_i += 1
                continue
        else:
            src_i += 1
            continue
        
        # Check if weight shapes match
        if src_layer.weight.shape == tgt_layer.weight.shape:
            tgt_layer.weight.data = src_layer.weight.data.clone()
            # print(f"Copied weights: {src_name}, {src_layer} -> {tgt_name}, {tgt_layer}")
        else:
            assert False, f"Weight shape mismatch: {src_name}, {src_layer} -> {tgt_name}, {tgt_layer}"
        
        # Copy bias if it exists in both layers
        if (
            src_layer.bias is not None and 
            tgt_layer.bias is not None and 
            src_layer.bias.shape == tgt_layer.bias.shape
        ):
            tgt_layer.bias.data = src_layer.bias.data.clone()
            # print(f"Copied bias: {src_name}, {src_layer} -> {tgt_name}, {tgt_layer}")


def map_and_load_weights(saved_state_dict, current_state_dict):
    """
    Maps weights from a saved state_dict to a target network's state_dict.
    
    Args:
        saved_state_dict (dict): State dictionary from the saved model.
        current_state_dict (dict): State dictionary of the current network.

    Returns:
        dict: Updated state_dict for the current network.
    """
    # Get keys from both state_dicts
    saved_keys = list(saved_state_dict.keys())
    current_keys = list(current_state_dict.keys())
    
    # Initialize layer indices
    src_i = 0
    tgt_i = 0

    # Iterate over saved and target state_dict keys to map weights
    while src_i < len(saved_keys) and tgt_i < len(current_keys):
        src_layer = saved_keys[src_i]
        tgt_layer = current_keys[tgt_i]
        if 'weight' in src_layer or 'bias' in src_layer:
            if 'weight' in tgt_layer or 'bias' in tgt_layer:
                # Map weights and biases
                print(f"Copy layer: {src_layer} {saved_state_dict[src_layer].shape} -> {tgt_layer} {current_state_dict[tgt_layer].shape}")
                current_state_dict[tgt_layer] = saved_state_dict[src_layer]
                src_i += 1
                tgt_i += 1
            else:
                # Skip unmatched target layers
                tgt_i += 1
        else:
            # Skip unmatched source layers
            src_i += 1

    return current_state_dict


# 1 epoch training하고 activation 읽어라.
def plot_activation_distribution(model):
    total_activation_values_only_encoder = []
    total_activation_values = []
    for idx, layer in enumerate(model.module.encoder):
        if str(layer) == 'SSBH_activation_collector()':
            print(str(layer))
            activations = np.concatenate([in_layer.cpu().detach().numpy() for in_layer in layer.activation], axis=0)
            activation_values = activations.flatten()
            total_activation_values.append(activation_values)
            total_activation_values_only_encoder.append(activation_values)
            # 최대값, 평균, 분산 계산
            max_val = activation_values.max()
            mean_val = activation_values.mean()
            var_val = activation_values.var()
            
            # 히스토그램 그리기
            plt.figure(figsize=(10, 6))
            plt.hist(activation_values, bins=100, alpha=0.7, label=f'Layer {idx}')
            plt.title(f'Layer {str(layer)} - Max: {max_val:.2f}, Mean: {mean_val:.2f}, Variance: {var_val:.2f}')
            plt.xlabel('Activation Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()
        else:
            print('skip',str(layer))
    
    for idx, layer in enumerate(model.module.decoder):
        if str(layer) == 'SSBH_activation_collector()':
            
            print(str(layer))
            activations = np.concatenate([in_layer.cpu().detach().numpy() for in_layer in layer.activation], axis=0)
            activation_values = activations.flatten()
            total_activation_values.append(activation_values)
            
            # 최대값, 평균, 분산 계산
            max_val = activation_values.max()
            mean_val = activation_values.mean()
            var_val = activation_values.var()
            
            # 히스토그램 그리기
            plt.figure(figsize=(10, 6))
            plt.hist(activation_values, bins=100, alpha=0.7, label=f'Layer {idx}')
            plt.title(f'Layer {str(layer)} - Max: {max_val:.2f}, Mean: {mean_val:.2f}, Variance: {var_val:.2f}')
            plt.xlabel('Activation Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()




    # Plot for total activation values in encoder layers
    total_activation_values_only_encoder = np.concatenate(total_activation_values_only_encoder, axis=0)
    percentile_list = [95, 99, 99.7, 99.9, 99.99]
    # Calculate percentiles
    percentiles = [np.percentile(total_activation_values_only_encoder, p) for p in percentile_list]

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(total_activation_values_only_encoder, bins=100, alpha=0.7, label='Total Encoder Activations')

    # Construct the title with percentile values
    percentile_text = ', '.join([f'{p}%: {percentiles[idx]:.2f}' for idx, p in enumerate(percentile_list)])
    plt.title(f'Total Encoder Activations - Max: {total_activation_values_only_encoder.max():.2f}, '
            f'Mean: {total_activation_values_only_encoder.mean():.2f}, '
            f'Variance: {total_activation_values_only_encoder.var():.2f}, '
            f'\nPercentile {percentile_text}')
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


    # # Plot for total activation values (both encoder and decoder)
    # total_activation_values = np.concatenate(total_activation_values, axis=0)
    # plt.figure(figsize=(10, 6))
    # plt.hist(total_activation_values, bins=100, alpha=0.7, label='Total Activations (Encoder + Decoder)')
    # plt.title(f'Total Activations - Max: {total_activation_values.max():.2f}, '
    #           f'Mean: {total_activation_values.mean():.2f}, '
    #           f'Variance: {total_activation_values.var():.2f}')
    # plt.xlabel('Activation Value')
    # plt.ylabel('Frequency')
    # plt.legend()
    # plt.show()


def l2_norm_loss(encoded_spike, target_norm=1.0):
    """
    각 feature 벡터의 L2 norm을 target_norm으로 맞추기 위한 Loss를 계산합니다.
    :param encoded_spike: [batch, feature] 차원의 텐서
    :param target_norm: 목표 L2 norm 크기 (default: 1.0)
    :return: L2Norm Loss
    """
    norms = encoded_spike.norm(p=2, dim=1)  # 각 feature 벡터의 L2 norm 계산
    loss = ((norms - target_norm) ** 2).mean()  # 목표 norm과의 차이 제곱합
    return loss

















########### dvs 데이터 시각화 코드#####################################################
########### dvs 데이터 시각화 코드#####################################################
########### dvs 데이터 시각화 코드#####################################################
# mapping = {0: 'Hand Clapping',1: 'Right Hand Wave',2: 'Left Hand Wave',3: 'Right Arm CW',4: 'Right Arm CCW',5: 'Left Arm CW',6: 'Left Arm CCW',7: 'Arm Roll',8: 'Air Drums',9: 'Air Guitar',10: 'Other'}
def dvs_visualization(inputs, labels, TIME, BATCH, my_seed):
    # if inputs.size(4)==128:
    #     print('\n\n 128x128 data 32x32로 maxpool 해서 보여줄게 \n\n')
    #     timestep, batch_size, in_c, h, w = inputs.shape
    #     maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    #     inputs = maxpool(inputs.reshape(timestep*batch_size, in_c, h, w))
    #     inputs = maxpool(inputs).reshape(timestep, batch_size, in_c, h//4, w//4)
    classes = [
        "hand_clapping",
        "right_hand_wave",
        "left_hand_wave",
        "right_arm_clockwise",
        "right_arm_counter_clockwise",
        "left_arm_clockwise",
        "left_arm_counter_clockwise",
        "arm_roll",
        "air_drums",
        "air_guitar",
        "other_gestures",
    ]
    seed_assign(seed = my_seed)
    what_input = random.randint(0, BATCH - 1)
    print(f'input: {inputs.shape}')
    inputs_for_view = inputs.permute(1, 0, 2, 3, 4)
    print(f'inputs_for_view: {inputs_for_view.shape}')
    for i in range(TIME):
        # 예시 데이터 생성
        data1 = inputs_for_view[what_input][i][0].cpu().numpy()  # torch tensor를 numpy 배열로 변환
        if inputs_for_view.shape[2] == 1:
            data2 = inputs_for_view[what_input][i][0].cpu().numpy()  # torch tensor를 numpy 배열로 변환
        else:
            data2 = inputs_for_view[what_input][i][1].cpu().numpy()  # torch tensor를 numpy 배열로 변환

        # # merge면 data1에 data2를 더하기 ##########
        # data1 = data1+data2
        # data1[data1 > 0] = 1
        # ##########################################

        # 데이터 플로팅
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1행 2열의 subplot 생성

        # 첫 번째 subplot에 데이터1 플로팅
        im1 = axs[0].imshow(data1, cmap='viridis', interpolation='nearest')
        axs[0].set_title(f'Channel 0\nLabel: {labels[what_input]} {classes[labels[what_input]]}  Time: {i}')  # 라벨값 맵핑하여 제목에 추가
        axs[0].set_xlabel('X axis')
        axs[0].set_ylabel('Y axis')
        axs[0].grid(False)
        fig.colorbar(im1, ax=axs[0])  # Color bar 추가

        # 두 번째 subplot에 데이터2 플로팅
        im2 = axs[1].imshow(data2, cmap='viridis', interpolation='nearest')
        axs[1].set_title(f'Channel 1\nLabel: {labels[what_input]} {classes[labels[what_input]]}  Time: {i}')  # 라벨값 맵핑하여 제목에 추가
        axs[1].set_xlabel('X axis')
        axs[1].set_ylabel('Y axis')
        axs[1].grid(False)
        fig.colorbar(im2, ax=axs[1])  # Color bar 추가

        plt.tight_layout()  # subplot 간 간격 조정
        plt.show()
    # sys.exit("정상 종료")

########### dvs 데이터 시각화 코드#####################################################
########### dvs 데이터 시각화 코드#####################################################
########### dvs 데이터 시각화 코드#####################################################





########### rated 데이터 시각화 코드#####################################################
# mapping = {0: 'Hand Clapping',1: 'Right Hand Wave',2: 'Left Hand Wave',3: 'Right Arm CW',4: 'Right Arm CCW',5: 'Left Arm CW',6: 'Left Arm CCW',7: 'Arm Roll',8: 'Air Drums',9: 'Air Guitar',10: 'Other'}
def rate_coded_visualization(inputs, labels, TIME, BATCH, my_seed):
    seed_assign(seed = my_seed)
    what_input = random.randint(0, BATCH - 1)
    print(f'input: {inputs.shape}')
    inputs_for_view = inputs.permute(1, 0, 2, 3, 4)
    print(f'inputs_for_view: {inputs_for_view.shape}')
    for i in range(TIME):
        # 예시 데이터 생성
        data1 = inputs_for_view[what_input][i][0].cpu().numpy()  # torch tensor를 numpy 배열로 변환

        # 데이터 플로팅
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1행 2열의 subplot 생성

        # 첫 번째 subplot에 데이터1 플로팅
        im1 = axs[0].imshow(data1, cmap='viridis', interpolation='nearest')
        axs[0].set_title(f'Channel 0\nLabel: {labels[what_input]}  Time: {i}')  # 라벨값 맵핑하여 제목에 추가
        axs[0].set_xlabel('X axis')
        axs[0].set_ylabel('Y axis')
        axs[0].grid(False)
        fig.colorbar(im1, ax=axs[0])  # Color bar 추가

        plt.tight_layout()  # subplot 간 간격 조정
        plt.show()
    # sys.exit("정상 종료")





import numpy as np
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
from tslearn.clustering import TimeSeriesKMeans
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from kmodes.kmodes import KModes

def cluster_accuracy(y_true, y_pred):
    """
    Hungarian algorithm을 사용하여 클러스터링 결과의 최적 매칭 후 accuracy 계산
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Confusion Matrix 생성
    D = max(y_pred.max(), y_true.max()) + 1  # 클러스터 개수 결정
    cost_matrix = np.zeros((D, D))
    
    for i in range(D):
        for j in range(D):
            cost_matrix[i, j] = np.sum((y_pred == i) & (y_true == j))
    
    # Hungarian 알고리즘 적용
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)  # Hungarian 알고리즘 (최대화 문제)
    
    # 최적 매핑을 이용하여 y_pred 수정
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    y_pred_mapped = np.array([mapping[label] for label in y_pred])

    return accuracy_score(y_true, y_pred_mapped)

def evaluate_clustering_accuracy(data, true_labels, n_clusters=3):
    """
    입력 데이터와 라벨을 받아서 3가지 클러스터링 수행 후 accuracy 반환
    """
    n_samples, timesteps, features = data.shape
    
    ##############################################
    # Approach 1: 시계열 DTW 기반 클러스터링
    ##############################################
    dtw_model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=0)
    labels_dtw = dtw_model.fit_predict(data)

    # ##############################################
    # # Approach 2: 평탄화 후 Hamming 거리 기반 계층적 클러스터링
    # ##############################################
    # data_flat = data.reshape(n_samples, -1)  # 50x4 = 200차원 벡터로 평탄화
    # distance_matrix = squareform(pdist(data_flat, metric="hamming"))  # Hamming 거리 계산
    # agg_cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage="average")
    # labels_hamming = agg_cluster.fit_predict(distance_matrix)

    # ##############################################
    # # Approach 3: k-modes 클러스터링 (범주형/이진 데이터에 적합)
    # ##############################################
    # km = KModes(n_clusters=n_clusters, init='Huang', n_init=5, verbose=0)
    # labels_kmodes = km.fit_predict(data_flat)

    # # 정확도 계산
    # acc_dtw = cluster_accuracy(true_labels, labels_dtw)
    # acc_hamming = cluster_accuracy(true_labels, labels_hamming)
    # acc_kmodes = cluster_accuracy(true_labels, labels_kmodes)

    # 정확도 계산
    acc_dtw = cluster_accuracy(true_labels, labels_dtw)
    acc_hamming = 0
    acc_kmodes = 0

    return {
        "DTW Clustering Accuracy": acc_dtw,
        "Hamming Clustering Accuracy": acc_hamming,
        "K-Modes Clustering Accuracy": acc_kmodes
    }


def plot_tsne(dataname, kmeans_accuracy, spike_hidden, label, n_components=2, perplexity=30, random_state=42):
    """
    t-SNE를 사용하여 spike_hidden 데이터를 시각화하는 함수.

    Parameters:
        spike_hidden (numpy.ndarray): (N, D) 형태의 입력 데이터.
        label (numpy.ndarray): (N,) 형태의 레이블 데이터.
        n_components (int): t-SNE의 출력 차원 (기본값: 2).
        perplexity (float): t-SNE의 perplexity 값 (기본값: 30).

    Returns:
        None
    """
    # t-SNE 변환 수행
    tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity)
    spike_hidden_tsne = tsne.fit_transform(spike_hidden)

    # 시각화
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(spike_hidden_tsne[:, 0], spike_hidden_tsne[:, 1], c=label, cmap='jet', alpha=0.7)
    plt.colorbar(scatter, label='Label')
    # plt.title(f"Dataset: {dataname[9:]}\nAccuracy: {kmeans_accuracy*100:.2f}%", fontsize=20)
    plt.title(f"Dataset: {dataname}\nAccuracy: {kmeans_accuracy*100:.2f}%", fontsize=20)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.savefig(f'/home/bhkim003/github_folder/ByeonghyeonKim/my_snn/picture/{dataname}.png', dpi=300, bbox_inches='tight')
    plt.show()
