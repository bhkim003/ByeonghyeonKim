import os
from typing import Callable, Optional
import numpy as np
import torch
from tonic.dataset import Dataset
import random


class DVSGesture(Dataset):
    """`IBM DVS Gestures <http://research.ibm.com/dvsgesture/>`_

    .. note::  This is (exceptionally) a preprocessed version of the original dataset,
               where recordings that originally contained multiple labels have already
               been cut into respective samples. Also temporal precision is reduced to ms.

    ::

        @inproceedings{amir2017low,
          title={A low power, fully event-based gesture recognition system},
          author={Amir, Arnon and Taba, Brian and Berg, David and Melano, Timothy and McKinstry, Jeffrey and Di Nolfo, Carmelo and Nayak, Tapan and Andreopoulos, Alexander and Garreau, Guillaume and Mendoza, Marcela and others},
          booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
          pages={7243--7252},
          year={2017}
        }

    Parameters:
        save_to (string): Location to save files to on disk.
        train (bool): If True, uses training subset, otherwise testing subset.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        transforms (callable, optional): A callable of transforms that is applied to both data and
                                         labels at the same time.
    """

    test_url = "https://figshare.com/ndownloader/files/38020584"
    train_url = "https://figshare.com/ndownloader/files/38022171"
    test_md5 = "56070e45dadaa85fff82e0fbfbc06de5"
    train_md5 = "3a8f0d4120a166bac7591f77409cb105"
    test_filename = "ibmGestureTest.tar.gz"
    train_filename = "ibmGestureTrain.tar.gz"
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

    sensor_size = (128, 128, 2)
    dtype = np.dtype([("x", np.int16), ("y", np.int16), ("p", bool), ("t", np.int64)])
    ordering = dtype.names

    def __init__(
        self,
        save_to: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        clipping: int = 1,
        time: int = 8,
        exclude_class: bool = False,
        crop_max_time: int = 600_000,
        time_slice_random_cropping_flag: bool = False,
    ):
        super().__init__(
            save_to,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        self.train = train
        if train:
            self.url = self.train_url
            self.file_md5 = self.train_md5
            self.filename = self.train_filename
            self.folder_name = "ibmGestureTrain"
        else:
            self.url = self.test_url
            self.file_md5 = self.test_md5
            self.filename = self.test_filename
            self.folder_name = "ibmGestureTest"

        if not self._check_exists():
            self.download()

        self.exclude_class = exclude_class
        self.crop_max_time = crop_max_time

        file_path = os.path.join(self.location_on_system, self.folder_name)
        data_num = 0
        for path, dirs, files in os.walk(file_path):
            dirs.sort()
            for file in files:
                if file.endswith("npy"):
                    this_file_max_time = np.load(path + "/" + file)[-1][3] * 1000 # us단위
                    if self.crop_max_time >= this_file_max_time:  # 시간 길이 짧으면 스킵
                        continue
                    if self.exclude_class == True and int(file[:-4]) == 10: ## other class 제거
                        continue
                    data_num += 1
                    self.data.append(path + "/" + file)
                    self.targets.append(int(file[:-4]))
        print("이 데이터셋의 데이터 개수는", data_num, "입니다. (test set은 안바뀌게 해놨다 알제)")
        self.clipping = clipping
        self.time = time
        self.time_slice_random_cropping_flag = time_slice_random_cropping_flag

        # minimum_length = 9999999999999
        # print('data개수', len(self.data))
        # for i in range(len(self.data)):
        #     x = np.load(self.data[i])
        #     time = x[-1, 3]
        #     if time < minimum_length:
        #         minimum_length = time
        # print('minimum_length', minimum_length)

        # data개수 979
        # minimum_length 2456.344
        # data개수 240
        # minimum_length 1798.364


        # last_spike_time_box = []
        # for i in range(len(self.data)):
        #     last_spike_time_box.append(np.load(self.data[i])[-1][-1]*0.001)

        # # 분포 히스토그램 그리기
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 6))
        # plt.hist(last_spike_time_box, bins=20, color='skyblue', edgecolor='black')
        # plt.title(f"Distribution of Final Spike Timesteps (Total samples: {len(self.data)})")
        # plt.xlabel("Final Timestep")
        # plt.ylabel("Frequency")

        # # max, min, 평균 계산 및 timestep 단위 환산
        # max_val = np.max(last_spike_time_box)
        # min_val = np.min(last_spike_time_box)
        # mean_val = np.mean(last_spike_time_box)

        # # 0.025로 나눈 값
        # max_ts = max_val/ 0.025
        # min_ts = min_val/ 0.025
        # mean_ts = mean_val/ 0.025

        # # 통계선 표시
        # plt.axvline(max_val, color='red', linestyle='dashed', linewidth=1.5,
        #             label=f"Max: {max_val} ({max_ts:.2f} timesteps)")
        # plt.axvline(min_val, color='green', linestyle='dashed', linewidth=1.5,
        #             label=f"Min: {min_val} ({min_ts:.2f} timesteps)")
        # plt.axvline(mean_val, color='orange', linestyle='dashed', linewidth=1.5,
        #             label=f"Mean: {mean_val:.2f} ({mean_ts:.2f} timesteps)")

        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        events = np.load(self.data[index])
        events[:, 3] *= 1000  # convert from ms to us
        events = np.lib.recfunctions.unstructured_to_structured(events, self.dtype)
        target = self.targets[index]
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transforms is not None:
            events, target = self.transforms(events, target)

        ## BH code ###############################################
        T, *spatial_dims = events.shape
        if T >= self.time:
            if self.time_slice_random_cropping_flag == True:
                # start_idx = random.randint(0, T - self.time)
                # start_idx = random.choice([i for i in range(0, T - self.time + 1, self.time)])
                # events = events[start_idx : start_idx + self.time]
                # 걍 밖에서 찦자

                events = events
            else:
                events = events[:self.time]
        else:
            assert False, f'self.time: {self.time}, T: {T}, events.shape: {events.shape}, index: {index}, self.data[index]: {self.data[index]}'
            return self.__getitem__(random.randint(0, len(self.data) - 1))
            
            # pad_shape = (self.time - T, *spatial_dims)
            # padding = np.zeros(pad_shape, dtype=events.dtype)
            # events = np.concatenate([events, padding], axis=0)

        if self.clipping != 0:
            events[events<self.clipping] = 0.0
            events[events>=self.clipping] = 1.0
        events = torch.tensor(events, dtype=torch.float32)
        ## BH code ###############################################
            
        return events, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return (
            self._is_file_present()
            and self._folder_contains_at_least_n_files_of_type(100, ".npy")
        )
