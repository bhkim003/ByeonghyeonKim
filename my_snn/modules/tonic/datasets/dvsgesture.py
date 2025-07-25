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
        self.time_slice_random_cropping_flag = time_slice_random_cropping_flag

        file_path = os.path.join(self.location_on_system, self.folder_name)
        data_num = 0
        for path, dirs, files in os.walk(file_path):
            dirs.sort()
            for file in files:
                if file.endswith("npy"):
                    this_file_max_time = np.load(path + "/" + file)[-1][3] * 1000 # us단위
                    if self.crop_max_time >= this_file_max_time:  # 시간 길이 짧으면 스킵
                        if self.time_slice_random_cropping_flag == False:
                            continue
                    if self.exclude_class == True and int(file[:-4]) == 10: ## other class 제거
                        continue
                    data_num += 1
                    self.data.append(path + "/" + file)
                    self.targets.append(int(file[:-4]))
        print("이 데이터셋의 데이터 개수는", data_num, "입니다. (test set은 안바뀌게 해놨다 알제)")
        self.clipping = clipping
        self.time = time

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
            
        if self.time_slice_random_cropping_flag == True:
            events = events[4:]

        T, *spatial_dims = events.shape
        if T >= self.time:
            if self.time_slice_random_cropping_flag == True:
                events = events
            else:
                events = events[:self.time]
        else:
            
            # zero padding
            temporal_filter = 5
            T_remainder = T % temporal_filter
            # print(f'T_remainder: {T_remainder}, events.shape: {events.shape}') # T_remainder: 2, events.shape: (67, 2, 14, 14)
            events = np.concatenate([events[:-T_remainder], np.zeros((self.time - T + T_remainder, *spatial_dims), dtype=events.dtype)], axis=0)
            # print(f'T_remainder: {T_remainder}, events.shape: {events.shape}') # T_remainder: 2, events.shape: (80, 2, 14, 14)
            
            # 걍랜덤딴거갖고가 하는 코드
            # return self.__getitem__(random.randint(0, len(self.data) - 1))

        
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
