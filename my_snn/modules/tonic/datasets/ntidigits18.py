#!/user/bin/env python

import numpy as np
import h5py
import os
from typing import Callable, Optional

from tonic.dataset import Dataset
from tonic.io import make_structured_array
import requests
from tqdm import tqdm
import random
from collections import defaultdict

class NTIDIGITS18(Dataset):
    """`N-TIDIGITS18 Dataset <https://docs.google.com/document/d/1Uxe7GsKKXcy6SlDUX4hoJVAC0-UkH-8kr5UXp0Ndi1M/edit?tab=t.0#heading=h.sbnu5gtazqjq/>`_
    Cochlea Spike Dataset.
    ::

        @article{anumula2018feature,
          title={Feature representations for neuromorphic audio spike streams},
          author={Anumula, Jithendar and Neil, Daniel and Delbruck, Tobi and Liu, Shih-Chii},
          journal={Frontiers in neuroscience},
          volume={12},
          pages={23},
          year={2018},
          publisher={Frontiers Media SA}
        }

    Parameters:
        save_to (string): Location to save files to on disk. Will put files in an 'hsd' subfolder.
        train (bool): If True, uses training subset, otherwise testing subset.
        single_digits (bool): If True, only returns samples with single digits (o, 1, 2, 3, 4, 5, 6, 7, 8, 9, z), with class 0 for 'o' and 11 for 'z'.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.

    Returns:
        A dataset object that can be indexed or iterated over. One sample returns a tuple of (events, targets).
    """

    # This paper introduces the N-TIDIGITS18 dataset by playing the audio files 
    # from the TIDIGITS dataset to the CochleaAMS1b sensor. The dataset is publicly 
    # accessible at http://sensors.ini.uzh. ch/databases.html. The dataset includes 
    # both single digits and connected digit sequences, with a vocabulary consisting 
    # of 11 digits (“oh,” “zero” and the digits “1” to “9”). Each digit sequence is of 
    # length 1–7 spoken digits. There is a total of 55 male and 56 female 
    # speakers in the training set with a total of 8,623 training samples, 
    # while the testing set has a total of 56 male and 53 female speakers 
    # with a total of 8,700 testing samples. The entire dataset is used or a reduced 
    # version of the dataset is used where only the single digit samples are used for 
    # training and testing. In the single digits dataset, there are two samples for each 
    # of the 11 single digits from every speaker, with a total of 2,464 training samples 
    # and 2,486 testing samples. The NTIDIGITS18 dataset with all the samples was used to
    #   train a sequence classification task while the digit samples were used to train a 
    #   digit recognition task. For most of our training, unless specified, we only use 
    #   events from one ear and one neuron.

    base_url = "https://www.dropbox.com/scl/fi/1x4lxt9yyw25sc3tez8oi/n-tidigits.hdf5?e=2&rlkey=w8gi5udvib2zqzosusa5tr3wq&dl=1"
    filename = "n-tidigits.hdf5"
    file_md5 = "360a2d11e5429555c9197381cf6b58e0"
    folder_name = ""

    sensor_size = (64, 1, 1)
    dtype = np.dtype([("t", int), ("x", int), ("p", int)])
    ordering = dtype.names

    # class_map = {"o": 0,
    #              "1": 1,
    #              "2": 2,
    #              "3": 3,
    #              "4": 4,
    #              "5": 5,
    #              "6": 6,
    #              "7": 7,
    #              "8": 8,
    #              "9": 9,
    #              "z": 10}
    
    class_map = {"o": 10,
                 "1": 1,
                 "2": 2,
                 "3": 3,
                 "4": 4,
                 "5": 5,
                 "6": 6,
                 "7": 7,
                 "8": 8,
                 "9": 9,
                 "z": 0}

    def __init__(
            self,
            save_to: str,
            train: bool = True,
            single_digits=False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            target_word: Optional[int] = None,
            clipping: Optional[int] = None,
    ):
        super().__init__(
            save_to,
            transform=transform,
            target_transform=target_transform,
        )

        self.url = self.base_url
        # load the data
        self.file_path = os.path.join(self.location_on_system, self.filename)

        if not self._check_exists():
            self.download()

        self.data = h5py.File(self.file_path, 'r')
        self.partition = "train" if train else "test"
        self.single_indices = [i for i in range(len(self.data[f"{self.partition}_labels"])) if
                               len(self.data[f"{self.partition}_labels"][i].decode().split("-")[-1]) == 1]
        self._samples = [x.decode() for x in self.data[f"{self.partition}_labels"]]
        self.single_digits = single_digits

        self.target_word = target_word
        print(f'\n\n\nTarget word: {self.target_word}\n\n\n')
        # self.target_word = str(target_word)

        self.clipping = clipping

        if single_digits:
            # 일단 모든 single digit 중에서 class_map이 0~9인 것만 필터링
            all_single_samples = [self._samples[i] for i in self.single_indices
                                if self.class_map[self._samples[i].split("-")[-1]] in range(10)]
            if self.target_word is not None:
                if not train:
                # if True:
                    # # target_word와 일치하는 샘플들
                    # positive_samples = [s for s in all_single_samples if self.class_map[s.split("-")[-1]] == self.target_word]
                    # # target_word와 일치하지 않는 샘플들
                    # negative_samples = [s for s in all_single_samples if self.class_map[s.split("-")[-1]] != self.target_word]
                    # # negative_samples = [s for s in all_single_samples if self.class_map[s.split("-")[-1]] == self.target_word+1]
                    # # 양쪽에서 min 수만큼 뽑아 균형 맞추기
                    # min_len = min(len(positive_samples), len(negative_samples))
                    # # 랜덤 셔플 후 절반씩 선택
                    # random.shuffle(positive_samples)
                    # random.shuffle(negative_samples)
                    # # print(f'positive_samples: {positive_samples}, \nnegative_samples: {negative_samples}')
                    # balanced_samples = positive_samples[:min_len] + negative_samples[:min_len]
                    # random.shuffle(balanced_samples)  # 전체 셔플
                    # self._samples = balanced_samples



                    # 1. Positive 샘플 추출
                    positive_samples = [s for s in all_single_samples if self.class_map[s.split("-")[-1]] == self.target_word]
                    N_pos = len(positive_samples)

                    # 2. Negative 클래스만 추출하여 클래스별 그룹화
                    neg_class_samples = defaultdict(list)
                    for s in all_single_samples:
                        label = self.class_map[s.split("-")[-1]]
                        if label != self.target_word:
                            neg_class_samples[label].append(s)

                    # 3. 9개 negative 클래스에 대해 분배할 개수 미리 계산
                    neg_classes = sorted(neg_class_samples.keys())  # 항상 같은 순서로
                    N_per_class = [N_pos // 9] * 9
                    for i in range(N_pos % 9):
                        N_per_class[i] += 1  # 앞에서부터 1개씩 더해줌

                    # 4. 각 클래스에서 해당 수만큼 샘플 가져오기
                    balanced_negatives = []
                    for label, n in zip(neg_classes, N_per_class):
                        samples = neg_class_samples[label]
                        random.shuffle(samples)
                        balanced_negatives += samples[:n]

                    # 5. 전체 셔플 및 합치기
                    random.shuffle(positive_samples)
                    random.shuffle(balanced_negatives)

                    assert len(positive_samples) == len(balanced_negatives), f'Positive and negative sample counts do not match: {len(positive_samples)} vs {len(balanced_negatives)}'
                    balanced_samples = positive_samples + balanced_negatives
                    random.shuffle(balanced_samples)

                    self._samples = balanced_samples






                else: 
                    # target_word와 일치하는 샘플들
                    positive_samples = [s for s in all_single_samples if self.class_map[s.split("-")[-1]] == self.target_word]
                    # target_word와 일치하지 않는 샘플들
                    negative_samples = [s for s in all_single_samples if self.class_map[s.split("-")[-1]] != self.target_word]

                    # 양쪽에서 max 수만큼 뽑아 균형 맞추기
                    max_len = max(len(positive_samples), len(negative_samples))

                    # 부족한 쪽은 중복 샘플링하여 채우기
                    if len(positive_samples) < max_len:
                        positive_samples += random.choices(positive_samples, k=max_len - len(positive_samples))
                    else:
                        positive_samples = positive_samples[:max_len]

                    if len(negative_samples) < max_len:
                        negative_samples += random.choices(negative_samples, k=max_len - len(negative_samples))
                    else:
                        negative_samples = negative_samples[:max_len]

                    # 랜덤 셔플
                    random.shuffle(positive_samples)
                    random.shuffle(negative_samples)

                    # 합치고 전체 셔플
                    balanced_samples = positive_samples + negative_samples
                    random.shuffle(balanced_samples)

                    self._samples = balanced_samples

                    # self._samples = all_single_samples


            else:
                # target_word가 없으면 전체 사용
                self._samples = all_single_samples
            

        self.labels = [x.decode().split("-")[-1] for x in self.data[f"{self.partition}_labels"]]

    def download(self) -> None:
        response = requests.get(self.base_url, stream=True)
        if response.status_code == 200:
            print("Downloading N-TIDIGITS from Dropbox at {}...".format(self.base_url))
            file_size = int(response.headers.get('Content-Length', 0))  # get total file size in bytes
            chunk_size = 8192

            os.makedirs(self.location_on_system, exist_ok=True)
            # Initialize progress bar
            with open(os.path.join(self.location_on_system, self.filename), 'wb') as f, tqdm(
                    total=file_size,
                    unit='B',
                    unit_scale=True,
                    desc="Downloading",
                    ascii=True
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        pbar.update(len(chunk))
        else:
            print("Failed to download N-TIDIGITS from Dropbox. Please try again later.")
            response.raise_for_status()

    def __getitem__(self, index):
        sample_id = self._samples[index]
        x = np.asarray(self.data[f"{self.partition}_addresses"][sample_id])
        t = np.asarray(self.data[f"{self.partition}_timestamps"][sample_id])
        events = make_structured_array(
            t * 1e6,
            x,
            1,
            dtype=self.dtype,
        )

        # scale_factor = 840_000 / events["t"].max() 
        # events["t"] = (events["t"] * scale_factor).astype(int)



        target = sample_id.split("-")[-1]
        if self.single_digits:
            assert len(target) == 1, "Single digit samples requested, but target is not single digit."
            target = self.class_map[target]
            if self.target_word is not None:
                target = 0 if target == self.target_word else 1
                # target = 1 if target == self.target_word else 0
                # target = 1


        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # print('hi2', events.shape)
        
        if self.clipping != 0:
            events[events<self.clipping] = 0.0
            events[events>=self.clipping] = 1.0

        return events, target
    
    def __len__(self):
        return len(self._samples)


    def _check_exists(self):
        return (
            self._is_file_present()
        )