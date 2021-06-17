import os
import numpy as np
import torch

from util.datasets import TrajectoryDataset
from .label_functions import label_functions_list
from .augmentations import augmentation_list
from util.logging import LogEntry
from .preprocess import *
import pickle


ROOT_DIR = 'util/datasets/mouse_v1/data'

# CalMs21 dataset unlabeled set.
TRAIN_FILE = 'calms21_unlabeled_train.npz'
TEST_FILE = 'calms21_unlabeled_val.npz'


class MouseV1Dataset(TrajectoryDataset):

    name = 'mouse_v1'
    all_label_functions = label_functions_list
    all_augmentations = augmentation_list

    # Default config
    _seq_len = 21
    _state_dim = 28
    _action_dim = 28
    _svd_computer = None
    _svd_mean = None

    normalize_data = True
    compute_svd = False

    test_name = TEST_FILE

    def __init__(self, data_config):
        super().__init__(data_config)

    def _load_data(self):
        # Process configs
        if 'normalize_data' in self.config:
            self.normalize_data = self.config['normalize_data']
        if 'compute_svd' in self.config:
            assert isinstance(self.config['compute_svd'], int)
            self.compute_svd = self.config['compute_svd']
        if 'test_name' in self.config:
            self.test_name = self.config['test_name']

        self.keypoints = []
        if 'keypoints' in self.config:
            assert isinstance(self.config['keypoints'], list)
            resi_start = [2 * k for k in self.config['keypoints']]
            resi_end = [k + 1 for k in resi_start]
            intr_start = [14 + k for k in resi_start]
            intr_end = [k + 1 for k in intr_start]
            self.keypoints = resi_start + resi_end + intr_start + intr_end
            self.keypoints.sort()


        if 'labels' in self.config:
            for lf_config in self.config['labels']:
                lf_config['data_normalized'] = self.normalize_data

        self.log = LogEntry()


        try:
            with open(svd_computer_path, 'rb') as f:
                self._svd_computer = pickle.load(f)
            with open(mean_path, 'rb') as f:        
                self._mean = pickle.load(f)
        except:
            pass

        self.train_states, self.train_actions = self._load_and_preprocess(
            train=True)
        self.test_states, self.test_actions = self._load_and_preprocess(
            train=False)


    def _load_and_preprocess(self, train):
        path = os.path.join(ROOT_DIR, TRAIN_FILE if train else self.test_name)
        file = np.load(path, allow_pickle = True)
        data = file['data']
        if len(data) == 2:
            data = data[0]
 
        # Subsample timesteps
        data = data[:, ::self.subsample]

        # Normalize data
        if self.normalize_data:
            data = normalize(data)

        # Select only certain keypoints
        if len(self.keypoints) > 0:
            data = data[:, :, self.keypoints]

        # Compute SVD on train data and apply to train and test data
        if self.compute_svd:
            seq_len = data.shape[1]

            data = data.reshape((-1, 2, 7, 2))
            # Input [seq_num x seq_len, 2, 7, 2]
            if train and self._svd_computer is None:
                data_svd, self._svd_computer, self._svd_mean = transform_to_svd_components(
                    data, center_index=3, n_components=self.compute_svd,
                    svd_computer=self._svd_computer, mean=self._svd_mean,
                    stack_agents = True, save_svd = True)
            else:
                data_svd, _, _ = transform_to_svd_components(
                    data, center_index=3, n_components=self.compute_svd,
                    svd_computer=self._svd_computer, mean=self._svd_mean,
                    stack_agents = True)
            # Output [seq_num x seq_len, 2, 4 + n_components]

            data = data_svd.reshape((-1, seq_len, data_svd.shape[-1] * 2))

            states = data
            actions = states[:, 1:] - states[:, :-1]

        else:
            states = data
            actions = states[:, 1:] - states[:, :-1]

        # Update dimensions
        self._seq_len = actions.shape[1]
        self._state_dim = states.shape[-1]
        self._action_dim = actions.shape[-1]

        return torch.Tensor(states), torch.Tensor(actions)

