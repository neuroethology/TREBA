import torch
import numpy as np
import pickle
from scipy.ndimage import gaussian_filter1d

from util.datasets import Augmentations
from util.datasets.mouse_v1.preprocess import *

FRAME_WIDTH_TOP = 1024
FRAME_HEIGHT_TOP = 570


class AllRandomAugmentations(Augmentations):

    name = 'all_random'

    svd_computer = None
    mean = None

    def __init__(self, aug_config):
        super().__init__(aug_config)
        with open(svd_computer_path, 'rb') as f:
            self.svd_computer = pickle.load(f)
        with open(mean_path, 'rb') as f:        
            self.mean = pickle.load(f)

    def augment_func(self, states, actions):
        keypoints = transform_svd_to_keypoints(states.cpu().numpy(), self.svd_computer, self.mean,
            stack_agents = True)
        keypoints = unnormalize(keypoints)

        seq_len = states.shape[0]
        keypoints = keypoints.reshape((2, seq_len, 7, 2))

        new_keypoints = keypoints
        new_keypoints = add_rotation_translation(new_keypoints)  

        # Add random reflection and noise augmentations based on chance.
        add_horizontal_reflection = np.random.uniform(0, 1)
        if add_horizontal_reflection > 0.5:
            new_keypoints = reflect_points(new_keypoints, 0, 1, -FRAME_HEIGHT_TOP//2)  

        add_vertical_reflection = np.random.uniform(0, 1)
        if add_vertical_reflection > 0.5:
            new_keypoints = reflect_points(new_keypoints, 1, 0, -FRAME_WIDTH_TOP//2) 
  
        add_noise_shift = np.random.uniform(0, 1)
        if add_noise_shift > 0.75:
            new_keypoints = add_gaussian_noise(new_keypoints)
            new_keypoints = add_relative_shift(new_keypoints)       

        # 2, seq_len, 7, 2
        new_keypoints = new_keypoints.transpose((1,0,2,3)).reshape((-1, 2, 14))
        new_keypoints = normalize(new_keypoints)
        # reshape
        new_keypoints = new_keypoints.reshape((seq_len, 2, 7, 2))
        new_states, _, _ = transform_to_svd_components(
            new_keypoints, center_index=3, n_components=self.svd_computer.n_components,
            svd_computer=self.svd_computer, mean=self.mean)

        new_states = torch.from_numpy(new_states).to(states.device).float()
        return new_states, new_states[1:] - new_states[:-1]



    def plot(self, ax, states):

        return ax                

def add_gaussian_noise(points, mu = 0, sigma = 2):
    # Add Gaussian noise
    noise = np.random.normal(mu, sigma, points.shape)
    return points + noise


def add_relative_shift(points, shift_range = 20):
    # Add small shift between mice
    rand_ind = np.random.randint(2)


    max_horizontal_shift = min(np.amax(FRAME_WIDTH_TOP - points[rand_ind, :, :, 0]), shift_range)
    min_horizontal_shift = min(np.amin(points[rand_ind, :, :, 0]), shift_range)
    max_vertical_shift = min(np.amax(FRAME_HEIGHT_TOP - points[rand_ind, :, :, 1]), shift_range)
    min_vertical_shift = min(np.amin(points[rand_ind, :, :, 1]), shift_range)

    horizontal_shift = np.random.uniform(low = -1*min_horizontal_shift, high = max_horizontal_shift)
    vertical_shift = np.random.uniform(low = -1*min_vertical_shift, high = max_vertical_shift)

    points[rand_ind, :, :, 0] = points[rand_ind, :, :, 0] + horizontal_shift
    points[rand_ind, :, :, 1] = points[rand_ind, :, :, 1] + vertical_shift

    return points

def add_rotation_translation(points, rotation_range = np.pi, translation_range = 500):

    original = points.copy()

    image_center = [FRAME_WIDTH_TOP/2, FRAME_HEIGHT_TOP/2]    

    mouse_rotation = np.repeat(np.random.uniform(low = -1*rotation_range, high = rotation_range), points.shape[1])
    R = (np.array([[np.cos(mouse_rotation), -np.sin(mouse_rotation)],
                   [np.sin(mouse_rotation),  np.cos(mouse_rotation)]]).transpose((2, 0, 1)))
    
    points[0] = np.matmul(R, (points[0] - image_center).transpose(0, 2, 1)).transpose(0,2,1) + image_center
    points[1] = np.matmul(R, (points[1] - image_center).transpose(0, 2, 1)).transpose(0,2,1) + image_center

    # Check if possible for trajectory to fit within borders
    bounded = ((np.amax(points[:, :, :, 0]) - np.amin(points[:, :, :, 0])) < FRAME_WIDTH_TOP) and ((np.amax(points[:, :, :, 1]) - np.amin(points[:, :, :, 1])) < FRAME_HEIGHT_TOP)

    if bounded:
        # Shift all points to within borders first
        horizontal_shift = np.amax(points[:, :, :, 0] - FRAME_WIDTH_TOP)
        horizontal_shift_2 = np.amin(points[:, :, :, 0])
        if horizontal_shift > 0:
            points[:, :, :, 0] = points[:, :, :, 0] - horizontal_shift
        if horizontal_shift_2 < 0:
            points[:, :, :, 0] = points[:, :, :, 0] - horizontal_shift_2
       
        vertical_shift = np.amax(points[:, :, :, 1] - FRAME_HEIGHT_TOP)
        vertical_shift_2 = np.amin(points[:, :, :, 1])
        if vertical_shift > 0:
            points[:, :, :, 1] = points[:, :, :, 1] - vertical_shift
        if vertical_shift_2 < 0:
            points[:, :, :, 1] = points[:, :, :, 1] - vertical_shift_2
       

        max_horizontal_shift = np.amin(FRAME_WIDTH_TOP - points[:, :, :, 0])
        min_horizontal_shift = np.amin(points[:, :, :, 0])
        max_vertical_shift = np.amin(FRAME_HEIGHT_TOP - points[:, :, :, 1])
        min_vertical_shift = np.amin(points[:, :, :, 1])
        horizontal_shift = np.random.uniform(low = -1*min_horizontal_shift, high = max_horizontal_shift)
        vertical_shift = np.random.uniform(low = -1*min_vertical_shift, high = max_vertical_shift)

        points[:, :, :, 0] = points[:, :, :, 0] + horizontal_shift
        points[:, :, :, 1] = points[:, :, :, 1] + vertical_shift

        return points
    else:
        return original

def reflect_points(points, A, B, C):
    # A * x + B * y + C = 0
    new_points = np.zeros(points.shape)

    M = np.sqrt(A*A + B*B)
    A = A/M
    B = B/M
    C = C/M

    D = A * points[:, :, :, 0] + B * points[:, :, :, 1] + C

    new_points[:, :, :, 0] = points[:, :, :, 0] - 2 * A * D
    new_points[:, :, :, 1] = points[:, :, :, 1] - 2 * B * D

    return new_points
