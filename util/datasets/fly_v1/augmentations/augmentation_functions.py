import torch
import numpy as np
import pickle
from scipy.ndimage import gaussian_filter1d

from util.datasets import Augmentations

FRAME_WIDTH_TOP = 144
FRAME_HEIGHT_TOP = 144

RESIDENT_COLOR = 'lawngreen'
INTRUDER_COLOR = 'skyblue'


class AllRandomAugmentations(Augmentations):
    '''Applies a series of random augmentations to the trajectory such that
       program outputs are not affected.'''

    name = 'all_random'

    def __init__(self, aug_config):
        super().__init__(aug_config)

    def augment_func(self, states, actions):
        keypoints = unnormalize(states.cpu().numpy())

        # 21 x 20
        seq_len = states.shape[0]
        new_keypoints = keypoints

        # ROTATION AND TRANSLATION
        new_keypoints = add_rotation_translation(new_keypoints)

        # Randomly apply vertical and horizonal reflection.
        add_vertical_reflection = np.random.uniform(0, 1)
        if add_vertical_reflection > 0.5:
            new_keypoints = reflect_points(new_keypoints, 0, 1,
               -FRAME_HEIGHT_TOP//2, vertical = True)

        add_horizontal_reflection = np.random.uniform(0, 1)
        if add_horizontal_reflection > 0.5:
            new_keypoints = reflect_points(new_keypoints, 1, 0,
              -FRAME_WIDTH_TOP//2, vertical = False)

        new_keypoints = normalize(new_keypoints)

        new_states = torch.from_numpy(new_keypoints).to(states.device).float()
        return new_states, new_states[1:] - new_states[:-1]

    def plot(self, ax, states):

        return ax


def add_rotation_translation(points, rotation_range=np.pi, translation_range=500):
    '''Apply a random rotation to input, and translate such that the trajectory
       remains within the bounds of the arena.'''

    original = points.copy()

    image_center = [FRAME_WIDTH_TOP / 2, FRAME_HEIGHT_TOP / 2]

    mouse_rotation = np.random.uniform(
        low=-1 * rotation_range, high=rotation_range)
    R = (np.array([[np.cos(mouse_rotation), -np.sin(mouse_rotation)],
                   [np.sin(mouse_rotation),  np.cos(mouse_rotation)]]))

    
    points[:, 2:4] = np.matmul(R, points[:, 2:4].transpose()).transpose()
    points[:, 12:14] = np.matmul(R, points[:, 12:14].transpose()).transpose()

    # points = 21 x 20, for the fly dataset
    indeces = [0, 6, 8, 10, 16, 18]
    y_indeces = [1, 7, 9, 11, 17, 19]
    for index in indeces:
        points[:, index:index + 2] = np.matmul(R, (points[:, index:index + 2] - image_center).transpose(
            )).transpose() + image_center  

    # Check if possible for trajectory to fit within borders
    bounded = ((np.amax(points[:, indeces]) - np.amin(points[:, indeces])) < FRAME_WIDTH_TOP) and ((np.amax(points[:, y_indeces]) - np.amin(points[:, y_indeces])) < FRAME_HEIGHT_TOP)

    if bounded:
        # Shift all points to within borders first
        horizontal_shift=np.amax(points[:, indeces] - FRAME_WIDTH_TOP)
        horizontal_shift_2=np.amin(points[:, indeces])
        if horizontal_shift > 0:
            points[:, indeces]=points[:, indeces] - horizontal_shift
        if horizontal_shift_2 < 0:
            points[:, indeces]=points[:, indeces] - horizontal_shift_2

        vertical_shift=np.amax(points[:, y_indeces] - FRAME_HEIGHT_TOP)
        vertical_shift_2=np.amin(points[:, y_indeces])
        if vertical_shift > 0:
            points[:, y_indeces]=points[:, y_indeces] - vertical_shift
        if vertical_shift_2 < 0:
            points[:, y_indeces]=points[:, y_indeces] - vertical_shift_2


        max_horizontal_shift=np.amin(FRAME_WIDTH_TOP - points[:, indeces])
        min_horizontal_shift=np.amin(points[:, indeces])
        max_vertical_shift=np.amin(FRAME_HEIGHT_TOP - points[:, y_indeces])
        min_vertical_shift=np.amin(points[:, y_indeces])
        horizontal_shift=np.random.uniform(
            low = -1 * min_horizontal_shift, high = max_horizontal_shift)
        vertical_shift=np.random.uniform(
            low = -1 * min_vertical_shift, high = max_vertical_shift)

        points[:, indeces]=points[:, indeces] + horizontal_shift
        points[:, y_indeces]=points[:, y_indeces] + vertical_shift

        return points
    # If cannot fit within borders, return the original set of points.
    else:
        return original


def reflect_points(points, A, B, C, vertical = False):
    # Reflect given fly poses along the given direction.
    # A * x + B * y + C = 0
    new_points=np.zeros(points.shape)

    M=np.sqrt(A * A + B * B)
    A=A / M
    B=B / M
    C=C / M

    # Fly indeces that represent x,y locations
    indeces = [0, 6, 8, 10, 16, 18]
    y_indeces = [1, 7, 9, 11, 17, 19]


    D=A * points[:, indeces] + B * points[:, y_indeces] + C

    new_points[:, indeces]=points[:, indeces] - 2 * A * D
    new_points[:, y_indeces]=points[:, y_indeces] - 2 * B * D


    # Vertical flip
    if vertical:
        theta = np.arctan2(points[:,2], points[:,3])
        theta_2 = np.arctan2(points[:,12], points[:,13])
        # vertical reflection
        sin_1 = np.sin(-1*theta) 
        cos_1 = np.cos(-1*theta)
        sin_2 = np.sin(-1*theta_2) 
        cos_2 = np.cos(-1*theta_2)             
        new_points[:, 2] = sin_1
        new_points[:, 3] = cos_1
        new_points[:, 12] = sin_2
        new_points[:, 13] = cos_2

    else:
        # Horizontal flip

        theta = np.arctan2(points[:,2], points[:,3])
        theta_2 = np.arctan2(points[:,12], points[:,13])

        theta[np.where((theta > 0))] = np.pi - theta[np.where((theta > 0))]
        theta[np.where((theta < 0))] = -1*(np.pi + theta[np.where((theta < 0))])

        theta_2[np.where((theta_2 > 0))] = np.pi - theta_2[np.where((theta_2 > 0))]
        theta_2[np.where((theta_2 < 0))] = -1*(np.pi + theta_2[np.where((theta_2 < 0))])        
        
        #horizontal
        sin_1 = np.sin(theta) 
        cos_1 = np.cos(theta)
        sin_2 = np.sin(theta_2)
        cos_2 = np.cos(theta_2)
        new_points[:, 2] = sin_1
        new_points[:, 3] = cos_1
        new_points[:, 12] = sin_2
        new_points[:, 13] = cos_2        

    # Copy lengths
    length_indeces = [4,5,14,15]
    new_points[:, length_indeces] = points[:, length_indeces]

    return new_points


def normalize(data):
    """Scale by dimensions of image and mean-shift to center of image."""
    data_2=np.zeros(data.shape)
    state_dim=data_2.shape[-1] // 2
    keypoint_indeces=[[0, 1], [6, 7], [8, 9]]
    length_indeces=[4, 5]
    # Data is sequence_num x seq_len x dims

    # Assume square image
    shift=int(FRAME_WIDTH_TOP / 2)
    scale=int(FRAME_WIDTH_TOP / 2)
    for index in keypoint_indeces:
        data_2[:,  index[0]]=(data[:, index[0]] - shift) / scale
        data_2[:, index[1]]=(data[:, index[1]] - shift) / scale

        data_2[:, index[0] + state_dim]=(data[:, index[0] + state_dim] - shift) / scale
        data_2[:, index[1] + state_dim]=(data[:, index[1] + state_dim] - shift) / scale

    for index in length_indeces:
        data_2[:, index]=data[:, index] / scale
        data_2[:, index + state_dim]=data[:, index + state_dim] / scale

    data_2[:, 2:4]=data[:, 2:4]
    data_2[:, 12:14]=data[:, 12:14]

    return data_2



def unnormalize(data):
    """Undo normalize."""
    data_2=np.zeros(data.shape)
    state_dim=data_2.shape[-1] // 2
    keypoint_indeces=[[0, 1], [6, 7], [8, 9]]
    length_indeces=[4, 5]
    # Data is sequence_num x seq_len x dims

    # Assume square image
    shift=int(FRAME_WIDTH_TOP / 2)
    scale=int(FRAME_WIDTH_TOP / 2)
    for index in keypoint_indeces:
        data_2[:, index[0]]=data[:, index[0]] * scale + shift
        data_2[:, index[1]]=data[:, index[1]] * scale + shift

        data_2[:, index[0] + state_dim]=data[:,
            index[0] + state_dim] * scale + shift
        data_2[:, index[1] + state_dim] =data[:,
            index[1] + state_dim] * scale + shift

    for index in length_indeces:
        data_2[:, index] = data[:, index] * scale
        data_2[:, index + state_dim] = data[:, index + state_dim] * scale

    data_2[:, 2:4] =data[:, 2:4]
    data_2[:, 12:14] =data[:, 12:14]
    return data_2
