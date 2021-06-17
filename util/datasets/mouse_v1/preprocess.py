import numpy as np
import pickle
from sklearn.decomposition import TruncatedSVD

########### MOUSE DATASET FRAME WIDTH
FRAME_WIDTH_TOP = 1024
FRAME_HEIGHT_TOP = 570

# SVD paths
svd_computer_path = 'util/datasets/mouse_v1/svd/calms21_svd_computer.pickle'
mean_path = 'util/datasets/mouse_v1/svd/calms21_mean.pickle'

def normalize(data):
    """Scale by dimensions of image and mean-shift to center of image."""
    state_dim = data.shape[2] // 2
    shift = [int(FRAME_WIDTH_TOP / 2), int(FRAME_HEIGHT_TOP / 2)] * state_dim
    scale = [int(FRAME_WIDTH_TOP / 2), int(FRAME_HEIGHT_TOP / 2)] * state_dim
    return np.divide(data - shift, scale)


def unnormalize(data):
    """Undo normalize."""
    state_dim = data.shape[2] // 2
    shift = [int(FRAME_WIDTH_TOP / 2), int(FRAME_HEIGHT_TOP / 2)] * state_dim
    scale = [int(FRAME_WIDTH_TOP / 2), int(FRAME_HEIGHT_TOP / 2)] * state_dim
    return np.multiply(data, scale) + shift


def transform_to_svd_components(data,
                                center_index=3,
                                n_components=5,
                                svd_computer=None,
                                mean=None,
                                stack_agents = False,
                                stack_axis = 1,
                                save_svd = False):
    # data shape is num_seq x 2 x 7 x 2
    resident_keypoints = data[:, 0, :, :]
    intruder_keypoints = data[:, 1, :, :]
    data = np.concatenate([resident_keypoints, intruder_keypoints], axis=0)

    # Center the data using given center_index
    mouse_center = data[:, center_index, :]
    centered_data = data - mouse_center[:, np.newaxis, :]

    # Rotate such that keypoints 3 and 6 are parallel with the y axis
    mouse_rotation = np.arctan2(
        data[:, 3, 0] - data[:, 6, 0], data[:, 3, 1] - data[:, 6, 1])

    R = (np.array([[np.cos(mouse_rotation), -np.sin(mouse_rotation)],
                   [np.sin(mouse_rotation),  np.cos(mouse_rotation)]]).transpose((2, 0, 1)))

    # Encode mouse rotation as sine and cosine
    mouse_rotation = np.concatenate([np.sin(mouse_rotation)[:, np.newaxis], np.cos(
        mouse_rotation)[:, np.newaxis]], axis=-1)

    centered_data = np.matmul(R, centered_data.transpose(0, 2, 1))
    centered_data = centered_data.transpose((0, 2, 1))

    centered_data = centered_data.reshape((-1, 14))

    if mean is None:
        mean = np.mean(centered_data, axis=0)
    centered_data = centered_data - mean

    # Compute SVD components
    if svd_computer is None:
        svd_computer = TruncatedSVD(n_components=n_components)
        svd_data = svd_computer.fit_transform(centered_data)
    else:
        svd_data = svd_computer.transform(centered_data)
        explained_variances = np.var(svd_data, axis=0) / np.var(centered_data, axis=0).sum()

    # Concatenate state as mouse center, mouse rotation and svd components
    data = np.concatenate([mouse_center, mouse_rotation, svd_data], axis=1)
    resident_keypoints = data[:data.shape[0] // 2, :]
    intruder_keypoints = data[data.shape[0] // 2:, :]

    if not stack_agents:
        data = np.concatenate([resident_keypoints, intruder_keypoints], axis=-1)
    else:
        data = np.stack([resident_keypoints, intruder_keypoints], axis=stack_axis)

    if save_svd:
        with open(svd_computer_path, 'wb') as f:
            pickle.dump(svd_computer, f)
        with open(mean_path, 'wb') as f:        
            pickle.dump(mean, f)

    return data, svd_computer, mean


def unnormalize_keypoint_center_rotation(keypoints, center, rotation):

    keypoints = keypoints.reshape((-1, 7, 2))

    # Apply inverse rotation
    rotation = -1 * rotation
    R = np.array([[np.cos(rotation), -np.sin(rotation)],
                  [np.sin(rotation),  np.cos(rotation)]]).transpose((2, 0, 1))
    centered_data = np.matmul(R, keypoints.transpose(0, 2, 1))

    keypoints = centered_data + center[:, :, np.newaxis]
    keypoints = keypoints.transpose(0, 2, 1)

    return keypoints.reshape(-1, 14)


def transform_svd_to_keypoints(data, svd_computer, mean, stack_agents = False,
                            stack_axis = 0):
    num_components = data.shape[1] // 2
    resident_center = data[:, :2]
    resident_rotation = data[:, 2:4]
    resident_components = data[:, 4:num_components]
    intruder_center = data[:, num_components:num_components + 2]
    intruder_rotation = data[:, num_components + 2:num_components + 4]
    intruder_components = data[:, num_components + 4:]

    resident_keypoints = svd_computer.inverse_transform(resident_components)
    intruder_keypoints = svd_computer.inverse_transform(intruder_components)

    if mean is not None:
        resident_keypoints = resident_keypoints + mean
        intruder_keypoints = intruder_keypoints + mean

    # Compute rotation angle from sine and cosine representation
    resident_rotation = np.arctan2(
        resident_rotation[:, 0], resident_rotation[:, 1])
    intruder_rotation = np.arctan2(
        intruder_rotation[:, 0], intruder_rotation[:, 1])

    resident_keypoints = unnormalize_keypoint_center_rotation(
        resident_keypoints, resident_center, resident_rotation)
    intruder_keypoints = unnormalize_keypoint_center_rotation(
        intruder_keypoints, intruder_center, intruder_rotation)

    if not stack_agents:
        data = np.concatenate([resident_keypoints, intruder_keypoints], axis=-1)
    else:
        data = np.stack([resident_keypoints, intruder_keypoints], axis=stack_axis)        

    return data

