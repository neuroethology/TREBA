import os
import numpy as np
import torch

from util.datasets import TrajectoryDataset
from .label_functions import label_functions_list
from .augmentations import augmentation_list
from util.logging import LogEntry
import pickle

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import animation

from sklearn.decomposition import TruncatedSVD


ROOT_DIR = 'util/datasets/fly_v1/data'

TRAIN_FILE = 'fly_train_shuffled'
TEST_FILE = 'fly_val_shuffled'


FRAME_WIDTH_TOP = 144
FRAME_HEIGHT_TOP = 144

RESIDENT_COLOR = 'lawngreen'
RESIDENT_WING_COLOR = 'mintcream'

INTRUDER_COLOR = 'skyblue'
INTRUDER_WING_COLOR = 'aliceblue'

plot_title = "Fly Trajectory Plot"


class FlyV1Dataset(TrajectoryDataset):

    name = 'fly_v1'
    all_label_functions = label_functions_list
    all_augmentations = augmentation_list

    # Default config
    _seq_len = 21
    _state_dim = 20
    _action_dim = 20

    normalize_data = True

    test_name = TEST_FILE

    def __init__(self, data_config):
        super().__init__(data_config)

    def _load_data(self):
        # Process configs
        if 'normalize_data' in self.config:
            self.normalize_data = self.config['normalize_data']

        if 'test_name' in self.config:
            self.test_name = self.config['test_name']            

        if 'labels' in self.config:
            for lf_config in self.config['labels']:
                lf_config['data_normalized'] = self.normalize_data

        self.log = LogEntry()

        self.train_states, self.train_actions = self._load_and_preprocess(
            train=True)
        self.test_states, self.test_actions = self._load_and_preprocess(
            train=False)

    def _load_and_preprocess(self, train):
        # Load the dataset from either train or test, then normalize by image dimensions.

        path = os.path.join(ROOT_DIR, TRAIN_FILE if train else self.test_name)
        file = np.load(path, allow_pickle=True)
        data = file['data']
        if len(data) == 2:
            data = data[0]

        # Subsample timesteps if needed
        data = data[:, ::self.subsample]

        # Normalize data
        # Here we use 10 dimensions for each fly:
        # centroid pos x, centroid pos y, sine of orientation, cosine of orientation,
        # major axis len, minor axis len, wing l x, wing l y, wing r x, wing r y,
        if self.normalize_data:
            data = normalize(data)

        states = data
        actions = states[:, 1:] - states[:, :-1]

        # Update dimensions
        self._seq_len = actions.shape[1]
        self._state_dim = states.shape[-1]
        self._action_dim = actions.shape[-1]

        return torch.Tensor(states), torch.Tensor(actions)

    def compute_and_log_metrics(self, states, actions=[], save=None):
        # Compute number of times agent is out of bounds.

        self.log.reset()

        states = states.detach().numpy()

        # Out of bounds rate
        # batch_size x seq_length
        if self.normalize_data:

            out_of_bounds = np.array(
                np.sum(np.absolute(states) > 1, axis=(-2, -1)) > 0)
            self.log.metrics['out_of_bounds_rate'] = np.sum(out_of_bounds)
            states = unnormalize(states)
        else:
            norm_states = normalize(states)
            out_of_bounds = np.array(
                np.sum(np.absolute(norm_states) > 1, axis=(-2, -1)) > 0)
            self.log.metrics['out_of_bounds_rate'] = np.sum(out_of_bounds)

        return self.log

    def save(self, states,
             actions=[],
             save_path='',
             save_name='',
             burn_in=0,
             labels=None,
             lf_list=[],
             single_plot=False):
        # Visualize the generated results.

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        states = states.detach().numpy()

        if self.normalize_data:
            states = unnormalize(states)

        for i in range(len(states)):
            current_save_path = os.path.join(save_path, "{:03}".format(i))
            if not os.path.exists(current_save_path):
                os.makedirs(current_save_path)

            seq = states[i]

            image_list = []
            for j in range(seq.shape[0]):
                fig, ax = _set_figax()

                plot_fly(ax, seq[j, :self._state_dim // 2],
                         color=RESIDENT_COLOR, wingcolor = RESIDENT_WING_COLOR)
                plot_fly(ax, seq[j, self._state_dim // 2:],
                         color=INTRUDER_COLOR, wingcolor = INTRUDER_WING_COLOR)

                ax.set_title(
                    plot_title + '\nseq {:03d}.png'.format(i) + ', frame {:03d}.png'.format(j))

                plt.tight_layout(pad=0)

                if len(save_name) == 0:
                    plt.savefig(os.path.join(
                        current_save_path, '{:03d}.png'.format(j)))
                else:
                    plt.savefig(os.path.join(current_save_path,
                                             '{}.png'.format(save_name)))

                image = np.fromstring(
                    fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                image = image.reshape(
                    fig.canvas.get_width_height()[::-1] + (3,))
                image_list.append(image)

                plt.close()

            # Plot animation.
            fig = plt.figure()
            im = plt.imshow(image_list[0])

            def animate(k):
                im.set_array(image_list[k])
                return im,
            ani = animation.FuncAnimation(
                fig, animate, frames=self._seq_len, blit=True)
            ani.save(os.path.join(save_path, '{:03d}_animation.gif'.format(i)),
                     writer='imagemagick', fps=10)
            plt.close()

    def save_augmented(self, states, augmented_states,
                       save_path='',
                       save_name='',
                       burn_in=0,
                       title='',
                       single_plot=False):
        # Visualize the trajectory with the augmented trajectory.

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        states = states.detach().numpy()
        augmented_states = augmented_states.detach().numpy()

        if self.normalize_data:
            states = unnormalize(states)
            augmented_states = unnormalize(augmented_states)

        img = np.zeros((FRAME_HEIGHT_TOP, FRAME_WIDTH_TOP, 3))

        for i in range(len(states)):

            seq = states[i] #.reshape((-1, 2, 7, 2))
            augmented_seq = augmented_states[i] #.reshape((-1, 2, 7, 2))

            image_list = []
            for j in range(seq.shape[0]):
                fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)

                ax1.imshow(img)
                ax1.get_xaxis().set_visible(False)
                ax1.get_yaxis().set_visible(False)

                ax2.imshow(img)
                ax2.get_xaxis().set_visible(False)
                ax2.get_yaxis().set_visible(False)

                plot_fly(ax1, seq[j, :self._state_dim // 2],
                         color=RESIDENT_COLOR, wingcolor = RESIDENT_WING_COLOR)
                plot_fly(ax1, seq[j, self._state_dim // 2:],
                         color=INTRUDER_COLOR, wingcolor = INTRUDER_WING_COLOR)

                plot_fly(ax2, augmented_seq[j, :self._state_dim // 2],
                         color=RESIDENT_COLOR, wingcolor = RESIDENT_WING_COLOR)
                plot_fly(ax2, augmented_seq[j, self._state_dim // 2:],
                         color=INTRUDER_COLOR, wingcolor = INTRUDER_WING_COLOR)

                fig.suptitle(
                    title + '\nseq {:03d}.png'.format(i) + ', frame {:03d}.png'.format(j))

                ax1.axis('off')
                ax2.axis('off')
                plt.tight_layout(pad=0)

                fig.canvas.draw()
                image = np.fromstring(
                    fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                image = image.reshape(
                    fig.canvas.get_width_height()[::-1] + (3,))
                image_list.append(image)

                plt.close()

            # Plot animation.
            fig = plt.figure()
            im = plt.imshow(image_list[0])

            def animate(k):
                im.set_array(image_list[k])
                return im,
            ani = animation.FuncAnimation(
                fig, animate, frames=self._seq_len, blit=True)
            ani.save(os.path.join(save_path, title + '_{:03d}_animation.gif'.format(i)),
                     writer='imagemagick', fps=15)
            plt.close()


def normalize(data):
    """Scale by dimensions of image and mean-shift to center of image."""
    state_dim = data.shape[2] // 2
    keypoint_indeces = [[0, 1], [6, 7], [8, 9]]
    length_indeces = [4, 5]
    # Data is sequence_num x seq_len x dims

    # Assume square image
    shift = int(FRAME_WIDTH_TOP / 2)
    scale = int(FRAME_WIDTH_TOP / 2)
    for index in keypoint_indeces:
        data[:, :, index[0]] = (data[:, :, index[0]] - shift) / scale
        data[:, :, index[1]] = (data[:, :, index[1]] - shift) / scale

        data[:, :, index[0] +
             state_dim] = (data[:, :, index[0] + state_dim] - shift) / scale
        data[:, :, index[1] +
             state_dim] = (data[:, :, index[1] + state_dim] - shift) / scale

    for index in length_indeces:
        data[:, :, index] = data[:, :, index] / scale
        data[:, :, index + state_dim] = data[:, :, index + state_dim] / scale

    return data


def unnormalize(data):
    """Undo normalize."""
    state_dim = data.shape[2] // 2
    keypoint_indeces = [[0, 1], [6, 7], [8, 9]]
    length_indeces = [4, 5]
    # Data is sequence_num x seq_len x dims

    # Assume square image for this fly dataset.
    shift = int(FRAME_WIDTH_TOP / 2)
    scale = int(FRAME_WIDTH_TOP / 2)
    for index in keypoint_indeces:
        data[:, :, index[0]] = data[:, :, index[0]] * scale + shift
        data[:, :, index[1]] = data[:, :, index[1]] * scale + shift

        data[:, :, index[0] + state_dim] = data[:,
                                                :, index[0] + state_dim] * scale + shift
        data[:, :, index[1] + state_dim] = data[:,
                                                :, index[1] + state_dim] * scale + shift

    for index in length_indeces:
        data[:, :, index] = data[:, :, index] * scale
        data[:, :, index + state_dim] = data[:, :, index + state_dim] * scale

    return data


def _set_figax():
    # Returns an empty figure for visualization.

    fig = plt.figure(figsize=(5, 5))

    img = np.zeros((FRAME_HEIGHT_TOP, FRAME_WIDTH_TOP, 3))

    ax = fig.add_subplot(111)
    ax.imshow(img)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    return fig, ax


def plot_fly(ax, pose, color, wingcolor):
    # Draws a fly in the given pose with color and wings of wingcolor
    # This function assumes order of inputs is:
    # centroid pos x, centroid pos y, sine of orientation, cosine of orientation,
    # major axis len, minor axis len, wing l x, wing l y, wing r x, wing r y,

    # Draw each keypoint
    ax.plot(pose[0], pose[1], 'o', color=color, markersize=7)
    ax.plot(pose[6], pose[7], 'o', color=wingcolor, markersize=4, alpha=0.8)
    ax.plot(pose[8], pose[9], 'o', color=wingcolor, markersize=4, alpha=0.8)

    # Plot major and minor axis
    major_axis = pose[4] / 2
    minor_axis = pose[5] / 2
    ax.plot([pose[0], pose[0] - major_axis * pose[3]],
             [pose[1], pose[1] + major_axis * pose[2]],
             linewidth = 2, color = color)

    ax.plot([pose[0], pose[0] + major_axis * pose[3]],
             [pose[1], pose[1] - major_axis * pose[2]],
             linewidth = 4, color = 'orange')    

    ax.plot([pose[0], pose[0] + minor_axis * pose[2]],
             [pose[1], pose[1] + minor_axis * pose[3]],
             linewidth = 2, color = color)    

    ax.plot([pose[0], pose[0] - minor_axis * pose[2]],
             [pose[1], pose[1] - minor_axis * pose[3]],
             linewidth = 2, color = color)        

    # Connect body and wings
    ax.plot([pose[0], pose[6]],[pose[1], pose[7]], color = wingcolor, alpha = 0.8)
    ax.plot([pose[0], pose[8]],[pose[1], pose[9]], color = wingcolor, alpha = 0.8)    
