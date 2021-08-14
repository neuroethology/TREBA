import torch
import numpy as np
import pickle
from scipy.ndimage import gaussian_filter1d
import math as mh

from util.datasets import LabelFunction
from util.datasets.mouse_v1.preprocess import *

FRAME_WIDTH_TOP = 1024
FRAME_HEIGHT_TOP = 570

# Our dataset has 21 frames. 
# TODO: Make this more flexible.
MIDDLE_INDEX = 10


class ReadLabels(LabelFunction):

    name = 'read_labels'
    svd_computer = None
    mean = None

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)
        

    def label_func(self, states, actions, true_label):

        return true_label


    def plot(self, ax, states, label, width, length):
        return ax    


class MiddleAngleSocial(LabelFunction):

    name = 'middle_angle_social'
    svd_computer = None
    mean = None

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)
        with open(svd_computer_path, 'rb') as f:
            self.svd_computer = pickle.load(f)
        with open(mean_path, 'rb') as f:        
            self.mean = pickle.load(f)
        

    def label_func(self, states, actions, true_label=None):
        keypoints = transform_svd_to_keypoints(states.numpy()[:-1], self.svd_computer, self.mean,
            stack_agents = True)
        keypoints = unnormalize(keypoints)
        
        keypoints_resident = keypoints[0].reshape((-1, 7, 2))[MIDDLE_INDEX]
        keypoints_intruder = keypoints[1].reshape((-1, 7, 2))[MIDDLE_INDEX]  
        
        angle = social_angle(keypoints_resident[:, 0], keypoints_resident[:, 1],
            keypoints_intruder[:, 0], keypoints_intruder[:, 1])

        label_tensor = torch.from_numpy(np.array(angle))

        label_tensor = label_tensor.to(states.device)  

        return torch.mean(label_tensor.float())


    def plot(self, ax, states, label, width, length):
        return ax            


class MiddleAngleSocialIntruder(LabelFunction):

    name = 'middle_angle_social_intruder'
    svd_computer = None
    mean = None

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)
        with open(svd_computer_path, 'rb') as f:
            self.svd_computer = pickle.load(f)
        with open(mean_path, 'rb') as f:        
            self.mean = pickle.load(f)
        

    def label_func(self, states, actions, true_label=None):
        keypoints = transform_svd_to_keypoints(states.numpy()[:-1], self.svd_computer, self.mean,
                        stack_agents = True)
        keypoints = unnormalize(keypoints)

        keypoints_resident = keypoints[0].reshape((-1, 7, 2))[MIDDLE_INDEX]
        keypoints_intruder = keypoints[1].reshape((-1, 7, 2))[MIDDLE_INDEX]   
        
        angle = social_angle(keypoints_intruder[:, 0], keypoints_intruder[:, 1],
            keypoints_resident[:, 0], keypoints_resident[:, 1])

        label_tensor = torch.from_numpy(np.array(angle))

        label_tensor = label_tensor.to(states.device)  

        return torch.mean(label_tensor.float())


    def plot(self, ax, states, label, width, length):
        return ax            


class MiddleMovementNoseResident(LabelFunction):

    name = 'middle_movement_nose_resident'
    svd_computer = None
    mean = None

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)
        with open(svd_computer_path, 'rb') as f:
            self.svd_computer = pickle.load(f)
        with open(mean_path, 'rb') as f:        
            self.mean = pickle.load(f)
        

    def label_func(self, states, actions, true_label=None):
        keypoints = transform_svd_to_keypoints(states.numpy()[:-1], self.svd_computer, self.mean,
            stack_agents = True)
        keypoints = unnormalize(keypoints)

        keypoints_resident = keypoints[0].reshape((-1, 7, 2))        

        speed = nose_to_centroid_movement(keypoints_resident[MIDDLE_INDEX-2, :,0], 
            keypoints_resident[MIDDLE_INDEX-2, :,1],
            keypoints_resident[MIDDLE_INDEX+2, :,0], 
            keypoints_resident[MIDDLE_INDEX+2, :,1])

        label_tensor = torch.from_numpy(np.array(speed))

        label_tensor = label_tensor.to(states.device)  

        return torch.mean(label_tensor.float())


    def plot(self, ax, states, label, width, length):
        return ax                   


class MiddleMovementNoseIntruder(LabelFunction):

    name = 'middle_movement_nose_intruder'
    svd_computer = None
    mean = None

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)
        with open(svd_computer_path, 'rb') as f:
            self.svd_computer = pickle.load(f)
        with open(mean_path, 'rb') as f:        
            self.mean = pickle.load(f)
        

    def label_func(self, states, actions, true_label=None):
        keypoints = transform_svd_to_keypoints(states.numpy()[:-1], self.svd_computer, self.mean,
            stack_agents = True)
        keypoints = unnormalize(keypoints)

        keypoints_intruder = keypoints[1].reshape((-1, 7, 2))        

        speed = nose_to_centroid_movement(keypoints_intruder[MIDDLE_INDEX-2, :,0],
            keypoints_intruder[MIDDLE_INDEX-2, :,1],
            keypoints_intruder[MIDDLE_INDEX+2, :,0],
            keypoints_intruder[MIDDLE_INDEX+2, :,1])

        label_tensor = torch.from_numpy(np.array(speed))

        label_tensor = label_tensor.to(states.device)  

        return torch.mean(label_tensor.float())


    def plot(self, ax, states, label, width, length):
        return ax                           



class MiddleDistNoseNose(LabelFunction):

    name = 'middle_dist_nose_nose'
    svd_computer = None
    mean = None

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)
        with open(svd_computer_path, 'rb') as f:
            self.svd_computer = pickle.load(f)
        with open(mean_path, 'rb') as f:        
            self.mean = pickle.load(f)
        

    def label_func(self, states, actions, true_label=None):
        keypoints = transform_svd_to_keypoints(states.numpy()[:-1], self.svd_computer, self.mean,
            stack_agents = True)
        keypoints = unnormalize(keypoints)
    
        keypoints_resident = keypoints[0].reshape((-1, 7, 2))[MIDDLE_INDEX]
        keypoints_intruder = keypoints[1].reshape((-1, 7, 2))[MIDDLE_INDEX]  
        
        dist = dist_nose_nose(keypoints_resident[:,0], keypoints_resident[:, 1],
            keypoints_intruder[:,0], keypoints_intruder[:, 1])

        label_tensor = torch.from_numpy(np.array(dist))

        label_tensor = label_tensor.to(states.device)  

        return torch.mean(label_tensor.float())


    def plot(self, ax, states, label, width, length):
        return ax          



class MiddleDistNoseTail(LabelFunction):

    name = 'middle_dist_nose_tail'
    svd_computer = None
    mean = None

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)
        with open(svd_computer_path, 'rb') as f:
            self.svd_computer = pickle.load(f)
        with open(mean_path, 'rb') as f:        
            self.mean = pickle.load(f)
        

    def label_func(self, states, actions, true_label=None):
        keypoints = transform_svd_to_keypoints(states.numpy()[:-1], self.svd_computer, self.mean,
            stack_agents = True)
        keypoints = unnormalize(keypoints)

        keypoints_resident = keypoints[0].reshape((-1, 7, 2))[MIDDLE_INDEX]
        keypoints_intruder = keypoints[1].reshape((-1, 7, 2))[MIDDLE_INDEX] 
        
        dist = dist_nose_tail(keypoints_resident[:,0], keypoints_resident[:, 1],
            keypoints_intruder[:,0], keypoints_intruder[:, 1])

        label_tensor = torch.from_numpy(np.array(dist))

        label_tensor = label_tensor.to(states.device)  

        return torch.mean(label_tensor.float())


    def plot(self, ax, states, label, width, length):
        return ax          


class MiddleAngleHeadBodyResident(LabelFunction):

    name = 'middle_angle_head_body_resident'
    svd_computer = None
    mean = None

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)
        with open(svd_computer_path, 'rb') as f:
            self.svd_computer = pickle.load(f)
        with open(mean_path, 'rb') as f:        
            self.mean = pickle.load(f)
        

    def label_func(self, states, actions, true_label=None):
        keypoints = transform_svd_to_keypoints(states.numpy()[:-1], self.svd_computer, self.mean,
            stack_agents = True)
        keypoints = unnormalize(keypoints)

        keypoints_resident = keypoints[0].reshape((-1, 7, 2))[MIDDLE_INDEX]

        angle = interior_angle(keypoints_resident[0], keypoints_resident[3], keypoints_resident[6])

        label_tensor = torch.from_numpy(np.array(angle))

        label_tensor = label_tensor.to(states.device)  

        return torch.mean(label_tensor.float())


    def plot(self, ax, states, label, width, length):
        return ax          


class MiddleAngleHeadBodyIntruder(LabelFunction):

    name = 'middle_angle_head_body_intruder'
    svd_computer = None
    mean = None

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)
        with open(svd_computer_path, 'rb') as f:
            self.svd_computer = pickle.load(f)
        with open(mean_path, 'rb') as f:        
            self.mean = pickle.load(f)
        

    def label_func(self, states, actions, true_label=None):
        keypoints = transform_svd_to_keypoints(states.numpy()[:-1], self.svd_computer, self.mean,
            stack_agents = True)
        keypoints = unnormalize(keypoints)

        keypoints_intruder = keypoints[1].reshape((-1, 7, 2))[MIDDLE_INDEX]

        angle = interior_angle(keypoints_intruder[0], keypoints_intruder[3], keypoints_intruder[6])

        label_tensor = torch.from_numpy(np.array(angle))

        label_tensor = label_tensor.to(states.device)  

        return torch.mean(label_tensor.float())


    def plot(self, ax, states, label, width, length):
        return ax                  


class MiddleSpeedResident(LabelFunction):

    name = 'middle_speed_resident'
    svd_computer = None
    mean = None

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)
        with open(svd_computer_path, 'rb') as f:
            self.svd_computer = pickle.load(f)
        with open(mean_path, 'rb') as f:        
            self.mean = pickle.load(f)
        

    def label_func(self, states, actions, true_label=None):
        keypoints = transform_svd_to_keypoints(states.numpy()[:-1], self.svd_computer, self.mean,
            stack_agents = True)
        keypoints = unnormalize(keypoints)
        
        keypoints_resident = keypoints[0].reshape((-1, 7, 2))

        speed = speed_centroid(keypoints_resident[MIDDLE_INDEX-2, :,0], 
            keypoints_resident[MIDDLE_INDEX-2, :,1],
            keypoints_resident[MIDDLE_INDEX+2, :,0],
            keypoints_resident[MIDDLE_INDEX+2, :,1])

        label_tensor = torch.from_numpy(np.array(speed))

        label_tensor = label_tensor.to(states.device)  

        return torch.mean(label_tensor.float())


    def plot(self, ax, states, label, width, length):
        return ax          


class MiddleSpeedIntruder(LabelFunction):

    name = 'middle_speed_intruder'
    svd_computer = None
    mean = None

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)
        with open(svd_computer_path, 'rb') as f:
            self.svd_computer = pickle.load(f)
        with open(mean_path, 'rb') as f:        
            self.mean = pickle.load(f)
        

    def label_func(self, states, actions, true_label=None):
        keypoints = transform_svd_to_keypoints(states.numpy()[:-1], self.svd_computer, self.mean,
            stack_agents = True)
        keypoints = unnormalize(keypoints)

        keypoints_intruder = keypoints[1].reshape((-1, 7, 2))

        speed = speed_centroid(keypoints_intruder[MIDDLE_INDEX-2, :,0], 
            keypoints_intruder[MIDDLE_INDEX-2, :,1],
            keypoints_intruder[MIDDLE_INDEX+2, :,0],
            keypoints_intruder[MIDDLE_INDEX+2, :,1])

        label_tensor = torch.from_numpy(np.array(speed))

        label_tensor = label_tensor.to(states.device)  

        return torch.mean(label_tensor.float())


    def plot(self, ax, states, label, width, length):
        return ax                    


def speed_centroid(x1, y1, x2, y2):
    x1_c = np.mean(x1)
    y1_c = np.mean(y1)
    x2_c = np.mean(x2)
    y2_c = np.mean(y2)

    dx = x2_c - x1_c
    dy = y2_c - y1_c    

    return np.linalg.norm([dx, dy])


def nose_to_centroid_movement(x1, y1, x2, y2):
    x1_c = np.mean(x1)
    y1_c = np.mean(y1)
    x2_c = np.mean(x2)
    y2_c = np.mean(y2)

    dx = x2_c - x1_c
    dy = y2_c - y1_c    
    
    dx_nose = x2[0] - x1[0]
    dy_nose = y2[0] - y1[0]

    return np.linalg.norm([dx_nose - dx, dy_nose - dy])   

def acceleration_centroid(x0, y0, x1, y1, x2, y2):

    x0_c = np.mean(x0)
    y0_c = np.mean(y0)
    x1_c = np.mean(x1)
    y1_c = np.mean(y1)
    x2_c = np.mean(x2)
    y2_c = np.mean(y2)

    ax = x2_c - 2*x1_c + x0_c
    ay = y2_c - 2*y1_c + y0_c    

    return np.linalg.norm([ax, ay])


def get_angle(Ax, Ay, Bx, By):
    angle = (mh.atan2(Ax - Bx, Ay - By) + mh.pi/2) % (mh.pi*2)
    return angle

def social_angle(x1, y1, x2, y2):
    x_dif = np.mean(x1) - np.mean(x2)
    y_dif = np.mean(y1) - np.mean(y2)
    theta = (np.arctan2(y_dif, x_dif) + 2*np.pi) % 2*np.pi

    ori_body = get_angle(x1[6], y1[6], x1[3], y1[3])
    ang = np.mod(theta - ori_body, 2*np.pi)
    return np.minimum(ang, 2*np.pi - ang)


def dist_nose_nose(x1, y1, x2, y2):
    x_dif = x1[0] - x2[0]
    y_dif = y1[0] - y2[0]
    return np.linalg.norm([x_dif, y_dif])

def dist_nose_tail(x1, y1, x2, y2):
    x_dif = x1[0] - x2[6]
    y_dif = y1[0] - y2[6]
    return np.linalg.norm([x_dif, y_dif])


def interior_angle(p0, p1, p2):

    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p1) - np.array(p2)

    return mh.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
