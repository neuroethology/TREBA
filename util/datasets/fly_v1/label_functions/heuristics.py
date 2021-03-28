import torch
import numpy as np
import pickle
from scipy.ndimage import gaussian_filter1d
import math as mh
from copy import deepcopy


from util.datasets import LabelFunction

FRAME_WIDTH_TOP = 144
FRAME_HEIGHT_TOP = 144

# Our dataset has 21 frames. 
# TODO: Make this more flexible.
MIDDLE_INDEX = 10

EPSILON = 0.000001

class RandomLabeler(LabelFunction):
    ''' Returns a random value from 0 to 1. '''

    name = 'random_labeler'

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)

    def label_func(self, states, actions, true_label=None):

        label_tensor = torch.from_numpy(np.array([np.random.rand()]))

        label_tensor = label_tensor.to(states.device)  

        return torch.mean(label_tensor.float())


    def plot(self, ax, states, label, width, length):
        return ax    


class MiddleSpeedResident(LabelFunction):
    ''' Returns speed of the resident centroid point. '''

    name = 'middle_speed_resident'

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)

    def label_func(self, states, actions, true_label=None):
        
        keypoints = unnormalize(states)

        speed = speed_centroid(keypoints[MIDDLE_INDEX-2,0], keypoints[MIDDLE_INDEX-2,1],
            keypoints[MIDDLE_INDEX+2,0], keypoints[MIDDLE_INDEX+2, 1])

        label_tensor = torch.from_numpy(np.array(speed))

        label_tensor = label_tensor.to(states.device)  

        return label_tensor.float()


    def plot(self, ax, states, label, width, length):
        return ax    


class MiddleSpeedIntruder(LabelFunction):
    ''' Returns speed of the intruder centroid point. '''

    name = 'middle_speed_intruder'

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)

    def label_func(self, states, actions, true_label=None):
        #states: shape 21,20
        keypoints = unnormalize(states)
        
        speed = speed_centroid(keypoints[MIDDLE_INDEX-2,10], keypoints[MIDDLE_INDEX-2,11],
            keypoints[MIDDLE_INDEX+2, 10], keypoints[MIDDLE_INDEX+2, 11])

        label_tensor = torch.from_numpy(np.array(speed))

        label_tensor = label_tensor.to(states.device)  

        return label_tensor.float()


class MiddleAngularSpeedResident(LabelFunction):
    ''' Returns angular speed of the resident. '''

    name = 'middle_angular_speed_resident'

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)

    def label_func(self, states, actions, true_label=None):

        keypoints = unnormalize(states)
        
        middle_index = 10

        speed = angular_speed(keypoints[MIDDLE_INDEX-2,2], keypoints[MIDDLE_INDEX-2,3],
            keypoints[MIDDLE_INDEX+2,2], keypoints[MIDDLE_INDEX+2, 3])

        label_tensor = torch.from_numpy(np.array(speed))

        label_tensor = label_tensor.to(states.device)  

        return label_tensor.float()


    def plot(self, ax, states, label, width, length):
        return ax    


class MiddleAngularSpeedIntruder(LabelFunction):
    ''' Returns angular speed of intruder. '''

    name = 'middle_angular_speed_intruder'

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)

    def label_func(self, states, actions, true_label=None):
        #states: shape 21,20
        keypoints = unnormalize(states)
        
        speed = angular_speed(keypoints[MIDDLE_INDEX-2,12], keypoints[MIDDLE_INDEX-2,13],
            keypoints[MIDDLE_INDEX+2, 12], keypoints[MIDDLE_INDEX+2, 13])

        label_tensor = torch.from_numpy(np.array(speed))

        label_tensor = label_tensor.to(states.device)  

        return label_tensor.float()



    def plot(self, ax, states, label, width, length):
        return ax          


class MiddleAxisRatioResident(LabelFunction):
    ''' Returns ratio between the major and minor axis length of the fitted ellipse
        to the resident fly body. '''

    name = 'middle_axis_ratio_resident'

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)

    def label_func(self, states, actions, true_label=None):
        
        keypoints = unnormalize(states)

        if keypoints[MIDDLE_INDEX, 5] > EPSILON:
            ratio = keypoints[MIDDLE_INDEX, 4]/keypoints[MIDDLE_INDEX, 5]
        else:
            ratio = 1.0        

        label_tensor = torch.from_numpy(np.array(ratio))

        label_tensor = label_tensor.to(states.device)  

        return torch.mean(label_tensor.float())


    def plot(self, ax, states, label, width, length):
        return ax      



class MiddleAxisRatioIntruder(LabelFunction):
    ''' Returns ratio between the major and minor axis length of the fitted ellipse
        to the intruder fly body. '''

    name = 'middle_axis_ratio_intruder'

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)

    def label_func(self, states, actions, true_label=None):
        
        keypoints = unnormalize(states)

        if keypoints[MIDDLE_INDEX, 15] > EPSILON:
            ratio = keypoints[MIDDLE_INDEX, 14]/keypoints[MIDDLE_INDEX, 15]
        else:
            ratio = 1.0      
        

        label_tensor = torch.from_numpy(np.array(ratio))

        label_tensor = label_tensor.to(states.device)  

        return torch.mean(label_tensor.float())


    def plot(self, ax, states, label, width, length):
        return ax      



class MiddleAngleSocial(LabelFunction):
    '''Facing angle of resident. This is the angle between resident fly orientation
        and the line between resident and intruder fly centroid.'''

    name = 'middle_angle_social'

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)

    def label_func(self, states, actions, true_label=None):
        
        keypoints = unnormalize(states)
        
        angle = social_angle(keypoints[MIDDLE_INDEX, :10], keypoints[MIDDLE_INDEX, 10:])

        label_tensor = torch.from_numpy(np.array(angle))

        label_tensor = label_tensor.to(states.device)  

        return torch.mean(label_tensor.float())


    def plot(self, ax, states, label, width, length):
        return ax      



class MiddleAngleSocialIntruder(LabelFunction):
    '''Facing angle of intruder. This is the angle between intruder fly orientation
        and the line between resident and intruder fly centroid.'''

    name = 'middle_angle_social_intruder'

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)

    def label_func(self, states, actions, true_label=None):
        
        keypoints = unnormalize(states)
        
        angle = social_angle(keypoints[MIDDLE_INDEX, 10:], keypoints[MIDDLE_INDEX, :10])

        label_tensor = torch.from_numpy(np.array(angle))

        label_tensor = label_tensor.to(states.device)  

        return torch.mean(label_tensor.float())


    def plot(self, ax, states, label, width, length):
        return ax                    


class MiddleDistCentroid(LabelFunction):
    '''Distance between resident and intruder fly centroid.'''

    name = 'middle_dist_centroid'

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)

    def label_func(self, states, actions, true_label=None):
        keypoints = unnormalize(states)
        
        middle_index = 10

        dist_x = keypoints[MIDDLE_INDEX, 0] - keypoints[MIDDLE_INDEX, 10]
        dist_y = keypoints[MIDDLE_INDEX, 1] - keypoints[MIDDLE_INDEX, 11]
        dist = np.linalg.norm([dist_x, dist_y])

        label_tensor = torch.from_numpy(np.array(dist))

        label_tensor = label_tensor.to(states.device)  

        return torch.mean(label_tensor.float())


    def plot(self, ax, states, label, width, length):
        return ax          
  


class MiddleWingAngleMinResident(LabelFunction):
    '''Minimum wing angle of resident.'''

    name = 'middle_wing_angle_min_resident'

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)

    def label_func(self, states, actions, true_label=None):
        keypoints = unnormalize(states)
        
        angle_l, angle_r = wing_angles(keypoints[MIDDLE_INDEX, :10])
        min_angle = np.minimum(angle_l, angle_r)

        label_tensor = torch.from_numpy(np.array(min_angle))

        label_tensor = label_tensor.to(states.device)  

        return torch.mean(label_tensor.float())


    def plot(self, ax, states, label, width, length):
        return ax    


class MiddleWingAngleMaxResident(LabelFunction):
    '''Maximum wing angle of resident.'''

    name = 'middle_wing_angle_max_resident'

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)

    def label_func(self, states, actions, true_label=None):
        keypoints = unnormalize(states)
        
        angle_l, angle_r = wing_angles(keypoints[MIDDLE_INDEX, :10])
        max_angle = np.maximum(angle_l, angle_r)

        label_tensor = torch.from_numpy(np.array(max_angle))

        label_tensor = label_tensor.to(states.device)          

        return torch.mean(label_tensor.float())


    def plot(self, ax, states, label, width, length):
        return ax    


class MiddleWingAngleMinIntruder(LabelFunction):
    '''Minimum wing angle of intruder.'''    

    name = 'middle_wing_angle_min_intruder'

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)

    def label_func(self, states, actions, true_label=None):
        keypoints = unnormalize(states)
        
        angle_l, angle_r = wing_angles(keypoints[MIDDLE_INDEX, 10:])
        min_angle = np.minimum(angle_l, angle_r)

        label_tensor = torch.from_numpy(np.array(min_angle))

        label_tensor = label_tensor.to(states.device)  

        return torch.mean(label_tensor.float())


    def plot(self, ax, states, label, width, length):
        return ax    


class MiddleWingAngleMaxIntruder(LabelFunction):
    '''Maximum wing angle of intruder.'''    

    name = 'middle_wing_angle_max_intruder'

    def __init__(self, lf_config):
        super().__init__(lf_config, output_dim=1)

    def label_func(self, states, actions, true_label=None):
        keypoints = unnormalize(states)

        angle_l, angle_r = wing_angles(keypoints[MIDDLE_INDEX, 10:])
        max_angle = np.maximum(angle_l, angle_r)

        label_tensor = torch.from_numpy(np.array(max_angle))

        label_tensor = label_tensor.to(states.device)          

        return torch.mean(label_tensor.float())


    def plot(self, ax, states, label, width, length):
        return ax    


def speed_centroid(x1, y1, x2, y2):

    dx = x2 - x1
    dy = y2 - y1    

    return np.linalg.norm([dx, dy])    


def unnormalize(data):
    """Undo normalize."""
    data_2 = np.zeros(data.shape)
    state_dim = data_2.shape[1] // 2
    keypoint_indeces = [[0, 1], [6, 7], [8, 9]]
    length_indeces = [4,5]
    # Data is sequence_num x seq_len x dims

    # Assume square image
    shift = int(FRAME_WIDTH_TOP / 2)
    scale = int(FRAME_WIDTH_TOP / 2)
    for index in keypoint_indeces:
        data_2[:, index[0]]  = data[:, index[0]]* scale + shift
        data_2[:, index[1]]  = data[:, index[1]]*scale + shift

        data_2[:, index[0] + state_dim]  = data[:, index[0] + state_dim]* scale + shift
        data_2[:, index[1] + state_dim]  = data[:, index[1] + state_dim]* scale + shift

    for index in length_indeces:
        data_2[:, index]  = data[:, index]*scale       
        data_2[:, index + state_dim]  = data[:, index + state_dim]*scale 
      
    data_2[:, 2:4] = data[:, 2:4]
    data_2[:, 12:14] = data[:, 12:14]   
    return data_2


def social_angle(x1, x2):
    x_dif = x1[0] - x2[0]
    y_dif = x1[1] - x2[1]
    theta = (np.arctan2(y_dif, x_dif) + 2*np.pi) % (2*np.pi)

    #facing angle of fly
    ori_body = (np.arctan2(-1*x1[2], x1[3])  + 2*np.pi) % (np.pi*2)
    ang = np.mod(theta - ori_body, 2*np.pi)
    return np.minimum(ang, 2*np.pi - ang)
    

def dist_centroid(x1, y1, x2, y2):
    x_dif = np.mean(x1) - np.mean(x2)
    y_dif = np.mean(y1) - np.mean(y2)
    return np.linalg.norm([x_dif, y_dif])


def interior_angle(p0, p1, p2):

    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p1) - np.array(p2)

    return mh.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))


def wing_angles(pose):
    wing_l = pose[6:8]
    wing_r = pose[8:10]
    center = pose[0:2]
    back_point = pose[0:2] + [-1*pose[3], pose[2]]

    angle_l = interior_angle(wing_l, center, back_point)%(np.pi)
    angle_r = interior_angle(wing_r, center, back_point)%(np.pi)


    return np.minimum(angle_l, np.pi - angle_l),  np.minimum(angle_r, np.pi - angle_r)    


def angular_speed(sin_1, cos_1, sin_2, cos_2):
    ori_1 = mh.atan2(sin_1, cos_1)
    ori_2 = mh.atan2(sin_2, cos_2)

    ang = np.mod(ori_1 - ori_2, 2*np.pi)
    return np.minimum(ang, 2*np.pi - ang)
