from .core import TrajectoryDataset, LabelFunction, Augmentations
from .fly_v1 import FlyV1Dataset
from .mouse_v1 import MouseV1Dataset

dataset_dict = {
    'fly_v1' : FlyV1Dataset,
    'mouse_v1' : MouseV1Dataset    
}


def load_dataset(data_config):
    dataset_name = data_config['name'].lower()

    if dataset_name in dataset_dict:
        return dataset_dict[dataset_name](data_config)
    else:
        raise NotImplementedError
