import argparse
import json
import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from util.datasets import load_dataset
from lib.models import get_model_class

def extract_features(exp_dir,
                     trial_id,
                     test_name,
                     save_name):
    '''Saves features after training. '''
    print('#################### Feature Extraction {} ####################'.format(trial_id))

    # Get trial folder
    trial_dir = os.path.join(exp_dir, trial_id)
    assert os.path.isfile(os.path.join(trial_dir, 'summary.json'))

    # Load config
    with open(os.path.join(exp_dir, 'configs', '{}.json'.format(trial_id)), 'r') as f:
        config = json.load(f)
    data_config = config['data_config']
    model_config = config['model_config']

    # No need to load training data for feature extraction.

    data_config["label_train_set"] = False
    # Load dataset
    if test_name is not None:
        data_config["test_name"] = test_name

    dataset = load_dataset(data_config)
    dataset.eval()

    # Load best model
    state_dict = torch.load(os.path.join(
            trial_dir, 'best.pth'), map_location=lambda storage, loc: storage)

    model_class = get_model_class(model_config['name'].lower())
    
    model_config['label_functions'] = dataset.active_label_functions
    
    model_config['augmentations'] = dataset.active_augmentations    
    model = model_class(model_config)
    model.load_state_dict(state_dict)

    num_samples = 128
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=False)

    save_array = np.array([])

    for batch_idx, (states, actions, labels_dict) in enumerate(loader):

        labels_dict = {key: value for key, value in labels_dict.items()}

        states = states.transpose(0, 1)
        actions = actions.transpose(0, 1)

        with torch.no_grad():   
            if len(dataset.active_label_functions) > 0:         
                label_list = []
                for lf_idx, lf_name in enumerate(labels_dict):
                    label_list.append(labels_dict[lf_name])
                label_input = torch.cat(label_list, -1)

                encodings_mean, _ = model.encode_mean(states[:-1], actions,
                                               labels=label_input)
            else:
                encodings_mean, _ = model.encode_mean(states[:-1], actions)

            if save_array.shape[0] == 0:
                save_array = encodings_mean
            else:
                save_array = np.concatenate([save_array, encodings_mean], axis=0)

    np.savez(os.path.join(trial_dir, save_name), save_array)
    print("Saved Features: " + os.path.join(trial_dir, save_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--exp_folder', type=str,
                        required=True, default=None,
                        help='folder of experiments from which to load models')
    parser.add_argument('--save_dir', type=str,
                        required=False, default='saved',
                        help='save directory for experiments from project directory')
    parser.add_argument('--feature_extraction', type=str,
                        required=False, default=None,
                        help='paths to trajectory data for feature extraction')
    parser.add_argument('--feature_names', type=str,
                        required=False, default=None,
                        help='paths to save extracted features')
    args = parser.parse_args()    

    # Get exp_directory
    exp_dir = os.path.join(os.getcwd(), args.save_dir, args.exp_folder)

    # Load master file
    print(exp_dir)
    assert os.path.isfile(os.path.join(exp_dir, 'master.json'))
    with open(os.path.join(exp_dir, 'master.json'), 'r') as f:
        master = json.load(f)

    input_feature_files = args.feature_extraction.split(',')
    
    assert args.feature_names is not None

    output_feature_names = args.feature_names.split(',')    
    assert len(input_feature_files) == len(output_feature_names)


    for trial_id in master['summaries']:

        for index, input_features in enumerate(input_feature_files):
            extract_features(exp_dir, trial_id, input_features, 
                output_feature_names[index] + "_" + trial_id)
