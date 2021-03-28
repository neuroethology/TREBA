import argparse
import json
import os
import torch

from torch.utils.data import DataLoader
from util.datasets import load_dataset
from lib.models import get_model_class
from time import gmtime, strftime
from train import start_training
import numpy as np

from feature_extraction import extract_features

# Script starts here.
parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', type=str,
                    required=False, default='',
                    help='path to all config files for experiments')
parser.add_argument('--save_dir', type=str,
                    required=False, default='saved',
                    help='save directory for experiments from project directory')
parser.add_argument('--exp_name', type=str,
                    required=False, default='',
                    help='experiment name (default will be config folder name)')
parser.add_argument('-d', '--device', type=int,
                    required=False, default=-1,
                    help='device to run the experiment on')
parser.add_argument('-i', '--index', type=int,
                    required=False, default=-1,
                    help='run a single experiment in the folder, specified by index')
parser.add_argument('--test_code', action='store_true',
                    required=False, default=False,
                    help='whether or not to test code')
parser.add_argument('--feature_extraction', type=str,
                    required=False, default=None,
                    help='paths to trajectory data for feature extraction')
parser.add_argument('--feature_names', type=str,
                    required=False, default=None,
                    help='paths to save extracted features')                                        
args = parser.parse_args()


# Get JSON files
config_dir = os.path.join(os.getcwd(), 'configs', args.config_dir)
config_files = sorted([str(f) for f in os.listdir(config_dir) if os.path.isfile(os.path.join(config_dir, f))])
assert len(config_files) > 0

# Get experiment name
exp_name = args.exp_name if len(args.exp_name) > 0 else args.config_dir
print('Config folder:\t {}'.format(exp_name))

# Get save directory
save_dir = os.path.join(os.getcwd(), args.save_dir, exp_name)
if not os.path.exists(save_dir):
    os.makedirs(os.path.join(save_dir, 'configs'))
print('Save directory:\t {}'.format(save_dir))

# Get device ID
if torch.cuda.is_available() and args.device >= 0:
    assert args.device < torch.cuda.device_count()
    device = 'cuda:{:d}'.format(args.device)
else:
    device = 'cpu'
print('Device:\t {}'.format(device))

# Make sure feature extraction and feature names have the same length.
if args.feature_extraction is not None:
    input_feature_files = args.feature_extraction.split(',')
    
    assert args.feature_names is not None

    output_feature_names = args.feature_names.split(',')    
    assert len(input_feature_files) == len(output_feature_names)
    
# Create master file
master = {
    'start_time' : strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
    'experiment_name' : exp_name,
    'device' : device,
    'summaries' : {}
}

# Run a single experiment
if args.index >= 0:
    if args.index < len(config_files):
        config_files = [config_files[args.index]]
    else:
        print("WARNING: Index out of range, will run all experiments in folder.")

for config_file in config_files:

    # Load JSON config file
    with open(os.path.join(config_dir, config_file), 'r') as f:
        config = json.load(f)

    trial_id = config_file[:-5] # remove .json at the end
    print('########## Trial {}:{} ##########'.format(exp_name, trial_id))

    # Create save folder
    save_path = os.path.join(save_dir, trial_id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(os.path.join(save_path, 'checkpoints')) # for model checkpoints
        os.makedirs(os.path.join(save_path, 'results')) # for saving various results afterwards (e.g. plots, samples, etc.)

    # Start training
    summary, log, data_config, model_config, train_config = start_training(
        save_path=save_path,
        data_config=config['data_config'],
        model_config=config['model_config'],
        train_config=config['train_config'],
        device=device,
        test_code=args.test_code
    )

    # Save config file (for reproducability)
    config['data_config'] = data_config
    config['model_config'] = model_config
    config['train_config'] = train_config

    with open(os.path.join(save_dir, 'configs', config_file), 'w') as f:
        json.dump(config, f, indent=4)

    # Save summary file
    with open(os.path.join(save_path, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    # Save log file
    with open(os.path.join(save_path, 'log.json'), 'w') as f:
        json.dump(log, f, indent=4)

    # Save entry in master file
    summary['log_path'] = os.path.join(args.save_dir, exp_name, trial_id, 'log.json')
    master['summaries'][trial_id] = summary

    # Save master file
    with open(os.path.join(save_dir, 'master.json'), 'w') as f:
        json.dump(master, f, indent=4)
        
    if args.feature_extraction is not None:    
        # Get exp_directory
        exp_dir = os.path.join(os.getcwd(), args.save_dir, exp_name)
        
        for index, input_features in enumerate(input_feature_files):
            extract_features(exp_dir, trial_id, input_features, 
                output_feature_names[index] + "_" + config_file[:-5])
        
