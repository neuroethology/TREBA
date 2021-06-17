import numpy as np
import os
import json
import argparse

'''
Script for converting unlabeled videos from CalMS21 .json files into .npz files for input into TREBA.
The .npz files are stacked trajectories of length N (default = 21).
In our work, we use the  unlabeled videos set (first 229 videos for train 
and the remaining 53 for validation).
'''

def sliding_window_stack(input_array, seq_length=100, sliding_window = 1):
    total = input_array.shape[0]

    return np.stack([input_array[i:total+i-seq_length+1:sliding_window] 
        for i in range(seq_length)], axis = 1)

def stack_pose_to_traj_list(input_pose, seq_length, sliding_window=1):
    """
    Cut pose list into array of seq_number x seq_length x 28.
    If sliding_window size == seq_length, there will be no overlaps in the saved trajectories.
    If sliding_window == 1, the number of trajectories will be equal to the number of input frames.
    """
    pose_list = []

    if sliding_window is None:
        sliding_window = seq_length

    for pose_value in input_pose:

        # Processing mouse trajectories.
        current_pose_list = pose_value.transpose((0, 1, 3, 2))

        current_pose_list = current_pose_list.reshape((current_pose_list.shape[0], -1))

        # Do edge padding.
        converted_padded = np.pad(current_pose_list, ((seq_length//2, 
            seq_length-1-seq_length//2), (0, 0)), mode='edge')

        cut_pose_list = sliding_window_stack(converted_padded, seq_length = seq_length, 
            sliding_window = sliding_window)

        if len(pose_list) == 0:
            pose_list = cut_pose_list
        else:
            pose_list = np.concatenate([pose_list, cut_pose_list], axis=0)

    pose_list = np.array(pose_list)
    return pose_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory', type=str, required = True, 
        help='Directory to CalMS21 json files (all calms21 json files in the directory will be used)')
    parser.add_argument('--output_path', type=str, required = True, 
        help='Path to output .npz files (in the format for training/feature extraction from TREBA)')    
    parser.add_argument('--trajectory_length', type=int, default = 21, required = False,
                        help='Length to cut trajectories into (number of frames)')
    parser.add_argument('--sliding_window', type=int, default = 21, required = False,
                        help='Sliding Window size (number of frames).' + 
                            'If sliding_sindow == trajectory_length, there will be no overlaps.')
    parser.add_argument('--data_split', type=int, default = -1, required = False,
                        help='Number of videos to split into train/val.' + 
                            'Use -1 to disable, otherwise specify the number of train vids.')
    parser.add_argument('--no_shuffle', action='store_true', required = False,
                        help='whether to shuffle the trajectories before saving.')    

    args = parser.parse_args()

    # Parse all jsons in a directory.
    json_to_parse = []
    for file in sorted(os.listdir(args.input_directory)):
        if file.endswith(".json") and 'features' not in file:
            json_to_parse.append((os.path.join(args.input_directory, file)))

    # Put all pose data from the dictionary into a list.
    input_pose_list = []

    for json_file in sorted(json_to_parse):

        print('Reading file: ' + json_file)

        with open(json_file, 'r') as fp:
            input_data = json.load(fp)

        # First key is the group name for the sequences
        for groupname in sorted(input_data.keys()):
            # Next key is the sequence id
            for sequence_id in sorted(input_data[groupname].keys()):
                input_pose_list.append(np.array(input_data[groupname][sequence_id]['keypoints']))

    if args.data_split > 0:
        assert args.data_split < len(input_pose_list)

        # Splitting into train and val set.
        processed_pose_train = stack_pose_to_traj_list(input_pose_list[:args.data_split], 
            seq_length = args.trajectory_length, sliding_window = args.sliding_window)

        print("Saving array of size: " + str(processed_pose_train.shape))
        if not args.no_shuffle:
            np.random.shuffle(processed_pose_train)

        np.savez(args.output_path + '_train', data = processed_pose_train)

        processed_pose_val = stack_pose_to_traj_list(input_pose_list[args.data_split:], 
            seq_length = args.trajectory_length, sliding_window = args.sliding_window)

        print("Saving array of size: " + str(processed_pose_val.shape))
        if not args.no_shuffle:
            np.random.shuffle(processed_pose_val)

        np.savez(args.output_path + '_val', data = processed_pose_val)        

    else:
        # Save all the videos together.
        processed_pose = stack_pose_to_traj_list(input_pose_list, 
            seq_length = args.trajectory_length, sliding_window = args.sliding_window)

        print("Saving array of size: " + str(processed_pose.shape))
        if not args.no_shuffle:
            np.random.shuffle(processed_pose)

        np.savez(args.output_path, data = processed_pose)
