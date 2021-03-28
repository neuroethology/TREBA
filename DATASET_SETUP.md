# How to Setup New Data for TREBA

This section describes the steps if you would like to try TREBA on your own data.

1. Preprocess your trajectory data into trajectories of length `T`. For our sample fly dataset, `T=21` but you can set it to any length that is able to capture some unit of behavior in your dataset.
2. Create a new directory in `util/datasets` called DATASET_NAME, where DATASET_NAME is the name of your dataset. Inside `util/datasets/DATASET_NAME`, create files `__init__.py` and `core.py` following the example we provide for the [fly dataset](util/datasets/fly_v1). ``core.py`` defines the states and actions for your dataset. Usually, the states are set as the detected keypoints for your dataset, and the actions are set as the change in state in one timestamp, but you should adapt this to your domain.
If you are using task programming/label functions to supervise TREBA, also create a folder called `label_functions`. If you are using trajectory augmentations, create a folder called `augmentations`. Note that programs and augmentations are optional for training TREBA, but if you would like to use them, here are additional notes:
     * The `label_functions` directory should contain `__init__.py` and `heuristic.py`. The programs/label functions will be coded in `heuristic.py`. See the [fly programs](util/datasets/fly_v1/label_functions/heuristics.py) for examples on how to create your own programs.
     * The `augmentations` directory should contain `__init__.py` and `augmentation_functions.py`. See the See the [fly augmentations](util/datasets/fly_v1/augmentations/augmentation_functions.py) for examples on how to create your own trajectory augmentations.
3. Currently, the code supports datasets where action(t) = state(t+1) - state(t). However, if your actions are not defined as the change in state, you need to modify the `generate_rollout` function in treba model so that future states are properly updated when given current state and action.
4. Update `util/datasets/__init__.py` with your new dataset.
5. Create a config directory and file for your custom dataset following the instructions in [config_setup](CONFIG_SETUP.md). 
6. Train TREBA:
```
python run_single.py \
-d 0 \
--config_dir CONFIG_DIR_NAME
```
7. Apply TREBA to your downstream tasks! We provide feature extraction scripts (feature_extraction.py) as well as instructions on running TREBA for [behavior classification](README.md).
