# Setting up new configurations

The config file for launching new TREBA training runs are in the `configs` folder. We recommending making new folders for each new configuration you are running, and the runs will be saved in the `saved` directory by default.

Each directory in `configs` should contain at least 1 .json file, which will be used to configure the TREBA model. An example config file is provided [here](configs/fly_treba_original/run_1.json).

The file has three major sections:
* `data_config`: Configurations relating to the dataset, including dataset name, names of the programs/labeling functions, and augmentations used on the data.
* `model_config`: Configurations relating to the model, including model name, embedding dimensions, and loss weights.
* `train_config`: Configurations used during training, including batch size, number of epochs, and learning rate.

We describe the fields in each section below.

## Data_config Section

* `name` describe the name of the dataset to load. If you use a custom dataset, make sure that your dataset can be seen from `util/datasets/__init__.py` so that the dataset can be properly associated with the name you use in the config file.

* For each item in `labels`, it should have a `name`, corresponding to the name of the program/labeling function. This name should also be in `util/datasets/DATASET_NAME/label_functions/__init__.py`. See the [fly_v1 dataset label directory](util/datasets/fly_v1/label_functions) for an example on how to set this up. For each label, it can optionally have a `thresholds` field to divide the label into discrete classes (the current implementation of contrastive loss requires discrete classes if labels are used). If you do not want to apply any programs to train TREBA, remove the entire `labels` field.

* The `augmentations` field describe the data augmentation to apply to the input trajectories. The augmentation `name` should be in `util/datasets/DATASET_NAME/augmentations/__init__.py`. See the [fly_v1 dataset augmentations directory](util/datasets/fly_v1/augmentations) for an example on how to set this up.If  you do not want to apply any augmentations to train TREBA, remove the entire `augmentations` field.

* For a custom dataset, you may add any additional fields here to be read by the training script.

## Model_config Section

* The `name` specifies the model to use, and here we provide the TREBA model. Any additional models needs to be included in the [model init file](lib/models/__init__.py).

* `z_dim` is the learned embedding dimension by TREBA.

* `h_dim` is the hidden layer dimension used by neural network encoders and decoders in TREBA.

* `rnn_dim` is the dimension of a recurrent unit used in the decoder in TREBA. 

* `<X>_loss_weight` specifies the weight on either the consistency, contrastive, or decoding loss in TREBA. Any combination of these losses is able to train TREBA. If a loss is set to 0, that means that it is not used. See [our paper](https://arxiv.org/abs/2011.13917) for a description of each loss. If you use the consistency loss (consistency_loss_weight > 0), then you need to use staged training in the `train_config` section to first train the label approximators, then train TREBA.

* If you work with a custom model, you may add additional fields here for your model configurations.

## Train_config Section
This section includes standard training parameters, such as `batch_size`, `learning_rate`, and `num_epochs`. Note that when `num_epochs` has more than 1 item, we do staged training (stage 1 and 2 may be training different models and losses). In the sample config file, `"num_epochs": [100,200]` corresponds to first training the label approximators for 100 epochs, then training TREBA for 200 epochs. The specific losses used in each stage depends on the model setup.
