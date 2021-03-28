import argparse
import json
import os

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

# Change working directory to access the trained encodings in "save"
os.chdir('../../')

# For faster runs, can modify this just to run on a single training data amount
train_amount_list = [1,2,5,10, 25, 50, 75, 100]
# For faster runs, can reduce the number of repeats. Each value in the repeat corresponds
# to the random sample selection index. For example, [1,1,1] corresponds to repeating
# sample index 1 three times, and [1,2,3] runs each random sample once.
repeats = [1,1,1,2,2,2,3,3,3]


parser = argparse.ArgumentParser()
parser.add_argument('--encodings', type=str,
                    required=False, default=None,
                    help='name of encodings to use')
                    
parser.add_argument('--input_type', type=str,
                    required=True, default=None,
                    help='whether we use pose or features')
                    
                    
parser.add_argument('--log_name', type=str,
                    required=True, default=None,
                    help='where to record the training logs')
                    
parser.add_argument('--model_name', type=str,
                    required=True, default=None,
                    help='what to name the trained classifier models')
  
args = parser.parse_args()
                    
input_type = args.input_type
add_encodings =  args.encodings


model_name_source = args.model_name

save_path_source = "downstream_tasks/fly_classification/"

os.makedirs(os.path.join(save_path_source, 'classifier_logs'), exist_ok = True)
log_file_name = "downstream_tasks/fly_classification/classifier_logs/" + args.log_name


import datetime

def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    return datetime.datetime.now().strftime(fmt).format(fname=fname)
log_file = open(log_file_name, "a")
log_file.write(timeStamped("---------New Log---------") + '\n')
log_file.write("input_type: " + input_type + '\n')
log_file.write("add_encodings: " + str(add_encodings) + '\n')
log_file.write("train_amount_list: " + str(train_amount_list) + '\n')
log_file.write("repeats: " + str(repeats) + '\n')
log_file.write("save_path: " + str(save_path_source) + '\n')
log_file.close()


import numpy as np

def load_data_splits(filename, feature_key):
    all_data = np.load(filename, allow_pickle=True)
    return all_data[feature_key], all_data['annotations']  

# Loop through training amounts
for train_amount in train_amount_list:

  log_file = open(log_file_name, "a")
  log_file.write(str(input_type) + "," + str(train_amount) + ", repeats: " + str(repeats) + "\n")
  log_file.write(save_path_source + ', ' + str(train_amount) + ',' +model_name_source + '\n')

  map_repeat_list = []

  # Loop through repeats
  for repeat in repeats:

    print("Model for " + input_type + ", train amount " + str(train_amount) + ", repeat " + str(repeat))

    # Read in either keypoints/pose or hand-designed features.
    if input_type == 'pose':
      print("READ POSES")
      train_raw_features, train_raw_annotations = load_data_splits('util/datasets/fly_v1/data/fly_train_keypoints.npz', feature_key = 'keypoints')
      val_raw_features, val_raw_annotations = load_data_splits('util/datasets/fly_v1/data/fly_val_keypoints.npz', feature_key = 'keypoints')
      test_raw_features, test_raw_annotations = load_data_splits('util/datasets/fly_v1/data/fly_test_keypoints.npz', feature_key = 'keypoints')
    elif input_type == 'features':
      print("READ FEATURES")
      train_raw_features, train_raw_annotations = load_data_splits('util/datasets/fly_v1/data/fly_train_features.npz', feature_key = 'features')
      val_raw_features, val_raw_annotations = load_data_splits('util/datasets/fly_v1/data/fly_val_features.npz', feature_key = 'features')
      test_raw_features, test_raw_annotations = load_data_splits('util/datasets/fly_v1/data/fly_test_features.npz', feature_key = 'features')  


    def read_feature_files(path_name):
      train_raw_features = []

      file = "saved/"+path_name+"/run_1/fly_train_run_1.npz"
      data = np.load(file, allow_pickle = True)
      train_raw_features = data["arr_0"]


      file = "saved/"+path_name+"/run_1/fly_val_run_1.npz"
      data = np.load(file, allow_pickle = True)
      val_raw_features = data["arr_0"]

      file = "saved/"+path_name+"/run_1/fly_test_run_1.npz"
      data = np.load(file, allow_pickle = True)
      test_raw_features = data["arr_0"]

      return train_raw_features, val_raw_features, test_raw_features


    # ADD ENCODINGS, otherwise use keypoints or features only.
    if add_encodings is not None:

      train_raw_features_1, val_raw_features_1, test_raw_features_1 = read_feature_files(add_encodings)
      
      cut_train_features = []
      curr_counter = 0
      for i, item in enumerate(train_raw_features):
        next_counter = curr_counter + item.shape[0]

        curr_pose = np.array(train_raw_features[i]).reshape((train_raw_features[i].shape[0], -1))

        cut_train_features.append(np.concatenate([curr_pose, train_raw_features_1[curr_counter:next_counter, :]], axis = -1))
        curr_counter = next_counter
      train_raw_features = np.array(cut_train_features)

      cut_val_features = []
      curr_counter = 0
      for i, item in enumerate(val_raw_features):
        next_counter = curr_counter + item.shape[0]
        curr_pose = np.array(val_raw_features[i]).reshape((val_raw_features[i].shape[0], -1))

        cut_val_features.append(np.concatenate([curr_pose, val_raw_features_1[curr_counter:next_counter, :]], axis = -1))
        curr_counter = next_counter
      val_raw_features = np.array(cut_val_features)


      cut_test_features = []
      curr_counter = 0
      for i, item in enumerate(test_raw_features):
        next_counter = curr_counter + item.shape[0]
        curr_pose = np.array(test_raw_features[i]).reshape((test_raw_features[i].shape[0], -1))

        cut_test_features.append(np.concatenate([curr_pose, test_raw_features_1[curr_counter:next_counter, :]], axis = -1))
        curr_counter = next_counter
      test_raw_features = np.array(cut_test_features)

    # Process input features
    def process_input_features(raw_features):
        processed_features = []

        for item in raw_features:

          item = np.array(item)
          seq_len = item.shape[0]
          item = item.reshape((seq_len, -1))

          item = np.nan_to_num(item, 0.0)
          item = np.clip(item, a_min = -1.5e5, a_max = 1.5e5)

          if len(processed_features) == 0:
            processed_features = item
          else:
            processed_features = np.concatenate([processed_features, item], axis = 0)

        return processed_features

    train_features = process_input_features(train_raw_features)
    val_features = process_input_features(val_raw_features)
    test_features = process_input_features(test_raw_features)    

    # 0: Lunge
    # 1: Wing Threat
    # 2: Tussle
    # 3: Wing Extension
    # 4: Circle
    # 5: Copulation
    map_results_list = []

    # Loop through all 6 classes
    for class_index in range(6):
      print('training index ' + str(class_index))
      def parse_annotations(raw_annotations):

        annotation_list = np.stack(raw_annotations)
        return annotation_list


      train_onehot = parse_annotations(train_raw_annotations[class_index])
      val_onehot = parse_annotations(val_raw_annotations[class_index])
      test_onehot = parse_annotations(test_raw_annotations[class_index])

      from sklearn.preprocessing import OneHotEncoder
      enc = OneHotEncoder(handle_unknown='ignore')
      enc.fit(train_onehot)
      train_onehot = enc.transform(train_onehot).toarray()
      val_onehot = enc.transform(val_onehot).toarray()
      test_onehot = enc.transform(test_onehot).toarray()

      train_mean = train_features.mean(axis = 0)
      train_std = train_features.std(axis = 0)

      std_epsilon = 0.001

      # Normalize by training mean and standard deviation.
      normed_train_features = ((train_features - train_mean)[:, train_std > std_epsilon] / train_std[train_std > std_epsilon])
      normed_val_features = ((val_features - train_mean)[:, train_std > std_epsilon] / train_std[train_std > std_epsilon])
      normed_test_features = ((test_features - train_mean)[:, train_std > std_epsilon] / train_std[train_std > std_epsilon])


      # Reduce training set size.
      if train_amount < 100:
        data = np.load("downstream_tasks/fly_classification/"+str(train_amount)+"/sampled_indeces_"+str(repeat) + ".npz")
        indeces_to_train = data['train_indeces']

        normed_train_features = normed_train_features[indeces_to_train]
        train_onehot = train_onehot[indeces_to_train]

      #Define where to save the models.
      save_path = os.path.join(save_path_source, str(train_amount))
      model_name = model_name_source + '_' + str(repeat)

      import tensorflow.keras.callbacks as kcallbacks
      import gc
      from sklearn import metrics
      import os

      if 'model' in globals():
        del model

      gc.collect()

      # Garbage collection callback.
      class GCCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
          gc.collect()

      # Define model size based on amount of input data.
      if train_amount > 45:
        def build_model():
          model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=[normed_train_features.shape[-1]]),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation = "relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(2, activation = "softmax")
          ])
          return model
      elif train_amount <= 45 and train_amount >=10:
        def build_model():
          model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=[normed_train_features.shape[-1]]),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(16, activation = "relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(2, activation = "softmax")
          ])
          return model
      elif train_amount < 10:
        def build_model():
          model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=[normed_train_features.shape[-1]]),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(16, activation = "relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(2, activation = "softmax")
          ])
          return model


      model = build_model()

      optimizer = tf.keras.optimizers.Adam(0.001, clipvalue = 0.5)


      def getBestModel(save_path):
          model.compile(loss='categorical_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'])

          best_weights_filepath = save_path +'/' +model_name+'_class_'+str(class_index)+'_weights.h5'
          earlyStopping = kcallbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0, patience=10, verbose=1, mode='max') 

          saveBestModel = kcallbacks.ModelCheckpoint(best_weights_filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

          # Train model.
          history = model.fit(
            normed_train_features, train_onehot, epochs=500, verbose=0, 
            validation_data = (normed_val_features, val_onehot), batch_size = 512, shuffle = True, 
            callbacks=[earlyStopping, saveBestModel, GCCallback()])

          # Reload best weights.
          model.load_weights(best_weights_filepath)

          model.save(save_path +'/' +model_name+'_class_'+str(class_index)+'.h5') 
          return model

      getBestModel(save_path)


      from sklearn import metrics
      y_pred_raw = model.predict(normed_test_features)

      map_score = metrics.average_precision_score(test_onehot, y_pred_raw, average = None)

      map_results_list.append(map_score[1])

    #print(map_results_list, file = log_file)
    # Append MAP over the classes.
    map_repeat_list.append(np.mean(map_results_list))
  
  # Write the mean and standard dev to file.
  print(str(np.mean(map_repeat_list)) + ',' + str(np.std(map_repeat_list)), file = log_file)

  log_file.close()

