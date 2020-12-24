import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from datetime import datetime
import pickle
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn import metrics
import itertools
import io
import seaborn as sns
tf.compat.v1.enable_eager_execution()
#tf.compat.v1.disable_eager_execution()
#tf.debugging.set_log_device_placement(False)

# Cross validation inpired in https://www.machinecurve.com/index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
        for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
else:
        raise SystemError("NO GPUS")

#config.set_visible_devices([], 'GPU')
print("GPUS {}", gpus)


#physical_devices = tf.config.list_physical_devices('GPU') 
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

EEG_WINDOWS=156 # this number is the average of rows that the eeg dataset used has. To calculate it, get_fold_data from utils was used
EEG_COLUMNS = 660

BATCH_SIZE = 8 # Batch size 8 seems to be the limit with my machine
EEG_SHAPE = (EEG_WINDOWS,EEG_COLUMNS)
EPOCHS =100



def get_fold_data(data_dir, fold_data, dataType, labelEncoder):
    data = fold_data.get(dataType)
    X = np.empty((len(data), EEG_WINDOWS, EEG_COLUMNS), dtype=np.float64)
    y = list()
    for i, fname in enumerate(data):
        # each file contains a named tupple
        # 'patient_id','seizure_type', 'data'
        seizure = pickle.load(open(os.path.join(data_dir, fname), "rb"))
        y.append(seizure.seizure_type)
        length = len(seizure.data)
        if(EEG_WINDOWS > length):
          X[i] = np.pad(seizure.data, ((0, EEG_WINDOWS -length), (0,0)))
        else:
          X[i] = np.resize(seizure.data, (EEG_WINDOWS, len(seizure.data[0])))
    y = labelEncoder.transform(y)
    return X, y
  
@tf.function
def resize_eeg(data, label):
  return tf.stack([data,data,data], axis=-1), label#tf.py_function(lambda: tf.convert_to_tensor(np.stack([data.numpy()]*3, axis=-1), dtype=data.dtype), label, Tout=[tf.Tensor, type(label)] )

def get_dataset(X, y):
  return tf.data.Dataset.from_tensor_slices((X, y)).map(resize_eeg, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE).prefetch(2)


def get_fold_datasets(data_dir, fold_data, le):
  X_train, y_train = get_fold_data(data_dir, fold_data, "train", le)
  X_val, y_val = get_fold_data(data_dir, fold_data, "val", le)

  print("DIMENSIONS: ",len(X_train), ",", len(X_train[0]))
    
  return get_dataset(X_train, y_train), get_dataset(X_val, y_val) 

def get_test_dataset(data_dir, fold_data, le):
  train_dataset, val_dataset = get_fold_datasets(data_dir, fold_data, le)
  return train_dataset.concatenate(val_dataset)


data_dir = "/home/david/Documents/Machine Learning/raw_data/fft_seizures_wl1_ws_0.5_sf_250_fft_min_1_fft_max_12"
cross_val_file = "../seizure-type-classification-tuh/data_preparation/cv_split_3_fold_patient_wise_v1.5.2.pkl"


def create_model():
  #train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

  data_augmentation = tf.keras.Sequential([
    
  ])

  preprocess_input = tf.keras.applications.resnet_v2.preprocess_input


  # Create the base model from the pre-trained model MobileNet V2
  base_model = tf.keras.applications.ResNet50V2(input_shape=EEG_SHAPE + (3,),
                                                include_top=False,
                                                weights='imagenet')


  base_model.trainable = False


  global_average_layer = tf.keras.layers.GlobalAveragePooling2D()


  prediction_layer = tf.keras.layers.Dense(7)


  inputs = tf.keras.Input(shape=EEG_SHAPE + (3,))
  x = data_augmentation(inputs)
  #x = preprocess_input(x) # Apparently adding it reduces performance
  x = base_model(x, training=False)
  x = global_average_layer(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  outputs = prediction_layer(x)
  model = tf.keras.Model(inputs, outputs)
  model.summary()

  base_learning_rate = 0.0001
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(10, 8))
    sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names, 
              annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.title("Confusion matrix")
    
    return figure

def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    
    buf = io.BytesIO()
    
    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format='png')
    
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    
    # Use tf.image.decode_png to convert the PNG buffer
    # to a TF image. Make sure you use 4 channels.
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    
    # Use tf.expand_dims to add the batch dimension
    image = tf.expand_dims(image, 0)
    
    return image

if __name__ == "__main__":
  # Create a TensorBoard callback
  logs_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
  tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs_dir,histogram_freq = 1,profile_batch = '490,510')  
  early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    verbose=1,
    patience=3,
    mode='min',
    restore_best_weights=True)

  file_writer_cm = tf.summary.create_file_writer(logs_dir + '/cm') # Writer for the confusion matrix

  szr_type_list = ['TNSZ', 'SPSZ', 'ABSZ', 'TCSZ', 'CPSZ', 'GNSZ', 'FNSZ']

  le = LabelBinarizer()
  le.fit(szr_type_list)

  sz = pickle.load(open(cross_val_file, "rb"))
  
  seizure_folds = list(sz.values())

  k_validation_folds = seizure_folds[:-1] # TODO: Choose which dataset will be the test randomly
  test_fold = seizure_folds[-1]

  test_dataset = get_test_dataset(data_dir, test_fold, le)    
  test_labels = np.concatenate([y for x, y in test_dataset], axis=0).argmax(axis=1)
  
  # Define per-fold score containers
  acc_per_fold = []
  loss_per_fold = []
  for fold_no, fold_data in enumerate(k_validation_folds):
    train_dataset, val_dataset = get_fold_datasets(data_dir, fold_data, le)
    
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Create the model
    model = create_model()

    # Train the model
    history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, shuffle=True, callbacks=[tboard_callback, early_stopping])
    
    # Allow GC to collect the datasets. If not, they will be available in the next iteration fo the for loop
    # and won't end up fitting in the RAM
    train_dataset = None
    val_dataset = None
    
    # evaluate the model
    print("[INFO] evaluating network...")
    scores = model.evaluate(test_dataset, verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    print("[INFO] Predicting network and creating confusion matrix...")

    test_pred = np.argmax(model.predict(test_dataset), axis=1)
    
    cm = tf.math.confusion_matrix(test_labels, test_pred)
    figure = plot_confusion_matrix(cm, class_names=le.classes_)
    cn_image =plot_to_image(figure)
     # Log the confusion matrix as an image summary.
    with file_writer_cm.as_default():
      tf.summary.image(f"Confusion Matrix Fold {fold_no}", cn_image, step=0)
    
    
    
  # == Provide average scores ==
  print('------------------------------------------------------------------------')
  print('Score per fold')
  for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
  print('------------------------------------------------------------------------')
  print('Average scores for all folds:')
  print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
  print(f'> Loss: {np.mean(loss_per_fold)}')
  print('------------------------------------------------------------------------')