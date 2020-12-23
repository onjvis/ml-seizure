import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pickle
from util import get_fold_data

from tensorflow.keras.preprocessing import image_dataset_from_directory
tf.compat.v1.enable_eager_execution()
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 8
IMG_SIZE = (156,660)

data_dir = "/home/david/Documents/Machine Learning/raw_data/fft_seizures_wl1_ws_0.5_sf_250_fft_min_1_fft_max_12"
cross_val_file = "../seizure-type-classification-tuh/data_preparation/cv_split_3_fold_patient_wise_v1.5.2.pkl"

sz = pickle.load(open(cross_val_file, "rb"))
fold_data = sz.values()
fold_data = list(fold_data)[0]
# fold_data is a dictionary with train and val keys
# Each contains a list of name of files
from sklearn.preprocessing import LabelBinarizer
szr_type_list = ['TNSZ', 'SPSZ', 'ABSZ', 'TCSZ', 'CPSZ', 'GNSZ', 'FNSZ']

le = LabelBinarizer()
le.fit(szr_type_list)


X_train, y_train = get_fold_data(data_dir, fold_data, "train", le, 3)
print(len(X_train), ",", len(X_train[0]))
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
print("SIZE X:", X_train.nbytes)

@tf.function
def resize_eeg(data, label):
  return tf.stack([data,data,data], axis=-1), label#tf.py_function(lambda: tf.convert_to_tensor(np.stack([data.numpy()]*3, axis=-1), dtype=data.dtype), label, Tout=[tf.Tensor, type(label)] )
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).map(resize_eeg).batch(BATCH_SIZE)
      
AUTOTUNE = tf.data.experimental.AUTOTUNE

#train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
  
])

preprocess_input = tf.keras.applications.resnet_v2.preprocess_input


# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.ResNet50V2(input_shape=IMG_SIZE + (3,),
                                               include_top=False,
                                               weights='imagenet')


base_model.trainable = False


global_average_layer = tf.keras.layers.GlobalAveragePooling2D()


prediction_layer = tf.keras.layers.Dense(7)


inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
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

initial_epochs = 10

#loss0, accuracy0 = model.evaluate(validation_dataset)

#print("initial loss: {:.2f}".format(loss0))
#print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_dataset,
                    epochs=initial_epochs)
