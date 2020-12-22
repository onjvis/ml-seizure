import argparse
from util import get_fold_data, keras_model_memory_usage_in_bytes
import matplotlib.pyplot as plt
import pickle
import sparse, numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelBinarizer
from tensorflow import debugging, test, config
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.applications import ResNet50, ResNet50V2, MobileNet, VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras import Input
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
debugging.set_log_device_placement(False)

device_name = test.gpu_device_name()
if not device_name:
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

gpus = config.experimental.list_physical_devices('GPU')
if gpus:
        for gpu in gpus:
                config.experimental.set_memory_growth(gpu, True)
else:
        raise SystemError("NO GPUS")

#config.set_visible_devices([], 'GPU')
print("GPUS {}", gpus)

NUM_CLASSES = 7


RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'
resnet_weights_path = './weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


# Common accuracy metric for all outputs, but can use different metrics for different output
LOSS_METRICS = ['accuracy']

# EARLY_STOP_PATIENCE must be < NUM_EPOCHS
NUM_EPOCHS = 10
EARLY_STOP_PATIENCE = 3

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# Training images processed in each step would be no.-of-train-images / STEPS_PER_EPOCH_TRAINING
STEPS_PER_EPOCH_TRAINING = 10
STEPS_PER_EPOCH_VALIDATION = 10

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# NOTE that these BATCH* are for Keras ImageDataGenerator batching to fill epoch step input
BATCH_SIZE_TRAINING = 100
BATCH_SIZE_VALIDATION = 100

# Using 1 to easily manage mapping between test_generator & prediction for submission preparation
BATCH_SIZE_TESTING = 1



def create_model():
        
        model = Sequential()

        # 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
        # NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
        eeg_input = Input(shape=(32,660, 1))
        img_conc = Concatenate()([eeg_input, eeg_input,eeg_input])
        #model.add(ResNet50(include_top = False, input_tensor=img_conc, pooling = RESNET50_POOLING_AVERAGE, weights = resnet_weights_path, input_shape=(32,660,1)))
        #model.add(ResNet50V2(include_top = False, input_tensor=img_conc, pooling = RESNET50_POOLING_AVERAGE, input_shape=(32,660,3)))

        #model.add(VGG16(include_top = False, input_tensor=img_conc, pooling = RESNET50_POOLING_AVERAGE, input_shape=(32,660,3)))

        model.add(MobileNet(include_top = False, input_tensor=img_conc, pooling = RESNET50_POOLING_AVERAGE, input_shape=(96,660,3)))

        # 2nd layer as Dense for 2-class classification, i.e., dog or cat using SoftMax activation
        model.add(Dense(NUM_CLASSES, activation = DENSE_LAYER_ACTIVATION))

        # Say not to train first layer (ResNet) model as it is already trained
        model.layers[0].trainable = False
        print
        model.summary()

        from tensorflow.keras import optimizers

        #sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
        optimizer = optimizers.Adam(lr=0.001)
        model.compile(optimizer = optimizer, loss = OBJECTIVE_FUNCTION, metrics = LOSS_METRICS)
        return model

def logDataInfo(X_train, y_trin):
        print("X_instances: ", len(X_train))
        print("X_shape: ", X_train.data.shape )
        print("y_instances: ", len(y_train))
        print("y_shape: ", y_train.data.shape)

if __name__ == "__main__":
        # construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-c", "--cross_val_file", 
                help="Pkl cross validation file", required=True)
        ap.add_argument("-d", "--data_dir", 
                help="Folder containing all the preprocessed data" , required=True)
        ap.add_argument("-m", "--model", required=True,
                help="path to output trained model")
        ap.add_argument("-l", "--label-bin", required=True,
                help="path to output label binarizer")
        ap.add_argument("-p", "--plot", required=True,
                help="path to output accuracy/loss plot")
        args = vars(ap.parse_args())

        cross_val_file = args["cross_val_file"]
        data_dir = args["data_dir"]
        

        model = create_model()
        wait = input("Press Enter to continue.")
        ################## LOAD DATA ##########################

        # Create a TensorBoard callback
        logs_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

        tboard_callback = TensorBoard(log_dir = logs_dir,histogram_freq = 1)

        # Early stopping & checkpointing the best model in ../working dir & restoring that as our model for prediction

        cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = EARLY_STOP_PATIENCE)
        cb_checkpointer = ModelCheckpoint(filepath = '../working/best.hdf5', monitor = 'val_loss', save_best_only = True, mode = 'auto')

        sz = pickle.load(open(cross_val_file, "rb"))

        szr_type_list = ['TNSZ', 'SPSZ', 'ABSZ', 'TCSZ', 'CPSZ', 'GNSZ', 'FNSZ']
        le = LabelBinarizer()
        le.fit(szr_type_list)

        results = []
        original_labels = []
        predicted_labels = []
        print("ESTIMATED TOTAL MEMORY: ", keras_model_memory_usage_in_bytes(model,batch_size=1))
        from tensorflow import data
        # Iterate through the folds
        for i, fold_data in enumerate(sz.values()):
                # fold_data is a dictionary with train and val keys
                # Each contains a list of name of files

                X_train, y_train = get_fold_data(data_dir, fold_data, "train", le, 3)
                print(len(X_train), ",", len(X_train[0]))
                X_train = np.asarray(X_train)
                y_train = np.asarray(y_train)
                logDataInfo(X_train, y_train)
                print("SIZE X:", X_train.nbytes)
               # train_data = data.Dataset.from_tensor_slices((X_train, y_train)).batch(1)
                H = model.fit(x=X_train, y=y_train, batch_size=1, epochs = NUM_EPOCHS,callbacks = [tboard_callback], verbose = 1)
                
                ################################### PREDICT #################################
                X_test, y_test = get_fold_data(data_dir, fold_data, "val", le, 3)
                # evaluate the network
                print("[INFO] evaluating network...")
                predictions = model.predict(x=X_test, batch_size=32)
                print(classification_report(y_test.argmax(axis=1),
                predictions.argmax(axis=1), target_names=le.classes_))
                # plot the training loss and accuracy
                N = np.arange(0, NUM_EPOCHS)
                plt.style.use("ggplot")
                plt.figure()
                plt.plot(N, H.history["loss"], label="train_loss")
                plt.plot(N, H.history["val_loss"], label="val_loss")
                plt.plot(N, H.history["accuracy"], label="train_acc")
                plt.plot(N, H.history["val_accuracy"], label="val_acc")
                plt.title("Training Loss and Accuracy (Simple NN)")
                plt.xlabel("Epoch #")
                plt.ylabel("Loss/Accuracy")
                plt.legend()
                plt.savefig(args["plot"])
                exit()





fit_history = model.fit(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH_TRAINING,
        epochs = NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=STEPS_PER_EPOCH_VALIDATION,
        callbacks=[cb_checkpointer, cb_early_stopper]
)
model.load_weights("../working/best.hdf5")


#tensorboard --logdir logs/fit