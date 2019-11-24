from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.layers import Dense
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
import cv2
import scipy
import os
from keras.models import Model
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tensorflow.python.client import device_lib
from keras import optimizers
import pickle
from six.moves import cPickle

batchsize = 16

def get_data(folder):
    """
    Load the data and labels from the given folder.
    """
    X = []
    y = []

    for image_filename in os.listdir(folder):
        img_file = cv2.imread(folder + '/' + image_filename)
        if img_file is not None:
            # Downsample the image to 120, 160, 3
            if not image_filename.startswith('NoGlauc'):
                label = 0
            else:
                label = 1
            img_file = scipy.misc.imresize(arr=img_file, size=(175, 175, 3))
            img_file = img_file/255
            img_arr = np.asarray(img_file)
            X.append(img_arr)
            y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y
"""
X_train, y_train = get_data('Shalins_data/training/')
X_test, y_test = get_data('Shalins_data/testing/')
"""
X_train, y_train = get_data('Originals_for_training/training/')
X_test, y_test = get_data('Originals_for_training/testing/')

def batch_iter(data, labels, batch_size, shuffle=True):
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    def data_generator():
        data_size = len(data)
        while True:
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
                shuffled_labels = labels[shuffle_indices]
            else:
                shuffled_data = data
                shuffled_labels = labels

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                X, y = shuffled_data[start_index: end_index], shuffled_labels[start_index: end_index]
                yield X, y

    return num_batches_per_epoch, data_generator()

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(175, 175, 3))

# add a global spatial average pooling layer
x = base_model.output

x = Conv2D(64, (3,3), activation = 'relu', input_shape=(175, 175, 3))(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.7)(x)
predictions = Dense(1, activation='sigmoid')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# train the model on the new data for a few epochs
train_steps, train_batches = batch_iter(X_train, y_train, batchsize)
valid_steps, valid_batches = batch_iter(X_test, y_test, batchsize)
model.fit_generator(train_batches, train_steps, epochs=25, validation_data=valid_batches, validation_steps=valid_steps, verbose=1)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
train_steps, train_batches = batch_iter(X_train, y_train, batchsize)
valid_steps, valid_batches = batch_iter(X_test, y_test, batchsize)
model.fit_generator(train_batches, train_steps, epochs=25, validation_data=valid_batches, validation_steps=valid_steps, verbose=1)

model.save_weights('binary_model.h5')
model.save('saved_model.h5')

print('Predicting on test data')
y_pred = np.rint(model.predict(X_test))

print(accuracy_score(y_test, y_pred))

lens_X_test, lens_y_test = get_data('normalized_data/testing_lens/')

lens_y_test = encoder.transform(lens_y_test)


print('Predicting on lens data')
lens_y_pred = np.rint(model.predict(lens_X_test))

print(accuracy_score(lens_y_test, lens_y_pred))

print(confusion_matrix(lens_y_test, lens_y_pred))