import os
import numpy as np

from keras import backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
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
import os
import glob
from datetime import datetime
#import warnings
#warnings.simplefilter('ignore')
from scipy.misc import toimage
import scipy as sp
import scipy.ndimage
import numpy as np
import pandas as pd
import skimage
import skimage.exposure
import mahotas as mh
from sklearn.cross_validation import KFold
from PIL import Image
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm_notebook
from IPython.display import display
#from dual_IDG import DualImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, \
    Convolution2D, MaxPooling2D, ZeroPadding2D, Input, Embedding, LSTM, merge, \
    Lambda, UpSampling2D, Deconvolution2D, Cropping2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import imhandle as imhand

''' IMPORT YOUR SCRIPT FILE HERE TO CREATE YOUR MODEL LATER '''


''' BACKEND must be TENSORFLOW

This is a script to convert Theano models (Theano Backend, TH dim ordering)
to the other possible backend / dim ordering combinations.

Given weights and model for TH-kernels-TH-dim-ordering, produces a folder with
- TH-kernels-TF-dim-ordering
- TF-kernels-TH-dim-ordering
- TF-kernels-TF-dim-ordering

Needs 3 important inputs:

1) Theano model (model with TH dim ordering)
2) Tensorflow model (model with TF dim ordering)
3) Weight file for Theano model (theano-kernels-th-dim-ordering)

Supports : Multiple weights for same model (auto converts different weights for same model)

Usage:
1) Place script in the same directory as the weight file directory. If you want to place somewhere
   else, then you must provide absolute path to the weight files below instead of relative paths.

2) Edit the script to create your model :
    a) Import your model building script above (in the imports section)
    b) Set `th_dim_model` = ... (create your th dim model here and set it to th_dim_model)
    c) Set `tf_dim_model` = ... (create your tf dim model here and set it to tf_dim_model)
    d) Add the path to the weight files in `model_weights`.
       Note : The weight files must be for the Theano model (theano kernels, th dim ordering)

3) Run the script.

4) Use the weight files in the created folders : ["tf-kernels-tf-dim/", "tf-kernels-th-dim/", "th-kernels-tf-dim/"]
'''

def get_model_tf(img_rows=256, img_cols=256, channels=3):
  inputs = Input(shape=(img_rows, img_cols, channels))
  conv1 = Convolution2D(32, (3, 3), activation='relu', border_mode='same')(inputs)
  conv1 = Dropout(0.3)(conv1)
  conv1 = Convolution2D(32, (3, 3), activation='relu', border_mode='same')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2), dim_ordering="tf")(conv1)

  conv2 = Convolution2D(64, (3, 3), activation='relu', border_mode='same')(pool1)
  conv2 = Dropout(0.3)(conv2)
  conv2 = Convolution2D(64, (3, 3), activation='relu', border_mode='same')(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2), dim_ordering="tf")(conv2)

  conv3 = Convolution2D(64, (3, 3), activation='relu', border_mode='same')(pool2)
  conv3 = Dropout(0.3)(conv3)
  conv3 = Convolution2D(64, (3, 3), activation='relu', border_mode='same')(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2), dim_ordering="tf")(conv3)

  conv4 = Convolution2D(64, (3, 3), activation='relu', border_mode='same')(pool3)
  conv4 = Dropout(0.3)(conv4)
  conv4 = Convolution2D(64, (3, 3), activation='relu', border_mode='same')(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2), dim_ordering="tf")(conv4)

  conv5 = Convolution2D(64, (3, 3), activation='relu', border_mode='same')(pool4)
  conv5 = Dropout(0.3)(conv5)
  conv5 = Convolution2D(64, (3, 3), activation='relu', border_mode='same')(conv5)

  up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=-1)
  conv6 = Convolution2D(64, (3, 3), activation='relu', border_mode='same')(up6)
  conv6 = Dropout(0.3)(conv6)
  conv6 = Convolution2D(64, (3, 3), activation='relu', border_mode='same')(conv6)

  up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=-1)
  conv7 = Convolution2D(64, (3, 3), activation='relu', border_mode='same')(up7)
  conv7 = Dropout(0.3)(conv7)
  conv7 = Convolution2D(64, (3, 3), activation='relu', border_mode='same')(conv7)

  up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=-1)
  conv8 = Convolution2D(64, (3, 3), activation='relu', border_mode='same')(up8)
  conv8 = Dropout(0.3)(conv8)
  conv8 = Convolution2D(64, (3, 3), activation='relu', border_mode='same')(conv8)

  up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=-1)
  conv9 = Convolution2D(32, (3, 3), activation='relu', border_mode='same')(up9)
  conv9 = Dropout(0.3)(conv9)
  conv9 = Convolution2D(32, (3, 3), activation='relu', border_mode='same')(conv9)

  conv10 = Convolution2D(1, (1, 1), activation='sigmoid', border_mode='same')(conv9)
  #conv10 = Flatten()(conv10)

  model = Model(input=inputs, output=conv10)

  model.summary()

  return model


def get_model_th(img_rows=256, img_cols=256):
  inputs = Input((3, img_rows, img_cols))
  conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
  conv1 = Dropout(0.3)(conv1)
  conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2), dim_ordering="th")(conv1)

  conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
  conv2 = Dropout(0.3)(conv2)
  conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2), dim_ordering="th")(conv2)

  conv3 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool2)
  conv3 = Dropout(0.3)(conv3)
  conv3 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2), dim_ordering="th")(conv3)

  conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool3)
  conv4 = Dropout(0.3)(conv4)
  conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2), dim_ordering="th")(conv4)

  conv5 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool4)
  conv5 = Dropout(0.3)(conv5)
  conv5 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv5)

  up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
  conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up6)
  conv6 = Dropout(0.3)(conv6)
  conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv6)

  up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
  conv7 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up7)
  conv7 = Dropout(0.3)(conv7)
  conv7 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv7)

  up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
  conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
  conv8 = Dropout(0.3)(conv8)
  conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

  up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
  conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
  conv9 = Dropout(0.3)(conv9)
  conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

  conv10 = Convolution2D(1, 1, 1, activation='sigmoid', border_mode='same')(conv9)
  #conv10 = Flatten()(conv10)

  model = Model(input=inputs, output=conv10)

  return model


K.set_image_data_format('channels_first')
th_dim_model = get_model_th(img_rows=256, img_cols=256) # Create your theano model here with TH dim ordering

K.set_image_data_format('channels_last')
tf_dim_model = get_model_tf(img_rows=256, img_cols=256, channels=3) # Create your tensorflow model with TF dimordering here

model_weights = ['rim-one-second.hdf5'] # Add names of theano model weight file paths here.
                     # These weights are assumed to be for  theano backend
                     # (th kernels) with th dim ordering!

"""

No need to edit anything below this. Simply run the script now after
editing the above 3 inputs.

"""


def shuffle_rows(original_w, nb_last_conv, nb_rows_dense):
    ''' Note :
    This algorithm to shuffle dense layer rows was provided by Kent Sommers (@kentsommer)
    in a gist : https://gist.github.com/kentsommer/e872f65926f1a607b94c2b464a63d0d3
    '''
    converted_w = np.zeros(original_w.shape)
    count = 0
    for index in range(original_w.shape[0]):
        if (index % nb_last_conv) == 0 and index != 0:
            count += 1
        new_index = ((index % nb_last_conv) * nb_rows_dense) + count
        print("index from " + str(index) + " -> " + str(new_index))
        converted_w[index] = original_w[new_index]

    return converted_w


first_dense = True
nb_last_conv = 0

for dirpath in ["tf-kernels-channels-last-dim-ordering/", "tf-kernels-channels-first-dim-ordering/", "th-kernels-channels-last-dim-ordering/"]:
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

# Converts (theano kernels, th dim ordering) to (tensorflow kernels, th dim ordering)
K.set_image_dim_ordering('tf')
for weight_fn in model_weights:
    th_dim_model.load_weights(weight_fn)
    convert_all_kernels_in_model(th_dim_model)

    th_dim_model.save_weights("tf-kernels-channels-first-dim-ordering/%s" % weight_fn, overwrite=True)
    print("Done tf-kernels-channels-first-dim-ordering %s" % weight_fn)


# Converts (theano kernels, th dim ordering) to (tensorflow kernels, tf dim ordering)
K.set_image_dim_ordering('th')
for weight_fn in model_weights:
    th_dim_model.load_weights(weight_fn) # th-kernels-th-dim
    convert_all_kernels_in_model(th_dim_model) # tf-kernels-th-dim

    count_dense = 0
    for layer in th_dim_model.layers:
        if layer.__class__.__name__ == "Dense":
            count_dense += 1

    if count_dense == 1:
        first_dense = False # If there is only 1 dense, no need to perform row shuffle in Dense layer

    print("Nb layers : ", len(th_dim_model.layers))

    for index, th_layer in enumerate(th_dim_model.layers):
        if th_layer.__class__.__name__ in ['Conv1D',
                                           'Conv2D',
                                           'Conv3D',
                                           'AtrousConvolution1D'
                                           'AtrousConvolution2D',
                                           'Conv2DTranspose',
                                           'SeparableConv2D',
                                           'DepthwiseConv2D',
                                           ]:
            weights = th_layer.get_weights() # tf-kernels-th-dim
            weights[0] = weights[0].transpose((2, 3, 1, 0))
            tf_dim_model.layers[index].set_weights(weights) # tf-kernels-tf-dim

            nb_last_conv = th_layer.nb_filter # preserve last number of convolutions to use with dense layers
            print("Converted layer %d : %s" % (index + 1, th_layer.name))
        else:
            if th_layer.__class__.__name__ == "Dense" and first_dense:
                weights = th_layer.get_weights()
                nb_rows_dense_layer = weights[0].shape[0] // nb_last_conv

                print("Magic Number 1 : ", nb_last_conv)
                print("Magic nunber 2 : ", nb_rows_dense_layer)

                weights[0] = shuffle_rows(weights[0], nb_last_conv, nb_rows_dense_layer)
                tf_dim_model.layers[index].set_weights(weights)

                first_dense = False
                print("Shuffled Dense Weights layer and saved %d : %s" % (index + 1, th_layer.name))
            else:
                tf_dim_model.layers[index].set_weights(th_layer.get_weights())
                print("Saved layer %d : %s" % (index + 1, th_layer.name))


    tf_dim_model.save_weights("tf-kernels-channels-last-dim-ordering/%s" % weight_fn, overwrite=True)
    print("Done tf-kernels-channels-last-dim-ordering %s" % weight_fn)


# Converts (theano kernels, th dim ordering) to (theano kernels, tf dim ordering)
for weight_fn in model_weights:
    th_dim_model.load_weights(weight_fn)

    for index, th_layer in enumerate(th_dim_model.layers):
        if th_layer.__class__.__name__ in ['Conv1D',
                                           'Conv2D',
                                           'Conv3D',
                                           'AtrousConvolution1D'
                                           'AtrousConvolution2D',
                                           'Conv2DTranspose',
                                           'SeparableConv2D',
                                           'DepthwiseConv2D',
                                           ]:
            weights = th_layer.get_weights()
            weights[0] = weights[0].transpose((2, 3, 1, 0))
            tf_dim_model.layers[index].set_weights(weights)
        else:
            tf_dim_model.layers[index].set_weights(th_layer.get_weights())

        print("Changed dim %d : %s" % (index + 1, th_layer.name))

    tf_dim_model.save_weights("th-kernels-channels-last-dim-ordering/%s" % weight_fn, overwrite=True)
    print("Done th-kernels-channels-last-dim-ordering %s" % weight_fn)