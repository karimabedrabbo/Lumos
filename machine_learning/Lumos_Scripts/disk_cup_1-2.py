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

K.set_image_dim_ordering('tf')



def mean_IOU_cpu_for_binary(X, Y):
    """Computes mean Intersection-over-Union (IOU) for two arrays of binary images.
    Assuming X and Y are of shape (n_images, w, h)."""
    
    intersection, union = np.logical_and(X, Y).sum(axis=(1, 2, 3)), np.logical_or(X, Y).sum(axis=(1, 2, 3))
    # if union == 0, it follows that intersection == 0 => score should be 0.
    union = np.where(union == 0, 1, union)
    return np.mean(intersection / union.astype(np.float64))


def mean_IOU_cpu(X, Y):
    X_fl = X > 0.5
    Y_fl = Y > 0.5
    
    intersection = K.sum(X_fl * Y_fl, axis=1)
    union = K.sum(K.maximum(X_fl, Y_fl), axis=1)
    # if union == 0, it follows that intersection == 0 => score should be 0.
    union = K.switch(K.equal(union, 0), 1, union)
    return K.mean(intersection / K.cast(union, 'float32'))

def mean_IOU_gpu(X, Y):
    """Computes mean Intersection-over-Union (IOU) for two arrays of binary images.
    Assuming X and Y are of shape (n_images, w, h)."""
    
    #X_fl = K.clip(K.batch_flatten(X), K.epsilon(), 1.)
    #Y_fl = K.clip(K.batch_flatten(Y), K.epsilon(), 1.)
    X_fl = K.clip(K.batch_flatten(X), 0., 1.)
    Y_fl = K.clip(K.batch_flatten(Y), 0., 1.)
    X_fl = K.greater(X_fl, 0.5)
    Y_fl = K.greater(Y_fl, 0.5)
    
    intersection = K.sum(X_fl * Y_fl, axis=1)
    union = K.sum(K.maximum(X_fl, Y_fl), axis=1)
    # if union == 0, it follows that intersection == 0 => score should be 0.
    union = K.switch(K.equal(union, 0), 1, union)
    return K.mean(intersection / K.cast(union, 'float32'))


def mean_IOU_gpu_for_flattened(x, y):
    x = K.clip(x, K.epsilon(), 1.)
    y = K.clip(y, K.epsilon(), 1.)
    x = K.greater(x, 0.5)
    y = K.greater(y, 0.5)
    
    intersection = K.sum(x * y, axis=1)
    union = K.sum(K.maximum(x, y), axis=1)
    # if union == 0, it follows that intersection == 0 => score should be 0.
    union = K.switch(K.equal(union, 0), 1, union)
    return K.mean(intersection / K.cast(union, 'float32'))


def mean_IOU_gpu_loss(X, Y):
    return -mean_IOU_gpu(X, Y)

def bce_gpu(y_true, y_pred):
    # Credits: https://www.kaggle.com/c/ultrasound-nerve-segmentation/forums/t/22951/would-like-to-see-how-others-approached-this-problem-doesn-t-have-to-be/131719
    
    # Workaround for shape bug.
    #y_true.set_shape(y_pred.get_shape())

    # Without K.clip, K.sum() behaves differently when compared to np.count_nonzero()
    y_true_f = K.clip(K.batch_flatten(y_true), K.epsilon(), 1.)
    y_pred_f = K.clip(K.batch_flatten(y_pred), K.epsilon(), 1.)
    
    bce = K.mean(K.binary_crossentropy(y_true_f, y_pred_f), axis=-1)
    return K.mean(bce)

def dice(y_true, y_pred):
    # Workaround for shape bug. For some reason y_true shape was not being set correctly
    #y_true.set_shape(y_pred.get_shape())

    # Without K.clip, K.sum() behaves differently when compared to np.count_nonzero()
    #y_true_f = K.clip(K.batch_flatten(y_true), K.epsilon(), 1.)
    #y_pred_f = K.clip(K.batch_flatten(y_pred), K.epsilon(), 1.)
    y_true_f = K.clip(K.batch_flatten(y_true), 0., 1.)
    y_pred_f = K.clip(K.batch_flatten(y_pred), 0., 1.)
    #y_pred_f = K.greater(y_pred_f, 0.5)

    intersection = 2 * K.sum(y_true_f * y_pred_f, axis=1)
    union = K.sum(y_true_f * y_true_f, axis=1) + K.sum(y_pred_f * y_pred_f, axis=1)
    return K.mean(intersection / union)


def dice_loss(y_true, y_pred):
    return -dice(y_true, y_pred)


def log_dice_loss(y_true, y_pred):
    return -K.log(dice(y_true, y_pred))


def dice_metric(y_true, y_pred):
    """An exact Dice score for binary tensors."""
    y_true_f = K.greater(y_true, 0.5)
    y_pred_f = K.greater(y_pred, 0.5)
    return dice(y_true_f, y_pred_f)

def tf_to_th_encoding(X):
    return np.rollaxis(X, 3, 1)


def th_to_tf_encoding(X):
    return np.rollaxis(X, 1, 4)

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
            img_file = scipy.misc.imresize(arr=img_file, size=(256, 256, 3))
            img_file = img_file/255.0
            img_arr = np.asarray(img_file)
            img_arr = np.expand_dims(img_arr, axis=0)
            X.append(img_arr)
            y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y
"""
X_train, y_train = get_data('Shalins_data/training/')
X_test, y_test = get_data('Shalins_data/testing/')
"""
"""
X_train, y_train = get_data('Originals_for_training/training/')
X_test, y_test = get_data('Originals_for_training/testing/')
"""


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

def get_model(img_rows=256, img_cols=256, channels=3):
    inputs = Input(shape=(img_rows, img_cols, channels))
    conv1 = Convolution2D(32, (3, 3), activation='relu', border_mode='same')(inputs)
    conv1 = Dropout(0.3)(conv1)
    conv1 = Convolution2D(32, (3, 3), activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, (3, 3), activation='relu', border_mode='same')(pool1)
    conv2 = Dropout(0.3)(conv2)
    conv2 = Convolution2D(64, (3, 3), activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(64, (3, 3), activation='relu', border_mode='same')(pool2)
    conv3 = Dropout(0.3)(conv3)
    conv3 = Convolution2D(64, (3, 3), activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(64, (3, 3), activation='relu', border_mode='same')(pool3)
    conv4 = Dropout(0.3)(conv4)
    conv4 = Convolution2D(64, (3, 3), activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

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


# drions-db.hdf5 and rim-one-second.hdf5 work the best for segmenting the cup. The rim-one-second works slightly better though
# drishti-gs only works on segmenting the cup for non cropped images, otherwise it's super inaccurate

model = get_model(img_rows=256, img_cols=256, channels=3)
model.load_weights("new_h5_files/drishti-gs.hdf5")
model.compile(optimizer=SGD(lr=3e-4, momentum=0.95),
              loss=log_dice_loss)



# img = image.load_img('Originals_for_training/training/NoGlauc55.jpg', target_size=(256, 256, 3))
img = image.load_img('single_image/13.png', target_size=(256, 256, 3))

arr = np.array(img)
arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
arr = np.stack((arr,)*3, axis=-1)
# arr = cv2.GaussianBlur(arr, (11, 11), 0)
lab= cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
limg = cv2.merge((cl,a,b))
arr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
# arr = skimage.exposure.equalize_adapthist(arr)
temp = toimage(arr)
temp.show()
arr = arr.astype('float')
arr /= 255
arr = np.expand_dims(arr, axis=0)
pred = model.predict(arr)
pred = np.reshape(pred, (256,256))
output = toimage(pred)
output.show()
output.save("prediction.jpg")





"""
print('Predicting on test data')
y_pred = np.rint(model.predict(X_test))

print(accuracy_score(y_test, y_pred))

lens_X_test, lens_y_test = get_data('normalized_data/testing_lens/')

lens_y_test = encoder.transform(lens_y_test)


print('Predicting on lens data')
lens_y_pred = np.rint(model.predict(lens_X_test))

print(accuracy_score(lens_y_test, lens_y_pred))

print(confusion_matrix(lens_y_test, lens_y_pred))

"""



















