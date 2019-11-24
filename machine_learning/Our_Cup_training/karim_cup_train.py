import os
import glob
import numpy as np


import keras
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Activation, concatenate
from keras.optimizers import Adam
from keras import losses
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

import cv2
import tensorflow as tf
from scipy.misc import toimage

def new_cross_entropy(y_, y_conv):
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    return cross_entropy
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def dice(y_true, y_pred):
    y_true_f = K.clip(K.batch_flatten(y_true), 0., 1.)
    y_pred_f = K.clip(K.batch_flatten(y_pred), 0., 1.)

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

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols, 3), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        arr = imgs[i]
        arr = arr.astype('float32')
        arr /= 255.
        imgs_p[i] = resize(arr, (img_cols, img_rows), preserve_range=True)
    return imgs_p

K.set_image_dim_ordering('tf')

imgs_cup_train = np.load('gt_cup_train_new.npy')
imgs_orig_train = np.load('orig_cup_train_new.npy')

def get_model():
    # input = Input(shape=(256,256,3),name = 'image_input')
    
    model_vgg19_conv = VGG19(weights='imagenet', include_top=False, input_shape=(256,256,3))
 
    # output_vgg19_conv = model_vgg19_conv(input)

    # concat = Input(shape=(256,256,3), name = 'concat')


    # conv2_3_16_zeropadding = ZeroPadding2D(padding=(1,1), name= "conv2_3_16_zeropadding", data_format="channels_last")(model_vgg19_conv.layers[-17].output)

    # conv3_3_16_zeropadding = ZeroPadding2D(padding=(1,1), name= "conv3_3_16_zeropadding", data_format="channels_last")(model_vgg19_conv.layers[-12].output)

    # conv4_3_16_zeropadding = ZeroPadding2D(padding=(1,1), name= "conv4_3_16_zeropadding", data_format="channels_last")(model_vgg19_conv.layers[-7].output)

    # conv5_3_16_zeropadding = ZeroPadding2D(padding=(1,1), name= "conv5_3_16_zeropadding", data_format="channels_last")(model_vgg19_conv.layers[-2].output)



    # conv2_3_16 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1,1), activation='linear', name= "conv2_3_16", data_format="channels_last")(conv2_3_16_zeropadding)

    # conv3_3_16 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1,1), activation='linear', name= "conv3_3_16", data_format="channels_last")(conv3_3_16_zeropadding)

    # conv4_3_16 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1,1), activation='linear', name= "conv4_3_16", data_format="channels_last")(conv4_3_16_zeropadding)

    # conv5_3_16 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1,1), activation='linear', name= "conv5_3_16", data_format="channels_last")(conv5_3_16_zeropadding)


    conv2_3_16 = Conv2D(filters=16, kernel_size=(3, 3), padding = "same", activation='linear', name= "conv2_3_16", data_format="channels_last")(model_vgg19_conv.layers[-17].output)

    conv3_3_16 = Conv2D(filters=16, kernel_size=(3, 3), padding = "same", activation='linear', name= "conv3_3_16", data_format="channels_last")(model_vgg19_conv.layers[-12].output)

    conv4_3_16 = Conv2D(filters=16, kernel_size=(3, 3), padding = "same", activation='linear', name= "conv4_3_16", data_format="channels_last")(model_vgg19_conv.layers[-7].output)

    conv5_3_16 = Conv2D(filters=16, kernel_size=(3, 3), padding = "same", activation='linear', name= "conv5_3_16", data_format="channels_last")(model_vgg19_conv.layers[-2].output)


    # upsample2_zeropadding = ZeroPadding2D(padding=(1,1), name= "upsample2_zeropadding", data_format="channels_last")(conv2_3_16)


    # new_score_weighting = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), dilation_rate=(1,1), activation='linear', name= "new_score_weighting", data_format="channels_last")(concat)


    # upsample2_ = Conv2DTranspose(filters=16, kernel_size=(4, 4), strides=(2, 2), activation='linear', name= "upsample2_", data_format="channels_last"dat)(upsample2_zeropadding)

    upsample2_ = Conv2DTranspose(filters=16, kernel_size=(4, 4), strides=(2, 2), padding = "same", activation='linear', name= "upsample2_", data_format="channels_last")(conv2_3_16)
    
    upsample4_ = Conv2DTranspose(filters=16, kernel_size=(8, 8), strides=(4, 4), padding = "same", activation='linear', name= "upsample4_", data_format="channels_last")(conv3_3_16)

    upsample8_ = Conv2DTranspose(filters=16, kernel_size=(16, 16), strides=(8, 8), padding = "same", activation='linear', name= "upsample8_", data_format="channels_last")(conv4_3_16)

    upsample16_ = Conv2DTranspose(filters=16, kernel_size=(32, 32), strides=(16, 16), padding = "same", activation='linear', name= "upsample16_", data_format="channels_last")(conv5_3_16)


    #sigmoid_fuse = Activation(activation="sigmoid")(new_score_weighting)

    
    concat_upscore = concatenate([upsample2_, upsample4_, upsample8_, upsample16_], axis=-1)
    
    # upscore_fuse = Activation(activation="sigmoid")(concat_upscore)
    
    new_score_weighting = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid', name= "new_score_weighting", data_format="channels_last")(concat_upscore)
    
    my_model = Model(input=model_vgg19_conv.input , output=new_score_weighting)

    return my_model

model = get_model()

print(model.summary())

model.compile(optimizer="Adam", loss=tf.losses.sigmoid_cross_entropy)
model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

model.fit(x=imgs_orig_train, y=imgs_cup_train, batch_size=16, epochs=40, shuffle=True, validation_split=0.2, callbacks=[model_checkpoint])
