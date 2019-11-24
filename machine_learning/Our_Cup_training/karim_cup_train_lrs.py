import os
import glob
import numpy as np


import keras
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Activation, concatenate, Reshape, Permute
from keras.optimizers import Adam
from keras import losses
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras import backend as K

import cv2
import tensorflow as tf
from scipy.misc import toimage
import itertools

K.set_image_dim_ordering('tf')

def getImageArr( path , width , height , imgNorm="sub_mean" , odering='channels_last' ):

    try:
        img = cv2.imread(path, 1)

        if imgNorm == "sub_and_divide":
            img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
        elif imgNorm == "sub_mean":
            img = cv2.resize(img, ( width , height ))
            img = img.astype(np.float32)
            img[:,:,0] -= 103.939
            img[:,:,1] -= 116.779
            img[:,:,2] -= 123.68
        elif imgNorm == "divide":
            img = cv2.resize(img, ( width , height ))
            img = img.astype(np.float32)
            img = img/255.0

        if odering == 'channels_first':
            img = np.rollaxis(img, 2, 0)
        return img
    except Exception, e:
        print path , e
        img = np.zeros((  height , width  , 3 ))
        if odering == 'channels_first':
            img = np.rollaxis(img, 2, 0)
        return img

def getSegmentationArr( path , nClasses ,  width , height  ):

    seg_labels = np.zeros((  height , width  , nClasses ))
    try:
        img = cv2.imread(path, 1)
        img = cv2.resize(img, ( width , height ))
        img = img[:, : , 0]

        for c in range(nClasses):
            seg_labels[: , : , c ] = (img == c ).astype(int)

    except Exception, e:
        print e

    seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
    return seg_labels

def imageSegmentationGenerator( images_path , segs_path ,  batch_size,  n_classes , input_height , input_width , output_height , output_width   ):
    
    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'

    images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
    images.sort()
    segmentations  = glob.glob( segs_path + "*.jpg"  ) + glob.glob( segs_path + "*.png"  ) +  glob.glob( segs_path + "*.jpeg"  )
    segmentations.sort()

    assert len( images ) == len(segmentations)
    for im , seg in zip(images,segmentations):
        assert(  im.split('/')[-1].split(".")[0] ==  seg.split('/')[-1].split(".")[0] )

    zipped = itertools.cycle( zip(images,segmentations) )

    while True:
        X = []
        Y = []
        for _ in range( batch_size) :
            im , seg = zipped.next()
            X.append( getImageArr(im , input_width , input_height )  )
            Y.append( getSegmentationArr( seg , n_classes , output_width , output_height )  )

        yield np.array(X) , np.array(Y)

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

def get_model():
    model_vgg19_conv = VGG19(weights='imagenet', include_top=False, input_shape=(256,256,3))

    conv2_3_16 = Conv2D(filters=16, kernel_size=(3, 3), padding = "same", kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.001), name= "conv2_3_16", data_format="channels_last")(model_vgg19_conv.layers[-17].output)

    conv3_3_16 = Conv2D(filters=16, kernel_size=(3, 3), padding = "same", kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.001), name= "conv3_3_16", data_format="channels_last")(model_vgg19_conv.layers[-12].output)

    conv4_3_16 = Conv2D(filters=16, kernel_size=(3, 3), padding = "same", kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.001), name= "conv4_3_16", data_format="channels_last")(model_vgg19_conv.layers[-7].output)

    conv5_3_16 = Conv2D(filters=16, kernel_size=(3, 3), padding = "same", kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.001), name= "conv5_3_16", data_format="channels_last")(model_vgg19_conv.layers[-2].output)

    upsample2_ = Conv2DTranspose(filters=16, kernel_size=(4, 4), strides=(2, 2), padding = "same", name= "upsample2_", data_format="channels_last")(conv2_3_16)
    
    upsample4_ = Conv2DTranspose(filters=16, kernel_size=(8, 8), strides=(4, 4), padding = "same", name= "upsample4_", data_format="channels_last")(conv3_3_16)

    upsample8_ = Conv2DTranspose(filters=16, kernel_size=(16, 16), strides=(8, 8), padding = "same",  name= "upsample8_", data_format="channels_last")(conv4_3_16)

    upsample16_ = Conv2DTranspose(filters=16, kernel_size=(32, 32), strides=(16, 16), padding = "same", name= "upsample16_", data_format="channels_last")(conv5_3_16)

    concat_upscore = concatenate([upsample2_, upsample4_, upsample8_, upsample16_], axis=-1)
    
    """
    new_score_weighting = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid', kernel_initializer=keras.initializers.Constant(value=0.01), name= "new_score_weighting", data_format="channels_last")(concat_upscore)
    
    my_model = Model(input=model_vgg19_conv.input , output=new_score_weighting)
    
    """

    conv6 = Conv2D(filters=1, kernel_size=(1,1), activation='relu', kernel_initializer=keras.initializers.Constant(value=0.01), name= "conv6", data_format="channels_last", border_mode='same')(concat_upscore)
    conv6 = Reshape((1,256*256))(conv6)
    conv6 = Permute((2,1))(conv6)

    conv7 = Activation('softmax')(conv6)

    my_model = Model(input=model_vgg19_conv.input, output=conv7)

    return my_model

#imgs_cup_train = np.load('gt_cup_train_new.npy')
#imgs_orig_train = np.load('orig_cup_train_new.npy')

training_gt_path = "/home/ubuntu/Karim_cup_train/train/cup/"
training_orig_path = "/home/ubuntu/Karim_cup_train/train/norm/"

validation_gt_path = "/home/ubuntu/Karim_cup_train/train/val_cup/"
validation_orig_path = "/home/ubuntu/Karim_cup_train/train/val_orig/"

cup_train_data_gen = imageSegmentationGenerator(training_orig_path, training_gt_path, 16, 1, 256, 256, 256, 256)

cup_val_data_gen  = imageSegmentationGenerator(validation_orig_path, validation_gt_path, 16, 1, 256, 256, 256, 256)


model = get_model()

print(model.summary())

model_checkpoint = ModelCheckpoint('karim_best_weights_new.h5', monitor='val_loss', save_best_only=True)
"""
adam = Adam(lr=.000001)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)

def scheduler(epoch):
    if epoch == 7:
        K.set_value(adam.lr, .00000000001)
    if epoch == 14:
        K.set_value(adam.lr, .000000000001)
    if epoch == 21:
        K.set_value(adam.lr, .0000000000001)
    if epoch == 28:
        K.set_value(adam.lr, .00000000000001)
    if epoch == 35:
        K.set_value(adam.lr, .000000000000001)
    return K.get_value(adam.lr)
"""

#model.load_weights("karim_best_weights_new.h5")

model.compile(optimizer="Adam", loss=log_dice_loss)

model.fit_generator(generator=cup_train_data_gen, steps_per_epoch=104, epochs=40, verbose=1, callbacks=[model_checkpoint], validation_data=cup_val_data_gen, validation_steps=26, shuffle=True)

#model.fit_generator(generator=cup_train_data_gen, steps_per_epoch=104, epochs=200, verbose=1, callbacks=[LearningRateScheduler(scheduler, verbose=1), model_checkpoint], validation_data=cup_val_data_gen, validation_steps=26, shuffle=True)

#model.fit(x=imgs_orig_train, y=imgs_cup_train, batch_size=16, epochs=200, shuffle=True, validation_split=0.2, verbose=1, callbacks=[LearningRateScheduler(scheduler, verbose=1), model_checkpoint])