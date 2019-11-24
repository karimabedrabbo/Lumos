from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, \
    Convolution2D, MaxPooling2D, ZeroPadding2D, Input, Embedding, LSTM, merge, \
    Lambda, UpSampling2D, Deconvolution2D, Cropping2D
from keras.models import load_model

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.models import model_from_json

from data import load_train_data, load_test_data
import cv2
import imhandle
from scipy.misc import toimage
import tensorflow as tf
import json



# K.set_image_data_format('channels_last')  # TF dimension ordering in this code
K.set_image_dim_ordering('tf')

img_rows = 256
img_cols = 256

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


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


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols, 3), dtype=np.uint8)
    for i in range(imgs.shape[0]):

        # toimage(imgs[i]).show()
        arr = imgs[i]

        # arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        # arr = np.stack((arr,), axis=-1)

            # lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
            # lab_planes = cv2.split(lab)
            # clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
            # lab_planes[0] = clahe.apply(lab_planes[0])
            # lab = cv2.merge(lab_planes)
            # bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # arr = np.array(bgr)
            # temp = toimage(arr)
            # temp.show()

        arr = arr.astype('float32')
        arr /= 255.
        # arr = np.reshape(arr, (256,256))
        # arr = np.expand_dims(arr, axis=0)
        imgs_p[i] = resize(arr, (img_cols, img_rows), preserve_range=True)

    # imgs_p = imgs_p[...]
    return imgs_p

    # imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    # for i in range(imgs.shape[0]):
    #     imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    # imgs_p = imgs_p[..., np.newaxis]
    # return imgs_p




# # img = image.load_img('Originals_for_training/training/Glauc30.jpg', target_size=(256, 256, 3))
# img = image.load_img('single_image/Glauc2.jpg', target_size=(256, 256, 3))
# # img = image.load_img('optic-nerve-cnn-master/data/DRIONS-DB/images/image_001.jpg')

# arr = np.array(img)
# arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
# arr = np.stack((arr,)*3, axis=-1)

# lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
# lab_planes = cv2.split(lab)
# clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(5,5))
# lab_planes[0] = clahe.apply(lab_planes[0])
# lab = cv2.merge(lab_planes)
# bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# arr = np.array(bgr)
# temp = toimage(arr)
# temp.show()
# temp.save("gray.jpg")

# arr = arr.astype('float')
# arr /= 255
# arr = np.expand_dims(arr, axis=0)
# pred = model.predict(arr)
# pred = np.reshape(pred, (256,256))


# output = toimage(pred)
# output.show()
# output.save("prediction.jpg")






def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    
    # imgs_cup_train, imgs_orig_train = load_train_data()

    # imgs_cup_train = preprocess(imgs_cup_train)
    # # imgs_disc_train = preprocess(imgs_disc_train)
    # imgs_orig_train = preprocess(imgs_orig_train)

    # np.save('imgs_cup_train_preprocess.npy', imgs_cup_train)
    # np.save('imgs_orig_train_preprocess.npy', imgs_orig_train)


    imgs_cup_train = np.load('imgs_cup_train_preprocess.npy')
    imgs_orig_train = np.load('imgs_orig_train_preprocess.npy')

    # imgs_train = imgs_train.astype('float32')
    # mean = np.mean(imgs_train)  # mean for data centering
    # std = np.std(imgs_train)  # std for data normalization

    # imgs_train -= mean
    # imgs_train /= std

    # imgs_mask_train = imgs_mask_train.astype('float32')
    # imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    def get_model():
        inputs = Input(shape=(img_rows, img_cols, 3))
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

        conv10 = Convolution2D(3, (1, 1), activation='sigmoid', border_mode='same')(conv9)
        #conv10 = Flatten()(conv10)

        model = Model(input=inputs, output=conv10)

        model.summary()

        return model


    

    # config = open("Keras_model_structure.json", "rb").read()
    # model = model_from_json(config)





    # model.save('cup_architecture_from_json.h5')

    # model = load_model('glaucoma.h5')
    model = get_model()



    # model_json = model.to_json()
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json)


    print(model.summary())
    sgd = SGD(lr=3e-4, clipnorm=1.)
    model.compile(optimizer=sgd, loss=log_dice_loss)
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)


    print('-'*30)
    print('Fitting model...')
    print('-'*60)
    print('getting shape')
    print(imgs_orig_train.shape)
    print(imgs_cup_train.shape)

    # input_tensor = Input(shape=(3,224, 224))  # this assumes K.image_data_format() == 'channels_first'
    # model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)


    history = model.fit(imgs_cup_train,imgs_orig_train, batch_size=16, epochs=40, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])

    model.save('saved_model.h5')


    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)


    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)

    model.load_weights('weights.h5')
    # model = model_from_json(json_string)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=0)
    np.save('imgs_mask_test.npy', imgs_mask_test)

    [toimage(imgs_mask_test[i]).show() for i in range(len(imgs_mask_test))]

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

# pred = np.reshape(pred, (256,256))

    for i in range(len(imgs_mask_test)):
        image = imgs_mask_test[i]
        image = np.reshape(image, (256,256))
        image = toimage(image)
        imsave(os.path.join(pred_dir, i + '_pred.jpg'), image)

    # for image, image_id in zip(imgs_mask_test, imgs_id_test):
    #     image = (image[:, :, 0] * 255.).astype(np.uint8)
    #     imsave(os.path.join(pred_dir, str(image_id) + '_pred.jpg'), image)

if __name__ == '__main__':
    train_and_predict()
