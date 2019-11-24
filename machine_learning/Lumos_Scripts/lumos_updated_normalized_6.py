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
from keras.preprocessing import image
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

print(device_lib.list_local_devices())

epochs = 100
# BASE_DIR = '../'
batchsize = 100


def get_model(weights_path=None):
    model = Sequential()

    model.add(Conv2D(32, (25, 25),input_shape=(175, 175, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(10, 10)))

    model.add(Conv2D(32, (3, 3),input_shape=(175, 175, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
 
    model.add(Conv2D(32, (3, 3),input_shape=(175, 175, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))

    model.add(Flatten())
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(1, activation = 'softmax'))
    
    model.summary()

    if weights_path:
        model.load_weights(weights_path)
    """
    sgd = optimizers.SGD(lr=0.0001, decay=1e-4, momentum=0.9, nesterov=True)
    """
    model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])

    return model

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



X_train, y_train = get_data('normalized_data_renamed/training/')
X_test, y_test = get_data('normalized_data_renamed/testing/')

"""
f = open('my_data.pkl', 'wb')
cPickle.dump((X_train, y_train), f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

with open('my_data.pkl', 'rb') as f:
    train_labels = pickle.load(f)

VGG19 = keras.applications.vgg19.VGG19(input_shape=(175, 175, 3), weights='imagenet', include_top=False)

VGG19.summary()

featurized_training_data = VGG19.predict(X_train, verbose=1)
featurized_test_data = VGG19.predict(X_test, verbose=1)

# Save featurizations
import pickle
with open('featurized_train_imgs.pkl', 'wb') as f:
    pickle.dump(featurized_training_data, f)
with open('featurized_test_imgs.pkl', 'wb') as f:
    pickle.dump(featurized_test_data, f)



with open('featurized_train_imgs.pkl', 'rb') as f:
    featurized_training_data = pickle.load(f)
with open('featurized_test_imgs.pkl', 'rb') as f:
    featurized_test_data = pickle.load(f)
"""
featurized_training_data = X_train
featurized_test_data = X_test

model = get_model()

# fits the model on batches
history = model.fit(
    featurized_training_data,
    y_train,
    validation_split=0.2,
    epochs=epochs,
    shuffle=True,
    batch_size=batchsize)


model.save_weights('binary_model.h5')
model.save('saved_model.h5')

print('Evaluation')

print("Loss: ", model.evaluate(featurized_test_data, y_test))

print('Predicting on test data')
y_pred = np.rint(model.predict(featurized_test_data))

print(accuracy_score(y_test, y_pred.round(), normalize=False))

lens_X_test, lens_y_test = get_data('normalized_data/testing_lens/')

lens_y_test = encoder.transform(lens_y_test)


print('Predicting on lens data')
lens_y_pred = np.rint(model.predict(lens_X_test))

print(accuracy_score(lens_y_test, lens_y_pred.round(), normalize=False))
