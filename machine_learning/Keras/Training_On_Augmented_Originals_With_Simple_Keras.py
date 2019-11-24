
# coding: utf-8

# # Training On Augmented Originals With Simple Keras

# Description: This is the Python Notebook for trying Simple Keras, aka LeNet, on the original Lumos dataset, without any fancy style transfer stuff. However, this data has been augmented a bunch. There are 10 different types of augmentations that were tested: cropped_glaucoma, light, lighter_medium, medium, medium_heavy, heavy, all_aug, all_heavy_aug, lens_light, and lens_medium_heavy.
# 
# They were split into training and testing sets, with an 80:20 split for training:testing. Both the validation accuracies as well as the testing accuracies are included below for the different types of augmentations.

# # All the imports

# In[88]:

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
# %matplotlib inline
import matplotlib.pyplot as plt
from keras import backend as K
import glob
from PIL import Image
from keras.optimizers import RMSprop, SGD
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, confusion_matrix,  roc_curve, auc
# import scikitplot as skplt


# In[39]:

K.set_image_dim_ordering('tf')


# # Preprocessing / Set-up Methods

# In[40]:

def preprocess(imgs, imgNorm):
    imgs = imgs.astype('float32')
    for i in range(len(imgs)):
        img = imgs[i]
        if imgNorm == "sub_and_divide":
            img = img.astype('float32')
            img = img / 127.5 - 1
            imgs[i] = img
        elif imgNorm == "sub_mean":
            img = img.astype('float32')
            img[:,:,0] -= 103.939
            img[:,:,1] -= 116.779
            img[:,:,2] -= 123.68
            imgs[i] = img
        elif imgNorm == "divide":
            img = img.astype('float32')
            img *= np.float32(1.0/255.0)
#             img[:,:,0] = np.float32(255.0)/img[:,:,0].max()
#             img[:,:,1] = np.float32(255.0)/img[:,:,1].max()
#             img[:,:,2] = np.float32(255.0)/img[:,:,2].max()
            #img = np.multiply(img, np.float(1./255.), out=img, casting='safe')
            imgs[i] = img
    return imgs


# In[41]:

def preprocess_other(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols, 3), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        arr = imgs[i]
        arr = arr.astype('float32')
        arr /= 255.
        imgs_p[i] = resize(arr, (img_cols, img_rows), preserve_range=True)
    return imgs_p


# In[42]:

def resize_multi(arr, rows, cols):
    for i in range(len(arr)):
        arr[i] = cv2.resize(arr[i], (rows, cols))
    return arr


# In[43]:

def create_list(arr):
    temp = []
    for image in arr:
        temp.append(image)
    return temp


# In[44]:

def get_data(numpy_arr, glauc_bool):
    X = []
    y = []

    for image in numpy_arr:
            X.append(image)
            y.append(glauc_bool)
    X = np.array(X)
    y = np.array(y)
    return X,y



# In[45]:

def multiple_image_numpy(imgs, rows, cols, channels):
    total = len(imgs)
    new = np.ndarray((total, rows, cols, channels), dtype=np.uint8)

    for i in range(total):
        new[i] = imgs[i]
    
    return new


# In[99]:

def sort(path, im_type, backend, percentage):
    
    images = glob.glob1(path,"*." + str(im_type))
    num_images = int(len(images) * percentage)
    images = images[:num_images]
#     print(images)
#     print(images[:num_images])
#     print(len(images))
#     print(len(images[:num_images]))

    im_type_length = len(im_type) + 1
    images = [element[:-im_type_length] for element in images]
    images.sort(key=int)
    images = [element + "." + str(im_type) for element in images]

    if backend == "keras" or backend == "Keras" or backend == "keras":
        images = [np.array(image.load_img(path + "/" + fname)) for fname in images]
    if backend == "opencv" or backend == "openCV" or backend == "OPENCV" or backend == "cv2":
        images = [np.array(cv2.imread(path + "/" + fname)) for fname in images]
    if backend == "PIL" or backend == "pil":
        images = [np.array(Image.open(path + "/" + fname)) for fname in images]
    return images

# sort('Images/cropped_not_glauc_orig_png', 'png', 'keras', .5)


# # Getting the Model

# In[47]:

def get_simple_keras(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    return model


# # Getting the training data

# In[48]:

orig_glauc = "Images/cropped_glauc_orig_png"
orig_not_glauc = "Images/cropped_not_glauc_orig_png"
light_glauc = "Images/light_3_9/glauc"
light_not_glauc = "Images/light_3_9/not_glauc"
lighter_medium_glauc = "Images/lighter_medium_3_9/glauc"
lighter_medium_not_glauc = "Images/lighter_medium_3_9/not_glauc"
medium_glauc = "Images/medium_3_9/glauc"
medium_not_glauc = "Images/medium_3_9/not_glauc"
medium_heavy_glauc = "Images/medium_heavy_3_9/glauc"
medium_heavy_not_glauc = "Images/medium_heavy_3_9/not_glauc"
heavy_glauc = "Images/heavy_3_9/glauc"
heavy_not_glauc = "Images/heavy_3_9/not_glauc"
all_glauc = "Images/all_aug_3_9/glauc"
all_not_glauc = "Images/all_aug_3_9/not_glauc"
all_heavy_glauc = "Images/all_heavy_aug_3_9/glauc"
all_heavy_not_glauc = "Images/all_heavy_aug_3_9/not_glauc"
lens_light_glauc = "Images/lens_light_3_9/glauc"
lens_light_not_glauc = "Images/lens_light_3_9/not_glauc"
lens_medium_heavy_glauc = "Images/lens_medium_heavy_3_9/glauc"
lens_medium_heavy_not_glauc = "Images/lens_medium_heavy_3_9/not_glauc"


# # Final Preprocessing

# In[53]:

def final_preprocess(path_glauc, path_not_glauc, percentage):
    glauc = sort(path_glauc, "png", "keras", percentage)
    not_glauc = sort(path_not_glauc, "png", "keras", percentage)
    y_glauc = np.ones(len(glauc))
    y_not_glauc = np.zeros(len(not_glauc))
    glauc = resize_multi(glauc, 128, 128)
    not_glauc = resize_multi(not_glauc, 128, 128)
    glauc = multiple_image_numpy(glauc, 128, 128, 3)
    not_glauc = multiple_image_numpy(not_glauc, 128, 128, 3)
    glauc = preprocess(glauc, "divide")
    not_glauc = preprocess(not_glauc, "divide")
    x_final = np.vstack((glauc, not_glauc))
    y_final = np.hstack((y_glauc, y_not_glauc))
    rng_state = np.random.get_state()
    np.random.shuffle(x_final)
    np.random.set_state(rng_state)
    np.random.shuffle(y_final)
    y_final = to_categorical(y_final)
    
    x_final, x_test, y_final, y_test = train_test_split(x_final, y_final, test_size=0.2)
    
    return x_final, x_test, y_final, y_test


# # Create Matrices 

# In[50]:

history = []

paths = [orig_glauc, orig_not_glauc, 
         light_glauc, light_not_glauc,
        lighter_medium_glauc, lighter_medium_not_glauc,
        medium_glauc, medium_not_glauc,
        medium_heavy_glauc, medium_heavy_not_glauc,
        heavy_glauc, heavy_not_glauc,
        all_glauc, all_not_glauc,
        all_heavy_glauc, all_heavy_not_glauc,
        lens_light_glauc, lens_light_not_glauc,
        lens_medium_heavy_glauc, lens_medium_heavy_not_glauc]

testing_data = []

path_glauc = paths[0].split('/')[1][:-4]
# path_glauc = paths[6]
print(path_glauc)

# Images/medium_3_9/glauc


# # Train

# In[51]:

for i in range(0, len(paths), 2):
    # get right data from paths
    path_glauc = paths[i]
    path_not_glauc = paths[i + 1]
    print(path_glauc.split('/')[1][:-4])
        
    # preprocess the images
    percent_of_dataset = .25      # a range from [0,1] that determines how many of the total dataset to use for training
    x_final, x_test, y_final, y_test = final_preprocess(path_glauc, path_not_glauc, percent_of_dataset)
    
    testing_data.append([])
    testing_data[i//2].append(x_test)
    testing_data[i//2].append(y_test)
        
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=10, min_lr=0.00001, verbose=1)
    classifier_model = get_simple_keras((128,128,3))

    opt = SGD(lr=0.01)
    classifier_model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=['accuracy'])
    history.append([])
    history[i//2].append(path_glauc.split('/')[1][:-4])
    history[i//2].append(classifier_model.fit(x=np.array(x_final), y=np.array(y_final), batch_size=16, epochs=150, verbose=1, validation_split=0.2, shuffle=True, callbacks=[reduce_lr]))
    
    #save the model
    classifier_model.save_weights('saved_models/' + str(path_glauc.split('/')[1][:-4]) + '_model_weights.h5')
    classifier_model.save('saved_models/' + str(path_glauc.split('/')[1][:-4]) + '_model_itself.h5')


# In[ ]:

for i in range(len(history)):
    print(history[i][0] + ": " + str(max(history[i][1].history['val_acc'])))
    
print("\n\nHistory Array:\n\n"+str(history))

print("\n\nHistory Array:\n\n"+str(paths))

# https://datascience.stackexchange.com/questions/11747/cross-validation-in-keras


# # Plot learning curves

# In[ ]:

def plot_learning_curve(history, num):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./accuracy_curves/' + history[num][0] + '_accuracy_curve.png')
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./accuracy_curves/' + history[num][0] + '_loss_curve.png')



# In[ ]:

print(history[0][0] + "c: \n\n")
plot_learning_curve(history[0][1], 0)


# In[ ]:

print(history[1][0] + ": \n\n")
plot_learning_curve(history[1][1], 1)


# In[ ]:

print(history[2][0] + ": \n\n")
plot_learning_curve(history[2][1], 2)


# In[ ]:

print(history[3][0] + ": \n\n")
plot_learning_curve(history[3][1], 3)


# In[ ]:

print(history[4][0] + ": \n\n")
plot_learning_curve(history[4][1], 4)


# In[ ]:

print(history[5][0] + ": \n\n")
plot_learning_curve(history[5][1], 5)


# In[ ]:

print(history[6][0] + ": \n\n")
plot_learning_curve(history[6][1], 6)


# In[ ]:

print(history[7][0] + ": \n\n")
plot_learning_curve(history[7][1], 7)


# In[ ]:

print(history[8][0] + ": \n\n")
plot_learning_curve(history[8][1], 8)


# In[ ]:

print(history[9][0] + ": \n\n")
plot_learning_curve(history[9][1], 9)


# # Predict on test data (accuracy and confusion matrix)

# In[ ]:

for i in range(0, len(testing_data)):
    # get right data from paths
    test_data = testing_data[i]
    x_test = test_data[0]
    y_test = test_data[1]
    
    
    
    classifier_model = get_simple_keras((128,128,3))
    classifier_model.load_weights('saved_models/' + str(paths[i].split('/')[1][:-4]) + '_model_weights.h5')
    
    y_pred = np.rint(classifier_model.predict(x_test))
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print("Accuracy: " + str(accuracy))
    print("Confusion Matrix: \n" + str(confusion))


# # ROC Curves

# In[ ]:

def plot_roc(i):
#     print(stats[i][0] + ": " + str(max(history[i][1].history['val_acc'])))
    
    plt.title(str(stats[i][0]))
    plt.plot(stats[i][1], stats[i][2], 'b',
    label='AUC = %0.2f'% stats[i][3])
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    print("\n\n\n")
    
print("\n\nStats Array:\n\n"+str(stats))


# In[ ]:

plot_roc(0)


# In[ ]:

plot_roc(1)


# In[ ]:

plot_roc(2)


# In[ ]:

plot_roc(3)


# In[ ]:

plot_roc(4)


# In[ ]:

plot_roc(5)


# In[ ]:

plot_roc(6)


# In[ ]:

plot_roc(7)


# In[ ]:

plot_roc(8)


# In[ ]:

plot_roc(9)

