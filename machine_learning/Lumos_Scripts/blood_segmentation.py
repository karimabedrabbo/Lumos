import numpy as np
import ConfigParser
from matplotlib import pyplot as plt
#Keras
from keras.models import model_from_json
from keras.models import Model
#scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
import sys
sys.path.insert(0, './lib/')
# help_functions.py
from help_functions import *
# extract_patches.py
from extract_patches import recompone
from extract_patches import recompone_overlap
from extract_patches import paint_border
from extract_patches import kill_border
from extract_patches import pred_only_FOV
from extract_patches import get_data_testing
from extract_patches import get_data_testing_overlap
# pre_processing.py
from pre_processing import my_PreProc

import os
import cv2
import scipy
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.layers import Dense
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split




patch_height,patch_width = 50,50
#number of total patches:
N_subimgs = 190000
#if patches are extracted only inside the field of view:
inside_FOV = False
#Number of training epochs
N_epochs = 150
batch_size = 32

#number of full images for the test (max 20)
Imgs_to_test = 20 # full_images_to_test
#How many original-groundTruth-prediction images are visualized in each image
N_visual = 1 # N_group_visual
#Compute average in the prediction, improve results but require more patches to be predicted
average_mode = True
#Only if average_mode==True. Stride for patch extraction, lower value require more patches to be predicted
stride_height = 5
stride_width = 5
assert (stride_height < patch_height and stride_width < patch_width)
#original test images (for FOV selection)
full_img_height = 350
full_img_width = 350


def get_data(folder):
    """
    Load the data and labels from the given folder.
    """
    X = []
    y = []

    for image_filename in os.listdir(folder):
        img_file = cv2.imread(folder + '/' + image_filename)
        if img_file is not None:

            if not image_filename.startswith('NoGlauc'):
                label = 0
            else:
                label = 1
            img_arr = np.asarray(img_file)
            # img_arr = my_PreProc(img_arr)    #all pre-processing
            # img_arr = np.expand_dims(img_arr, axis=0)
            X.append(img_arr)
            y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y



X_train, y_train = get_data('Originals_for_training/training/')
X_test, y_test = get_data('Originals_for_training/testing/')


#============ Load the data and divide in patches
patches_imgs_test = None
new_height = None
new_width = None
masks_test  = None
patches_masks_test = None
if average_mode == True:
    patches_imgs_test, new_height, new_width, masks_test = get_data_testing_overlap(
        LUMOS_test_imgs_original = X_train,  #original
        LUMOS_test_groudTruth = X_test,  #masks
        Imgs_to_test = Imgs_to_test,
        patch_height = patch_height,
        patch_width = patch_width,
        stride_height = stride_height,
        stride_width = stride_width
    )


#Load the saved model
model = model_from_json('test/test_architecture.json').read()
model.load_weights('test/test_last_weights.h5')



predictions = model.predict(patches_imgs_test, batch_size=32, verbose=2)
print "predicted images size :"
print predictions.shape

#===== Convert the prediction arrays in corresponding images
pred_patches = pred_to_imgs(predictions, patch_height, patch_width, "original")



#========== Elaborate and visualize the predicted images ====================
pred_imgs = None
orig_imgs = None
gtruth_masks = None
if average_mode == True:
    pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)# predictions
    orig_imgs = my_PreProc(X_train)    #originals
    gtruth_masks = masks_test  #ground truth masks
else:
    pred_imgs = recompone(pred_patches,13,12)       # predictions
    orig_imgs = recompone(patches_imgs_test,13,12)  # originals
    gtruth_masks = recompone(X_test,13,12)  #masks



orig_imgs = orig_imgs[:,:,0:full_img_height,0:full_img_width]
pred_imgs = pred_imgs[:,:,0:full_img_height,0:full_img_width]
gtruth_masks = gtruth_masks[:,:,0:full_img_height,0:full_img_width]
print "Orig imgs shape: " +str(orig_imgs.shape)
print "pred imgs shape: " +str(pred_imgs.shape)
print "Gtruth imgs shape: " +str(gtruth_masks.shape)
visualize(group_images(orig_imgs,N_visual),path_experiment+"all_originals")#.show()
visualize(group_images(pred_imgs,N_visual),path_experiment+"all_predictions")#.show()
visualize(group_images(gtruth_masks,N_visual),path_experiment+"all_groundTruths")#.show()




# # img = image.load_img('Originals_for_training/training/Glauc30.jpg', target_size=(256, 256, 3))
# # img = image.load_img('single_image/Glauc2.jpg', target_size=(256, 256, 3))
# img = np.asarray('single_image/Glauc2.jpg')
# # img = image.load_img('optic-nerve-cnn-master/data/DRIONS-DB/images/image_001.jpg')

# # arr = np.array(img)
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






#================ Run the prediction of the patches ==================================
# #Load the saved model
# model = model_from_json('test/test_architecture.json').read())
# model.load_weights('test/test_last_weights.h5')
# #Calculate the predictions
# predictions = model.predict(patches_imgs_test, batch_size=32, verbose=2)
# print "predicted images size :"
# print predictions.shape

# #===== Convert the prediction arrays in corresponding images
# pred_patches = pred_to_imgs(predictions, patch_height, patch_width, "original")

# pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)# predictions
# pred_imgs = pred_imgs[:,:,0:full_img_height,0:full_img_width]










