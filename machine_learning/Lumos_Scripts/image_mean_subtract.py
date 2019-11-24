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
from PIL import Image

originals_fucked_predictions_glaucoma_path = "/Users/karimabedrabbo/Desktop/Intermediates/originals_images_with_fucked_disc_predictions/glaucoma_disc_predictions_fucked"
originals_fucked_predictions_not_glaucoma_path = "/Users/karimabedrabbo/Desktop/Intermediates/originals_images_with_fucked_disc_predictions/not_glaucoma_disc_predictions_fucked"

originals_fucked_predictions_glaucoma = glob.glob1(originals_fucked_predictions_glaucoma_path,"*.jpg")
originals_fucked_predictions_not_glaucoma = glob.glob1(originals_fucked_predictions_not_glaucoma_path,"*.jpg")

output_path_glaucoma = "/Users/karimabedrabbo/Desktop/Intermediates/output/glaucoma"
output_path_not_glaucoma = "/Users/karimabedrabbo/Desktop/Intermediates/output/not_glaucoma"


for image_name in originals_fucked_predictions_glaucoma:
    img = np.array(image.load_img(originals_fucked_predictions_glaucoma_path + "/" + image_name))
    img = img.astype(np.float32)
    img[:,:,0] -= 103.939
    img[:,:,1] -= 116.779
    img[:,:,2] -= 123.68
    img = Image.fromarray(np.uint8(img))
    img.save(output_path_glaucoma + "/" + image_name)

for image_name in originals_fucked_predictions_not_glaucoma:
    img = np.array(image.load_img(originals_fucked_predictions_not_glaucoma_path + "/" + image_name))
    img = img.astype(np.float32)
    img[:,:,0] -= 103.939
    img[:,:,1] -= 116.779
    img[:,:,2] -= 123.68
    img = Image.fromarray(np.uint8(img))
    img.save(output_path_not_glaucoma + "/" + image_name)
