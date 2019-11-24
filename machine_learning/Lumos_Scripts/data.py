from __future__ import print_function

import os
import numpy as np

from skimage.io import imsave, imread

data_path = 'dataset/'

image_rows = 350
image_cols = 350


def create_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = len(images) // 3

    imgs = np.ndarray((total, image_rows, image_cols, 3), dtype=np.uint8)
    imgs_disc = np.ndarray((total, image_rows, image_cols, 3), dtype=np.uint8)
    imgs_cup = np.ndarray((total, image_rows, image_cols, 3), dtype=np.uint8)

    i_one = 0
    i_two = 0
    i_three = 0

    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if not image_name.startswith('.'):
            if 'Cup' in image_name:
                img_cup = imread(os.path.join(train_data_path, image_name))
                img_cup = np.array([img_cup])
                imgs_cup[i_one] = img_cup
                i_one += 1
            elif 'Disc' in image_name:
                img_disc = imread(os.path.join(train_data_path, image_name))
                img_disc = np.array([img_disc])
                imgs_disc[i_two] = img_disc
                i_two += 1
            else:
                img = imread(os.path.join(train_data_path, image_name))
                img = np.array([img])
                imgs[i_three] = img
                i_three += 1

        print('Done: {0}/{1} images'.format(i_one, total))

    print('Loading done.')

    np.save('imgs_cup_train.npy', imgs_cup)
    np.save('imgs_disc_train.npy', imgs_disc)
    np.save('imgs_orig_train.npy', imgs)
    print('Saving to .npy files done.')


def load_train_data():

    imgs_cup_train = np.load('imgs_cup_train.npy')
    imgs_disc_train = np.load('imgs_disc_train.npy')
    imgs_orig_train = np.load('imgs_orig_train.npy')

    return imgs_cup_train, imgs_disc_train, imgs_orig_train


def create_test_data():
    train_data_path = os.path.join(data_path, 'test')
    images = os.listdir(train_data_path)
    total = len(images) - 1

    imgs = np.ndarray((total, image_rows, image_cols, 3), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        if not image_name.startswith('.'):
            img_id = i+1
            img = imread(os.path.join(train_data_path, image_name))

            img = np.array([img])

            imgs[i] = img
            imgs_id[i] = img_id

            i += 1
            print('Done: {0}/{1} images'.format(i, total))

    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id

if __name__ == '__main__':
    create_train_data()
    create_test_data()
