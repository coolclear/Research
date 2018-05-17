import numpy as np
import glob
import os, sys
from scipy.misc import imread, imresize
import scipy.io as sio
from keras.datasets import cifar10, cifar100
import pickle as pkl

def list_load(data_dir, names, size=(224, 224)):

    # To load a list of images
    # You need to specify the image directory and the image name with the extension
    # We assume the image format is RGB
    # The image will be resized to 224 * 224 * 3 by default

    fns = []
    image_list = []

    for name in names:
        path = os.path.join(data_dir, '{}'.format(name))
        file_name = os.path.basename(path).split('.')[0]
        fns.append(file_name)

        image = imread(path, mode='RGB')
        image = imresize(image, size).astype(np.float32)
        image_list.append(image)

    batch_img = np.array(image_list) # put into a batch

    return batch_img, fns

def prepare_SVHN(data_dir):

    train_dict = sio.loadmat(data_dir + 'train_32x32.mat')
    X = np.asarray(train_dict['X'])

    X_train = []
    for i in range(X.shape[3]):
        X_train.append(X[:,:,:,i])
    X_train = np.asarray(X_train)

    Y_train = train_dict['y']
    for i in range(len(Y_train)):
        if Y_train[i]%10 == 0:
            Y_train[i] = 0

    test_dict = sio.loadmat(data_dir + 'test_32x32.mat')
    X = np.asarray(test_dict['X'])

    X_test = []
    for i in range(X.shape[3]):
        X_test.append(X[:,:,:,i])
    X_test = np.asarray(X_test)

    Y_test = test_dict['y']
    for i in range(len(Y_test)):
        if Y_test[i]%10 == 0:
            Y_test[i] = 0

    return (X_train, Y_train), (X_test, Y_test)

def prepare_CIFAR100():
    return cifar100.load_data(label_mode='fine')

def prepare_CIFAR10():
    return cifar10.load_data()

def pickle_load(data_dir, pickle_name):

    with open(data_dir + pickle_name, 'rb') as file:
        # (x_train, y_train) = pkl.load(file)
        (x_test, y_test) = pkl.load(file)

    # return (x_train, y_train), (x_test, y_test)
    return (x_test, y_test)
