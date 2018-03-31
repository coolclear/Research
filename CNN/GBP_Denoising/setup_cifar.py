## setup_cifar.py -- code to set up the CIFAR dataset
##
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

## Modified by Yang Zhang <yz78@rice.edu>

import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import pickle
import urllib.request
import keras
import pickle as pkl
import numpy as np
from keras.datasets import cifar10

from resnet import ResnetBuilder
from keras.layers import Dropout

class CIFAR:
    def __init__(self, tag):

        if tag == "ORI":
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        else if tag == "GBP":
            with open('CIFAR10_GBP.pkl', 'rb') as file:
                (x_train, y_train) = pkl.load(file)
                (x_test, y_test) = pkl.load(file)
        else:
            print('Cannot recognize this indicator, initialization failed ... ')

        self.validation_data = x_test
        self.validation_labels = y_test
        self.train_data = x_train
        self.train_labels = y_train

class CIFARModel:

    def __init__(self, restore=None, session=None, Dropout=Dropout, num_labels=10):

        self.num_channels = 3
        self.image_size = 32
        self.num_labels = num_labels

        model = ResnetBuilder.build_resnet_32((3, 32, 32), num_labels, activation=False,
                                              Dropout=Dropout)

        if restore != None:
            model.load_weights(restore)

        self.model = model

    def predict(self, data):
        return self.model(data)
        
    
