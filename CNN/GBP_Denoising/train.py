## train.py -- train the MNIST and CIFAR models
##
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

## Modified by Yang Zhang <yz78@rice.edu>

from setup_cifar import CIFAR, CIFARModel
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

import tensorflow as tf


def fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                   logits=predicted)
def get_lr(epoch):
    return 0.1*(.5**(epoch/300*10))

def main():

    batch_size = 128
    tag = "GBP"

    model = CIFARModel().model
    data = CIFAR(tag)

    sgd = SGD(lr=0.00, momentum=0.9, nesterov=False)
    schedule= LearningRateScheduler(get_lr)

    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

    datagen.fit(data.train_data)

    model.fit_generator(datagen.flow(data.train_data, data.train_labels,
                                     batch_size=batch_size),
                        steps_per_epoch=data.train_data.shape[0] // batch_size,
                        epochs=1,
                        verbose=1,
                        validation_data=(data.test_data, data.test_labels),
                        callbacks=[schedule])

    model.save_weights('Models/CIFAR10_{}'.format(tag))

if __name__ == "__main__":
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
