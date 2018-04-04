## train.py -- train the MNIST and CIFAR models
##
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

## Modified by Yang Zhang <yz78@rice.edu>

import os
import sys
import tensorflow as tf
import numpy as np

from keras.models import Model
from keras.layers import Lambda, Input
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from setup_cifar import CIFAR
sys.path.append('/home/yang/Research/CNN/Prepare_Model')
from Prepare_Model import prepare_GBPdenoising_end2end

# np.random.seed(13)
# tf.set_random_seed(13)

def fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                   logits=predicted)
def get_lr(epoch):
    return 0.1*(.5**(epoch/300*10))

def identical(x):
    return x

def main():

    batch_size = 128

    data = CIFAR("ORI")

    sgd = SGD(lr=0.00, momentum=0.9, nesterov=False)
    schedule = LearningRateScheduler(get_lr)

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

    datagen.fit(data.train_data)

    tf_model = prepare_GBPdenoising_end2end()

    saver = tf.train.Saver()

    ########################### we just borrow Keras to help us train the model ##############################

    # wrap the tf tensor into keras tensor
    keras_input = Input(tensor=tf_model.input)
    keras_output = Lambda(identical)(tf_model.output)

    keras_model = Model(inputs=keras_input, outputs=keras_output)

    keras_model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])

    keras_model.fit_generator(datagen.flow(data.train_data, data.train_labels,
                                     batch_size=batch_size),
                        steps_per_epoch=data.train_data.shape[0] // batch_size,
                        epochs=1,
                        verbose=1,
                        validation_data=(data.test_data, data.test_labels),
                        callbacks=[schedule])

    ########################### we just borrow Keras to help us train the model ##############################
    sess = K.get_session()
    saver.save(sess, 'Models/TEST.ckpt')

if __name__ == "__main__":
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
