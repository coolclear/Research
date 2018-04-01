## The following code was largely taken from
## https://github.com/raghakot/keras-resnet/blob/master/resnet.py
## under the MIT license.
## Changes copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

## Modified by Yang Zhang <yz78@rice.edu>

import sys
import os
sys.path.append('/home/yang/Research/CNN/')
sys.path.append('/home/yang/Research/CNN/Prepare_Model/')
sys.path.append('/home/yang/Research/CNN/Tools')

import tensorflow as tf
import numpy as np
from Prepare_Model import prepare_GBP_shallow_CNN


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, merge
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout, Lambda
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler

import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)

def _relu(input):
    """Helper to build a relu block
    """
    return Activation("relu")(input)

def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f

def _conv_relu(**conv_params):
    """Helper to build a conv -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _relu(conv)

    return f

def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f

def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])

def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            with tf.variable_scope("residual"+str(i)):
                input = block_function(filters=filters, init_strides=init_strides,
                                       is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f

def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f

def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3

def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier

def gbp_reconstruction(keras_input_tensor):

    tf_input = keras_input_tensor.output  # get the real tensor out of the keras layer wrap

    shallow_CNN = prepare_GBP_shallow_CNN(inputPH=tf_input)  # create a shallow CNN for the GBP reconstruction

    logits = shallow_CNN.logits  # get the logits we need

    random_index = np.random.choice(100, 1)  # not really random, will be fixed with the graph

    tfOp_gbp_raw = tf.gradients(logits[:, random_index], tf_input)[0] # raw gbp reconstruction

    # normalizations
    tfOp_gbp_submin = tf.map_fn(lambda img: img - tf.reduce_min(img), tfOp_gbp_raw)
    tfOp_gbp_divmax = tf.map_fn(lambda img: img / tf.reduce_max(img), tfOp_gbp_submin)
    tfOp_gbp_255 = tf.map_fn(lambda img: tf.cast(img * 255, tf.int32), tfOp_gbp_divmax, dtype=tf.int32)

    return tfOp_gbp_255

class ResnetBuilder_gbp(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions, with_detector=None, 
              activation=True, Dropout=Dropout):
        """Builds a custom ResNet like architecture.

        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved

        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)
        tmp = []

        input = Input(shape=input_shape)

        ################################ where different ###########################################

        input_gbp = Lambda(gbp_reconstruction)(input)

        ################################ where different ###########################################

        conv1 = _conv_bn_relu(filters=16, kernel_size=(3, 3))(input_gbp)

        block = conv1
        filters = 16
        for i, r in enumerate(repetitions):
            with tf.variable_scope("block"+str(i)):
                block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)
        block = Dropout(0.5)(block)

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="softmax" if activation else 'linear', name='classifier')(flatten1)

        outs = [dense]

        model = Model(inputs=input, outputs=outs)
        return model

    @staticmethod
    def build_resnet_32(input_shape, num_outputs, with_detector=None,
                        activation=True, Dropout=Dropout):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [5, 5, 5],
                                   with_detector=with_detector, activation=activation,
                                   Dropout=Dropout)
