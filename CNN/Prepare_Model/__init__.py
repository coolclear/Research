import tensorflow as tf
import os, sys
import re
sys.path.append('/home/yang/Research/CNN/Deep_Models/')
from vgg16 import Vgg16
from Resnet import Resnet
from Shallow_CNN import Shallow_CNN
from GBP_End2End import GBP_End2End

from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(tf.shape(grad)))

@ops.RegisterGradient("DeconvRelu")
def _DeconvReluGrad(op, grad):
    return tf.where(0. < grad, grad, tf.zeros(tf.shape(grad)))

"""
Key Design Principle:
1. the input/output placeholders should either be provided or instantiated by the prepare function 
"""

weight_path = '/home/yang/Research/Deep_Models/vgg16_weights.npz'

def prepare_vgg(sal_type, layer_idx, load_weights, sess):

    # construct the graph based on the gradient type we want
    if sal_type == 'GuidedBackprop':
        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
            vgg = Vgg16(sess=sess)

    elif sal_type == 'Deconv':
        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'DeconvRelu'}):
            vgg = Vgg16(sess=sess)

    elif sal_type == 'PlainSaliency':
        vgg = Vgg16(sess=sess)

    else:
        raise Exception("Unknown saliency_map type - 1")

    # different options for loading weights
    if load_weights == 'trained':
        vgg.load_weights(weight_path, sess)

    elif load_weights == 'random':
        vgg.init(sess)

    elif load_weights == 'part':
        # fill the first "idx" layers with the trained weights
        # randomly initialize the rest
        vgg.load_weights_part(layer_idx * 2 + 1, weight_path, sess)

    elif load_weights == 'reverse':
        # do not fill the first "idx" layers with the trained weights
        # randomly initialize them
        vgg.load_weights_reverse(layer_idx * 2 + 1, weight_path, sess)

    elif load_weights == 'only':
        # do not load a specific layer ("idx") with the trained weights
        # randomly initialize it
        vgg.load_weights_only(layer_idx * 2 + 1, weight_path, sess)

    else:
        raise Exception("Unknown load_weights type - 1")

    return vgg

def prepare_Resnet(output_dim,
                   sess=None, inputT=None, input_dim=None, num_logits=100, checkpoint_dir=None, reuse=False,
                   sal_type='PlainSaliency'):

    # construct the graph based on the gradient type we want
    if sal_type == 'GuidedBackprop':

        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):

            if inputT is not None:
                model = Resnet(inputT=inputT, output_dim=output_dim)
            elif input_dim is not None:
                inputT = tf.placeholder(tf.float32, [None, input_dim, input_dim, 3])  # RGB by default
                model = Resnet(inputT=inputT, output_dim=output_dim)
            else:
                raise Exception("Either inputT should be provided or input_dim should be specified!")

    elif sal_type == 'Deconv':

        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'DeconvRelu'}):

            if inputT is not None:
                model = Resnet(inputT=inputT, output_dim=output_dim)
            elif input_dim is not None:
                inputT = tf.placeholder(tf.float32, [None, input_dim, input_dim, 3])  # RGB by default
                model = Resnet(inputT=inputT, output_dim=output_dim)
            else:
                raise Exception("Either inputT should be provided or input_dim should be specified!")

    else:

        if inputT is not None:
            model = Resnet(inputT=inputT, output_dim=output_dim, reuse=reuse)
        elif input_dim is not None:
            inputT = tf.placeholder(tf.float32, [None, input_dim, input_dim, 3])  # RGB by default
            model = Resnet(inputT=inputT, output_dim=output_dim, reuse=reuse)
        else:
            raise Exception("Either inputT should be provided or input_dim should be specified!")

    if checkpoint_dir and sess:

        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_dir)
        print("Model Restored!")

    return inputT, model

def prepare_keras_vgg16(sal_type, init, sess):

    K.set_session(sess)  # set Keras to use the given sess

    # construct the graph based on the gradient type we want
    if sal_type == 'GuidedBackprop':
        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
            if init == 'random':
                vgg16 = VGG16(weights=None)
            else:
                vgg16 = VGG16(weights='imagenet')

    elif sal_type == 'Deconv':
        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'DeconvRelu'}):
            if init == 'random':
                vgg16 = VGG16(weights=None)
            else:
                vgg16 = VGG16(weights='imagenet')

    elif sal_type == 'PlainSaliency':
        if init == 'random':
            vgg16 = VGG16(weights=None)
        else:
            vgg16 = VGG16(weights='imagenet')

    else:
        raise Exception("Unknown saliency_map type - 1")

    if init == 'random':
        sess.run(tf.global_variables_initializer())

    return vgg16

def prepare_keras_resnet50(sal_type, init, sess):

    K.set_session(sess)  # set Keras to use the given sess
    K.set_learning_phase(0)

    # construct the graph based on the gradient type we want
    if sal_type == 'GuidedBackprop':
        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
            if init == 'random':
                resnet50 = ResNet50(weights=None)
            else:
                resnet50 = ResNet50(weights='imagenet')

    elif sal_type == 'Deconv':
        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'DeconvRelu'}):
            if init == 'random':
                resnet50 = ResNet50(weights=None)
            else:
                resnet50 = ResNet50(weights='imagenet')

    elif sal_type == 'PlainSaliency':
        if init == 'random':
            resnet50 = ResNet50(weights=None)
        else:
            resnet50 = ResNet50(weights='imagenet')

    else:
        raise Exception("Unknown saliency_map type - 1")

    if init == 'random':
        sess.run(tf.global_variables_initializer())

    return resnet50

def prepare_GBP_Shallow_CNN(output_dim, inputT=None, input_dim=None):

    """
    Notice that the gradient has been over-written to the GBP!!!
    :param output_dim: this should always be specified
    :param inputT: the input placeholder provided
    :param input_dim: this has to be specified if inputT is not provided
    :param num_logits: the number of logits of the shallow CNN, which is used for the GBP reconstruction
    :param sess: should always be provided
    :param loadback: load back the weights?
    :return:
    """

    eval_graph = tf.get_default_graph()
    with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):

        if inputT is not None:
            model = Shallow_CNN(inputT, output_dim=output_dim)
        elif input_dim is not None:
            inputT = tf.placeholder(tf.float32, [None, input_dim, input_dim, 3])  # RGB by default
            model = Shallow_CNN(inputT, output_dim=output_dim)
        else:
            raise Exception("Either inputT should be provided or input_dim should be specified!")

        return inputT, model

def prepare_GBP_End2End(output_dim,
                        sess=None, inputT=None, input_dim=None, num_logits=100, checkpoint_dir=None, reuse=False):

    """
    :param output_dim: this should always be specified
    :param inputT: the input placeholder provided
    :param input_dim: this has to be specified if inputT is not provided
    :param num_logits: the number of logits of the shallow CNN, which is used for the GBP reconstruction
    :param sess: should always be provided
    :param loadback: load back the weights?
    :return:
    """

    if inputT is not None:
        model = GBP_End2End(inputT, output_dim, num_logits=num_logits, reuse=reuse)
    elif input_dim is not None:
        inputT = tf.placeholder(tf.float32, [None, input_dim, input_dim, 3]) # RGB by default
        model = GBP_End2End(inputT, output_dim, num_logits=num_logits, reuse=reuse)
    else:
         raise Exception("Either inputT should be provided or input_dim should be specified!")

    if checkpoint_dir and sess:

        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_dir)
        print("Model Restored!")


    return inputT, model


