import numpy as np
import tensorflow as tf
import sys
sys.path.append('/home/yang/Research/')
from shallow_CNN import Shallow_CNN
from resnet import Resnet

class GBP_End2End(object):

    def __init__(self, trainable=False):

        eval_graph = tf.get_default_graph() # get the graph

        # construct the shallow CNN used for GBP reconstruction
        # the gradient has to be overwritten
        with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
            self.NN1 = Shallow_CNN(trainable=trainable)

        ##################################### GBP Reconstruction ###############################################

        logits = self.NN1.logits  # get the logits
        self.inputs = self.NN1.images # get the input

        index = tf.random_uniform([1], minval=0, maxval=100, dtype=tf.int32)[0]

        tfOp_gbp_raw = tf.gradients(logits[:, index], self.inputs)[0]  # raw gbp reconstruction

        # normalizations
        tfOp_gbp_submin = tf.map_fn(lambda img: img - tf.reduce_min(img), tfOp_gbp_raw)
        tfOp_gbp_divmax = tf.map_fn(lambda img: img / tf.reduce_max(img), tfOp_gbp_submin)
        tfOp_gbp_255 = tf.map_fn(lambda img: tf.cast(img * 255, tf.int32), tfOp_gbp_divmax, dtype=tf.int32)

        ##################################### GBP Reconstruction ###############################################

        # now use a Resnet to classify these GBP reconstructions
        # self.NN2 = Resnet(inputPH=tf.cast(tfOp_gbp_255, dtype=tf.float32), num_labels=10)
        self.NN2 = Resnet(inputPH=tfOp_gbp_divmax, num_labels=10)

        self.labels = self.NN2.labels
        self.phase = self.NN2.phase
        self.dp = self.NN2.dp
        self.gbp_reconstruction = tfOp_gbp_255
        self.logits = self.NN2.logits
        self.cost = self.NN2.cost
        self.accuracy = self.NN2.accuracy

    def init(self, sess):
        self.NN1.init(sess)
        self.NN2.init(sess)