import numpy as np
import tensorflow as tf
import sys
sys.path.append('/home/yang/Research/')
from shallow_CNN import Shallow_CNN
from resnet import Resnet

class GBP_End2End(object):

    def __init__(self, sess=None):

        eval_graph = tf.get_default_graph() # get the graph

        # construct the shallow CNN used for GBP reconstruction
        # the gradient has to be overwritten
        with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
            NN1 = Shallow_CNN(trainable=False)

        logits = NN1.logits  # get the logits
        self.input = NN1.images # get the input

        tfOp_gbp_raw = tf.gradients(logits[:, 13], self.input)[0]  # raw gbp reconstruction

        # normalizations
        tfOp_gbp_submin = tf.map_fn(lambda img: img - tf.reduce_min(img), tfOp_gbp_raw)
        tfOp_gbp_divmax = tf.map_fn(lambda img: img / tf.reduce_max(img), tfOp_gbp_submin)
        tfOp_gbp_255 = tf.map_fn(lambda img: tf.cast(img * 255, tf.int32), tfOp_gbp_divmax, dtype=tf.int32)

        NN2 = ResNet(inputPH=tfOp_gbp_255, num_labels=10)

        self.output = NN2.logits
        self.cost = NN2.cost

    def init(self, sess):
        sess.run(tf.global_variables_initializer())