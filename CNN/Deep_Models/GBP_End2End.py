import numpy as np
import tensorflow as tf
import sys
sys.path.append('/home/yang/Research/')
from Shallow_CNN import Shallow_CNN
from Resnet import Resnet

class GBP_End2End(object):

    def __init__(self, inputT, output_dim, num_logits=100):

        """
        GBP defense
        :param inputT: the input has to be a tensor; has to be provided at the constructing time
        :param output_dim: the output dim has to be specified
        :param num_logits: the num of logits of the shallow CNN
        """

        self.layers_dic = {}

        self.inputT = inputT
        self.layers_dic['GBP_End2End_input'] = self.inputT
        self.num_channel = self.inputT.get_shape().to_list()[-1]

        self.num_logits = num_logits
        self.output_dim = output_dim

        eval_graph = tf.get_default_graph() # get the graph
        # construct the shallow CNN used for the GBP reconstruction
        # the gradient has to be overwritten
        # the gradient registration happens in Prepare_Model
        with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
            self.NN1 = Shallow_CNN(self.inputT, output_dim=self.num_logits)

        ##################################### GBP Reconstruction ###############################################

        logits = self.NN1.logits  # get the logits

        index = tf.random_uniform([1], minval=0, maxval=self.num_logits, dtype=tf.int32)[0]

        tfOp_gbp_raw = tf.gradients(logits[:, index], self.inputT)[0]  # raw gbp reconstruction

        # normalizations
        tfOp_gbp_submin = tf.map_fn(lambda img: img - tf.reduce_min(img), tfOp_gbp_raw)
        tfOp_gbp_divmax = tf.map_fn(lambda img: img / tf.reduce_max(img), tfOp_gbp_submin)
        tfOp_gbp_255 = tf.map_fn(lambda img: tf.cast(img * 255, tf.int32), tfOp_gbp_divmax, dtype=tf.int32)

        ##################################### GBP Reconstruction ###############################################

        # now use a Resnet to classify these GBP reconstructions
        self.NN2 = Resnet(inputPH=tfOp_gbp_divmax, output_dim=self.output_dim)

        self.phase = self.NN2.phase
        self.kp = self.NN2.kp

        self.gbp_reconstructions = tfOp_gbp_255 # an output port if we want to visualize the reconstructions

        self.logits = self.NN2.logits
        self.probs = tf.nn.softmax(self.logits)

        self.layers_dic.update(self.NN1.layers_dic).update(self.NN2.layers_dic)

    def init(self, sess):
        self.NN1.init(sess)
        self.NN2.init(sess)