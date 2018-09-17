import tensorflow as tf
from Shallow_CNN import Shallow_CNN
from Resnet import Resnet

class GBP_End2End(object):

    def __init__(self, inputT, output_dim,
                 num_logits=100, reuse=False):

        print(" [*] Constructing GBP_End2End ... ")

        """
        GBP Defense
        :param inputT: 4D tensor, has to be provided
        :param output_dim: has to be provided
        :param num_logits: the num of logits of the shallow CNN used for the GBP reconstruction
        """

        self.inputT = inputT
        self.output_dim = output_dim

        self.num_logits = num_logits
        self.reuse = reuse

        print("Output dim = {}".format(self.output_dim))
        print("Reuse = {}, (T)Testing/(F)Training".format(self.reuse))

        self.layers_dic = {}
        self.layers_dic['GBP_End2End_input'] = self.inputT

        with tf.variable_scope("GBP_End2End") as scope:

            if self.reuse:
                scope.reuse_variables()

            eval_graph = tf.get_default_graph() # get the graph

            # Construct the shallow CNN used for the GBP reconstruction
            # The gradient has to be overwritten
            # The gradient registration happens in Prepare_Model
            with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
                self.NN1 = Shallow_CNN(self.inputT, output_dim=self.num_logits, reuse=self.reuse)


            ##################################### GBP Reconstruction ###############################################

            logits = self.NN1.logits  # get the logits

            # randomly pick one logit
            index = tf.random_uniform([1], minval=0, maxval=self.num_logits, dtype=tf.int32)[0]

            # raw gbp reconstruction
            tfOp_gbp_raw = tf.gradients(logits[:, index], self.inputT)[0]

            # normalizations
            tfOp_gbp_submin = tf.map_fn(lambda img: img - tf.reduce_min(img), tfOp_gbp_raw)
            tfOp_gbp_divmax = tf.map_fn(lambda img: img / tf.reduce_max(img), tfOp_gbp_submin)

            # just a port for the visualization if necessary
            tfOp_gbp_255 = tf.map_fn(lambda img: tf.cast(img * 255, tf.int32), tfOp_gbp_divmax, dtype=tf.int32)

            ##################################### GBP Reconstruction ###############################################

            # now use a Resnet to classify these GBP reconstructions
            self.NN2 = Resnet(tfOp_gbp_divmax, output_dim=self.output_dim, reuse=self.reuse)

        self.phase = self.NN2.phase
        self.kp = self.NN2.kp

        self.gbp_reconstructions = tfOp_gbp_255 # an output port for visualizing GBP reconstructions

        self.logits = self.NN2.logits
        self.probs = tf.nn.softmax(self.logits)

        self.layers_dic.update(self.NN1.layers_dic)
        self.layers_dic.update(self.NN2.layers_dic)

    def init(self, sess):
        self.NN1.init(sess)
        self.NN2.init(sess)