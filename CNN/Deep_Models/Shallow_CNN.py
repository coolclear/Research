import numpy as np
import tensorflow as tf
import sys
sys.path.append('/home/yang/Research/')

class Shallow_CNN(object):

    def __init__(self, inputT,
                 act_type='relu', pool_type='maxpool', trainable=False,
                 input_dim=32, output_dim=100, reuse=False):

        """
        :param inputT: the input has to be a tensor; has to be provided at the constructing time
        :param act_type: act type
        :param pool_type: pool type
        :param trainable: is the weights trainable
        :param input_dim: the input dim, default to 32
        :param output_dim: the length of the logits
        """

        self.act_type = act_type
        self.pool_type = pool_type
        self.trainable = trainable
        self.reuse = reuse

        print("Shallow CNN Trainable? {}".format(trainable))

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layers_dic = {}

        self.inputT = inputT
        self.layers_dic['Shallow_CNN_input'] = self.inputT
        self.num_channel = self.inputT.get_shape().as_list()[-1]

        self.convlayers()
        self.fc_layers()

        self.logits = self.fc2
        self.probs = tf.nn.softmax(self.logits)

    def act(self, tensor, name):

        if self.act_type == 'relu':
            return tf.nn.relu(tensor, name=name)

        if self.act_type == 'softplus':
            return tf.nn.softplus(tensor, name=name)

    def pool(self, tensor, name):

        if self.pool_type == 'maxpool':

            return tf.nn.max_pool(tensor,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  name=name)

        if self.pool_type == 'avgpool':

            return tf.nn.avg_pool(tensor,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  name=name)

    def convlayers(self):

        # conv1_1
        with tf.name_scope('Shallow_CNN_conv1_1', reuse=self.reuse) as scope:

            kernel = tf.Variable(tf.truncated_normal([2, 2, self.num_channel, 256], dtype=tf.float32, stddev=1e-1),
                                 trainable=self.trainable,
                                 name='w')

            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=self.trainable,
                                 name='b')

            conv = tf.nn.conv2d(self.inputT, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)

            self.conv1_1 = self.act(tensor=out, name=scope)

            self.layers_dic['Shallow_CNN_conv1_1'] = self.conv1_1

        # conv1_2
        with tf.name_scope('Shallow_CNN_conv1_2', reuse=self.reuse) as scope:

            kernel = tf.Variable(tf.truncated_normal([2, 2, 256, 256], dtype=tf.float32, stddev=1e-1),
                                 trainable=self.trainable,
                                 name='w')

            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=self.trainable,
                                 name='b')

            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)

            self.conv1_2 = self.act(tensor=out, name=scope)

            self.layers_dic['Shallow_CNN_conv1_2'] = self.conv1_2

        # pool1
        self.pool1 = self.pool(tensor=self.conv1_2, name='pool1')
        self.layers_dic['Shallow_CNN_pool1'] = self.pool1

    def fc_layers(self):

        # fc1
        with tf.name_scope('Shallow_CNN_fc1', reuse=self.reuse) as scope:

            shape = int(np.prod(self.pool1.get_shape()[1:]))

            fc1w = tf.Variable(tf.truncated_normal([shape, 1024], dtype=tf.float32, stddev=1e-1),
                               trainable=self.trainable,
                               name='w')

            fc1b = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32),
                               trainable=self.trainable,
                               name='b')

            pool1_flat = tf.reshape(self.pool1, [-1, shape])

            fc1l = tf.nn.bias_add(tf.matmul(pool1_flat, fc1w), fc1b)
            self.fc1 = self.act(tensor=fc1l, name=scope)

            self.layers_dic['Shallow_CNN_fc1'] = self.fc1

        # fc2
        with tf.name_scope('Shallow_CNN_fc2', reuse=self.reuse) as scope:

            fc2w = tf.Variable(tf.truncated_normal([1024, self.output_dim], dtype=tf.float32, stddev=1e-1),
                               trainable=self.trainable,
                               name='w')

            fc2b = tf.Variable(tf.constant(0.0, shape=[self.output_dim], dtype=tf.float32),
                               trainable=self.trainable,
                               name='b')

            self.fc2 = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)

            self.layers_dic['Shallow_CNN_logits'] = self.fc2

    def init(self, sess):
        sess.run(tf.global_variables_initializer())