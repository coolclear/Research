import numpy as np
import tensorflow as tf

class Shallow_CNN(object):

    def __init__(self, inputT, output_dim,
                 trainable=False, reuse=False,
                 act_type='relu', pool_type='maxpool'):

        print(" [*] Constructing Shallow_CNN ... ")

        """
        :param inputT: 4D tensor; has to be provided
        :param output_dim: has to be provided
        :param act_type: activation, default to relu
        :param pool_type: pooling, default to maxpool
        :param trainable: are the weights trainable? default to false

        """

        self.inputT = inputT
        self.output_dim = output_dim

        self.act_type = act_type
        self.pool_type = pool_type
        self.trainable = trainable
        self.reuse = reuse

        print("Output dim = {}".format(self.output_dim))
        print("Reuse = {}, (T)Testing/(F)Training".format(self.reuse))
        print("Shallow CNN trainable = {}".format(self.trainable))

        self.layers_dic = {}
        self.layers_dic['Shallow_CNN_input'] = self.inputT
        self.num_channel = self.inputT.get_shape().as_list()[-1]

        with tf.variable_scope("Shallow_CNN") as scope:

            if self.reuse:
                scope.reuse_variables()

            self.convlayers()
            self.fc_layers()

        self.logits = self.fc2
        self.probs = tf.nn.softmax(self.logits)

    def act(self, tensor):

        if self.act_type == 'relu':
            return tf.nn.relu(tensor)

        if self.act_type == 'softplus':
            return tf.nn.softplus(tensor)

    def pool(self, tensor):

        if self.pool_type == 'maxpool':

            return tf.nn.max_pool(tensor,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME')

        if self.pool_type == 'avgpool':

            return tf.nn.avg_pool(tensor,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME')

    def convlayers(self):

        # conv1_1
        with tf.variable_scope('Shallow_CNN_conv1_1', reuse=self.reuse) as scope:

            kernel = tf.get_variable(name='w', shape=[2, 2, self.num_channel, 256], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=1e-1), trainable=self.trainable)

            biases = tf.get_variable(name='b', shape=[256], dtype=tf.float32,
                                     initializer=tf.constant_initializer(value=0.0), trainable=self.trainable)

            conv = tf.nn.conv2d(self.inputT, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)

            self.conv1_1 = self.act(tensor=out)

            self.layers_dic['Shallow_CNN_conv1_1'] = self.conv1_1

        # conv1_2
        with tf.variable_scope('Shallow_CNN_conv1_2', reuse=self.reuse) as scope:

            kernel = tf.get_variable(name='w', shape=[2, 2, 256, 256], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=1e-1), trainable=self.trainable)

            biases = tf.get_variable(name='b', shape=[256], dtype=tf.float32,
                                     initializer=tf.constant_initializer(value=0.0), trainable=self.trainable)

            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)

            self.conv1_2 = self.act(tensor=out)

            self.layers_dic['Shallow_CNN_conv1_2'] = self.conv1_2

        # pool1
        self.pool1 = self.pool(tensor=self.conv1_2)
        self.layers_dic['Shallow_CNN_pool1'] = self.pool1

    def fc_layers(self):

        # fc1
        with tf.variable_scope('Shallow_CNN_fc1', reuse=self.reuse) as scope:

            shape = int(np.prod(self.pool1.get_shape()[1:]))

            fc1w = tf.get_variable(name='w', shape=[shape, 1024], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=1e-1), trainable=self.trainable)

            fc1b = tf.get_variable(name='b', shape=[1024], dtype=tf.float32,
                                     initializer=tf.constant_initializer(value=0.0), trainable=self.trainable)

            pool1_flat = tf.reshape(self.pool1, [-1, shape])

            fc1l = tf.nn.bias_add(tf.matmul(pool1_flat, fc1w), fc1b)

            self.fc1 = self.act(tensor=fc1l)

            self.layers_dic['Shallow_CNN_fc1'] = self.fc1

        # fc2
        with tf.variable_scope('Shallow_CNN_fc2', reuse=self.reuse) as scope:

            fc2w = tf.get_variable(name='w', shape=[1024, self.output_dim], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=1e-1), trainable=self.trainable)

            fc2b = tf.get_variable(name='b', shape=[self.output_dim], dtype=tf.float32,
                                     initializer=tf.constant_initializer(value=0.0), trainable=self.trainable)

            self.fc2 = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)

            self.layers_dic['Shallow_CNN_logits'] = self.fc2

    def init(self, sess):
        sess.run(tf.global_variables_initializer())