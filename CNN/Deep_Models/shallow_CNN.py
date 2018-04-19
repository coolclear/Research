import numpy as np
import tensorflow as tf
import sys
sys.path.append('/home/yang/Research/')

class Shallow_CNN(object):

    def __init__(self, inputPH=None, sess=None,
                 act_type='relu', pool_type='maxpool', trainable=False,
                 input_dim=32, output_dim=100):

        self.act_type = act_type
        self.pool_type = pool_type
        self.trainable = trainable

        print("Shallow CNN Trainable? {}".format(trainable))

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layers_dic = {}

        # zero-mean input
        with tf.name_scope('input') as scope:
            if inputPH == None:
                self.images = tf.placeholder(tf.float32, [None, self.input_dim, self.input_dim, 3])
                self.layers_dic['images'] = self.images
            else:
                print('Using given input placeholder')
                self.images = inputPH
                self.layers_dic['images'] = self.images

        with tf.name_scope('output') as scope:
            self.labels = tf.placeholder(tf.float32, [None, self.output_dim])

        self.convlayers()
        self.fc_layers()

        self.logits = self.fc2
        self.probs = tf.nn.softmax(self.logits)
        self.maxlogit = tf.reduce_max(self.logits, axis=1)

        if sess is not None:
            self.init(sess)
            print('Initialized ...')
        else:
            print("Initialization failed ... ")

    def act(self, tensor, name):

        if self.act_type == 'relu':
            return tf.nn.relu(tensor, name = name)

        if self.act_type == 'softplus':
            return tf.nn.softplus(tensor, name = name)

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
        with tf.name_scope('conv1_1') as scope:

            kernel = tf.Variable(tf.truncated_normal([2, 2, 3, 256], dtype=tf.float32, stddev=1e-1),
                                 trainable=self.trainable,
                                 name='w_conv1_1')

            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=self.trainable,
                                 name='b_conv1_1')

            conv = tf.nn.conv2d(self.images, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)

            self.conv1_1 = self.act(tensor=out, name=scope)

            self.layers_dic['conv1_1'] = self.conv1_1

        # conv1_2
        with tf.name_scope('conv1_2') as scope:

            kernel = tf.Variable(tf.truncated_normal([2, 2, 256, 256], dtype=tf.float32, stddev=1e-1),
                                 trainable=self.trainable,
                                 name='w_conv1_2')

            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=self.trainable,
                                 name='b_conv1_2')

            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)

            self.conv1_2 = self.act(tensor=out, name=scope)

            self.layers_dic['conv1_2'] = self.conv1_2

        # # pool1
        # self.pool1 = self.pool(tensor=self.conv1_2, name='pool1')
        # self.layers_dic['pool1'] = self.pool1

    def fc_layers(self):

        # fc1
        with tf.name_scope('fc1') as scope:

            shape = int(np.prod(self.conv1_2.get_shape()[1:]))

            fc1w = tf.Variable(tf.truncated_normal([shape, 1024], dtype=tf.float32, stddev=1e-1),
                               trainable=self.trainable,
                               name='w_fc1')

            fc1b = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32),
                               trainable=self.trainable,
                               name='b_fc1')

            pool1_flat = tf.reshape(self.conv1_2, [-1, shape])

            fc1l = tf.nn.bias_add(tf.matmul(pool1_flat, fc1w), fc1b)
            self.fc1 = self.act(tensor=fc1l, name=scope)

            self.layers_dic['fc1'] = self.fc1

        # fc2
        with tf.name_scope('fc2') as scope:

            fc2w = tf.Variable(tf.truncated_normal([1024, self.output_dim], dtype=tf.float32, stddev=1e-1),
                               trainable=self.trainable,
                               name='w_fc2')

            fc2b = tf.Variable(tf.constant(0.0, shape=[self.output_dim], dtype=tf.float32),
                               trainable=self.trainable,
                               name='b_fc2')

            self.fc2 = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)

            self.layers_dic['fc2'] = self.fc2

    def init(self, sess):
        sess.run(tf.global_variables_initializer())