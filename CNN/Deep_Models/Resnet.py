import numpy as np
import tensorflow as tf
import sys
sys.path.append('/home/yang/Research/')

class Resnet(object):

    def __init__(self, inputT=None,
                 input_dim=32, output_dim=100, act_type='relu', pool_type='maxpool',
                 res_blocks=5, phase=False, keepprop=0.5):

        """
        Construct a Resnet object.
        Total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
        :param inputT: input has to be a tensor; has to be provided when constructing this class
        :param input_dim:
        :param output_dim:
        :param act_type:
        :param pool_type:
        :param res_blocks:
        :param phase: False for testing phase and True for training phase
        :param keepprop: the keep probability for the dropout layer
        """

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act_type = act_type
        self.pool_type = pool_type
        self.res_blocks = res_blocks

        self.phase = phase
        self.keepprop = keepprop

        print("Phase = {}".format(self.phase))
        print("Keep probability = {}".format(self.keepprop))

        self.layers_dic = {}

        self.inputT = inputT
        self.layers_dic['Resnet_input'] = self.inputT
        self.num_channel = self.inputT.get_shape().as_list()[-1]

        with tf.name_scope('Resnet_phase') as scope:
            # by default we are in the testing phase
            self.phase = tf.placeholder_with_default(tf.constant(self.phase, dtype=tf.bool), [], name='phase')

        with tf.name_scope('Resnet_keepprob') as scope:
            # by default the drop probability is 0.5
            self.kp = tf.placeholder_with_default(tf.constant(self.keepprop, dtype=tf.float32), [], name='keepporb')

        # Build the TF computational graph for the ResNet architecture
        self.logits = self.build()
        self.probs = tf.nn.softmax(self.logits)

    def build(self):

        # we are stacking the layers
        # and thus we need an easy reference to the last layer of the current graph

        last_layer = self.inputT # starting with the input tensor of course

        with tf.variable_scope('Resnet_conv0'):

            conv0 = self.conv_bn_relu_layer(last_layer, [3, 3, self.num_channel, 16], 1)
            self.layers_dic['Resnet_conv0'] = conv0
            last_layer = conv0

        for i in range(self.res_blocks):

            # notice that for each residual block
            # we actually have two layers in it

            name = 'Resnet_conv1_%d' % i

            with tf.variable_scope(name):

                if i == 0:
                    conv1 = self.residual_block(last_layer, 16, first_block=True)
                else:
                    conv1 = self.residual_block(last_layer, 16)

                self.layers_dic[name] = conv1
                last_layer = conv1

        for i in range(self.res_blocks):

            # notice that for each residual block
            # we actually have two layers in it

            name = 'Resnet_conv2_%d' % i

            with tf.variable_scope(name):

                conv2 = self.residual_block(last_layer, 32)
                self.layers_dic[name] = conv2
                last_layer = conv2

        for i in range(self.res_blocks):

            # notice that for each residual block
            # we actually have two layers in it

            name = 'Resnet_conv3_%d' % i

            with tf.variable_scope(name):

                conv3 = self.residual_block(last_layer, 64)
                self.layers_dic[name] = conv3
                last_layer = conv3

        last_layer = tf.nn.dropout(last_layer, self.kp)
        self.layers_dic['Resnet_dropout'] = last_layer

        with tf.variable_scope('Resnet_fc'):

            channels = last_layer.get_shape().as_list()[-1]

            bn_layer = self.batch_normalization_layer(last_layer) # batch normalization

            relu_layer = tf.nn.relu(bn_layer)

            global_pool = tf.reduce_mean(relu_layer, [1, 2])

            input_dim = global_pool.get_shape().as_list()[-1]

            fc_w = tf.Variable(tf.truncated_normal([input_dim, self.output_dim],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='w')

            fc_b = tf.Variable(tf.constant(1.0, shape=[self.output_dim], dtype=tf.float32), name='b')

            fc_h = tf.matmul(global_pool, fc_w) + fc_b

            self.layers_dic['Resnet_logits'] = fc_h

        return fc_h

    def batch_normalization_layer(self, input_layer):

        '''
        batch normalization
        :param input_layer: the input tensor
        :return: a tensor batch normed
        '''

        bn_layer = tf.contrib.layers.batch_norm(input_layer,
                                          center=True,
                                          scale=True,
                                          is_training=self.phase)

        return bn_layer

    def conv_bn_relu_layer(self, input_layer, filter_shape, stride):
        '''
        Helper: conv, batch normalize and relu the input tensor sequentially
        :param input_layer: 4D tensor
        :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
        :param stride: stride size for conv
        :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
        '''

        out_channel = filter_shape[-1]
        filter = tf.Variable(tf.truncated_normal(shape=filter_shape, dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
        bn_layer = self.batch_normalization_layer(conv_layer)
        output = tf.nn.relu(bn_layer)
        return output

    def bn_relu_conv_layer(self, input_layer, filter_shape, stride):

        '''
        Helper: batch normalize, relu and conv the input layer sequentially
        :param input_layer: 4D tensor
        :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
        :param stride: stride size for conv
        :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
        '''

        in_channel = input_layer.get_shape().as_list()[-1]
        bn_layer = self.batch_normalization_layer(input_layer)
        relu_layer = tf.nn.relu(bn_layer)
        filter = tf.Variable(tf.truncated_normal(shape=filter_shape, dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
        return conv_layer

    def residual_block(self, input_layer, output_channel, first_block=False):

        '''
        A Residual Block
        :param input_layer: 4D tensor
        :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
        :param first_block: if this is the first residual block of the whole network
        :return: 4D tensor.
        '''

        input_channel = input_layer.get_shape().as_list()[-1]

        # when it's time to "shrink" the image size (and double the number of filters), we use stride = 2
        # so no pooling layers
        if input_channel * 2 == output_channel:
            increase_dim = True
            stride = 2
        elif input_channel == output_channel:
            increase_dim = False
            stride = 1
        else:
            raise ValueError('Output and input channel does not match in residual blocks!!!')

        # The first conv layer of the first residual block does not need to be normalized and relu-ed.
        with tf.variable_scope('conv1_in_block'):
            if first_block:
                filter = tf.Variable(tf.truncated_normal(shape=[3, 3, input_channel, output_channel], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
            else:
                conv1 = self.bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)

        with tf.variable_scope('conv2_in_block'):
            conv2 = self.bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1)

        # When the channels of input layer and conv2 does not match, we add zero pads to increase the
        #  depth of input layers
        if increase_dim is True:
            pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                          input_channel // 2]])
        else:
            padded_input = input_layer

        output = conv2 + padded_input
        return output

    def init(self, sess):
        sess.run(tf.global_variables_initializer())