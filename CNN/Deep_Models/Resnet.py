import tensorflow as tf

class Resnet(object):

    def __init__(self, inputT, output_dim,
                 act_type='relu', reuse=False, res_blocks=5):

        print(" [*] Constructing Resnet ... ")

        """
        Construct a Resnet object.
        Total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
        :param inputT: has to be a tensor; has to be provided
        :param input_dim: has to be provided
        :param output_dim: has to be provided
        :param act_type: relu by default
        :param pool_type: max pooling by default
        :param reuse: False for training; True for testing
        :param res_blocks: 5 by default
        :param phase: False for testing; True for training
        :param keepprob: 1.0 by default
        """

        self.inputT = inputT
        self.output_dim = output_dim

        self.act_type = act_type

        self.reuse = reuse
        self.res_blocks = res_blocks

        print("Output dim = {}".format(output_dim))
        print("Reuse = {}, (T)Testing/(F)Training".format(self.reuse))

        self.layers_dic = {}

        self.layers_dic['Resnet_input'] = self.inputT
        self.num_channel = self.inputT.get_shape().as_list()[-1]

        with tf.variable_scope("Resnet") as scope:

            if self.reuse:
                scope.reuse_variables()

            with tf.name_scope('Resnet_phase') as scope:
                # by default we are in the testing phase
                self.phase = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), [], name='phase')

            with tf.name_scope('Resnet_keepprob') as scope:
                # by default the drop probability is 0.0
                self.kp = tf.placeholder_with_default(tf.constant(1.0, dtype=tf.float32), [], name='keepporb')

            # Build up the computational graph for the Resnet architecture
            self.logits = self.build()
            self.probs = tf.nn.softmax(self.logits)

    def build(self):

        # we are stacking the layers
        # and thus we need an easy reference to the last layer of the current graph

        last_layer = self.inputT # starting with the input tensor

        with tf.variable_scope('Resnet_conv0', reuse=self.reuse):

            conv0 = self.conv_bn_relu_layer(last_layer, [3, 3, self.num_channel, 16], 1)
            self.layers_dic['Resnet_conv0'] = conv0
            last_layer = conv0

        for i in range(self.res_blocks):

            # notice that for each residual block
            # we actually have two layers in it

            name = 'Resnet_conv1_%d' % i

            with tf.variable_scope(name, reuse=self.reuse):

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

            with tf.variable_scope(name, reuse=self.reuse):

                conv2 = self.residual_block(last_layer, 32)
                self.layers_dic[name] = conv2
                last_layer = conv2

        for i in range(self.res_blocks):

            # notice that for each residual block
            # we actually have two layers in it

            name = 'Resnet_conv3_%d' % i

            with tf.variable_scope(name, reuse=self.reuse):

                conv3 = self.residual_block(last_layer, 64)
                self.layers_dic[name] = conv3
                last_layer = conv3

        last_layer = tf.nn.dropout(last_layer, self.kp)

        self.layers_dic['Resnet_dropout'] = last_layer

        with tf.variable_scope('Resnet_fc', reuse=self.reuse):

            bn_layer = self.batch_normalization_layer(last_layer, self.reuse) # batch normalization

            relu_layer = tf.nn.relu(bn_layer)

            global_pool = tf.reduce_mean(relu_layer, [1, 2])

            input_dim = global_pool.get_shape().as_list()[-1]

            fc_w = tf.get_variable(name='w', shape=[input_dim, self.output_dim], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=1e-1))

            fc_b = tf.get_variable(name='b', shape=[self.output_dim], dtype=tf.float32,
                                     initializer=tf.constant_initializer(value=1.0))

            fc_h = tf.matmul(global_pool, fc_w) + fc_b

            self.layers_dic['Resnet_logits'] = fc_h

        return fc_h

    def batch_normalization_layer(self, input_layer, reuse):

        '''
        batch normalization
        '''

        bn_layer = tf.contrib.layers.batch_norm(input_layer,
                                                center=True,
                                                scale=True,
                                                is_training=self.phase)

        return bn_layer

    def conv_bn_relu_layer(self, input_layer, filter_shape, stride):

        '''
        Helper: Conv, Batch Normalization, Relu sequentially
        :param input_layer: 4D tensor
        :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
        :param stride: stride size for conv
        :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
        '''

        # Conv
        filter = tf.get_variable(name='weights', shape=filter_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=1e-1))
        conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')

        # BN
        bn_layer = self.batch_normalization_layer(conv_layer, self.reuse)

        # Relu
        output = tf.nn.relu(bn_layer)

        return output

    def bn_relu_conv_layer(self, input_layer, filter_shape, stride):

        '''
        Helper: batch normalization, relu, conv sequentially
        :param input_layer: 4D tensor
        :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
        :param stride: stride size for conv
        :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
        '''

        bn_layer = self.batch_normalization_layer(input_layer, self.reuse) # BN

        relu_layer = tf.nn.relu(bn_layer) # Relu

        # Conv
        filter = tf.get_variable(name='weights', shape=filter_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=1e-1))
        conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')

        return conv_layer

    def residual_block(self, input_layer, output_channel, first_block=False):

        '''
        A Residual Block
        :param input_layer: 4D tensor
        :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
        :param first_block: indicate if this is the first residual block of the entire network
        :return: 4D tensor.
        '''

        ############################################### Analysis #######################################################

        input_channel = input_layer.get_shape().as_list()[-1]
        if input_channel * 2 == output_channel:
            increase_dim = True
            stride = 2
        elif input_channel == output_channel:
            increase_dim = False
            stride = 1
        else:
            raise ValueError('Output and input channel does not match in residual blocks!!!')

        ############################################### Analysis #######################################################





        ######################################### 2 Times Convolution ##################################################

        # The first conv layer of the first residual block does not need to be normalized and relu-ed.
        with tf.variable_scope('conv1_in_block'):
            if first_block:
                filter = tf.get_variable(name='weights', shape=[3, 3, input_channel, output_channel], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=1e-1))
                conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
            else:
                conv1 = self.bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)

        with tf.variable_scope('conv2_in_block'):
            conv2 = self.bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1)

        ######################################### 2 Times Convolution ##################################################





        ############################################ Skip Connection ###################################################

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

        ############################################ Skip Connection ###################################################

        return output

    def init(self, sess):
        sess.run(tf.global_variables_initializer())