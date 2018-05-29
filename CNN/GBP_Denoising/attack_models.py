import os
import sys
sys.path.append('/home/yang/Research/CNN/')
from Prepare_Data import pickle_load, prepare_CIFAR10, prepare_CIFAR100, prepare_SVHN
from Prepare_Model import prepare_GBP_End2End, prepare_Resnet

sys.path.append('/home/yang/Research/CNN/Tools')
from Plot import simple_plot

import numpy as np
import tensorflow as tf
import pickle as pkl

from cleverhans.model import CallableModelWrapper
from cleverhans.utils_tf import model_eval
from cleverhans.attacks import FastGradientMethod

output_dim = 10
input_dim = 32

eval_params = {'batch_size': 128}
size = 100

def main():

    (x_train, y_train), (x_test, y_test) = prepare_CIFAR10()

    with tf.Session() as sess:

        # prepare the input/output placeholders
        x = tf.placeholder(tf.float32, [None, input_dim, input_dim, 3])
        y = tf.placeholder(tf.float32, [None, 1])

        _, tf_model = \
            prepare_Resnet(output_dim, inputT=x, checkpoint_dir="./Models", reuse=True,
                           sess=sess)

        # create an attackable model for the cleverhans lib
        # we are doing a wrapping
        model = CallableModelWrapper(lambda input: tf_model.logits, 'logits')
        attack = FastGradientMethod(model, sess=sess)
        adv_x = attack.generate(x)

        sess.run(adv_x, feed_dic={x: x_test[:100]})

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()