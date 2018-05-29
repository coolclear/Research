import os
import sys
import argparse
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

eval_params = {'batch_size': 128}
size = 1000

def main(type="Resnet", dataset="CIFAR10"):

    if dataset == 'CIAFR10':
        (x_train, y_train), (x_test, y_test) = prepare_CIFAR10()
        num_classes = 10
        input_dim = 32
    elif dataset == 'CIFAR100':
        (x_train, y_train), (x_test, y_test) = prepare_CIFAR100()
        num_classes = 100
        input_dim = 32
    else:
        (x_train, y_train), (x_test, y_test) = prepare_SVHN("./")
        num_classes = 10
        input_dim = 32

    with tf.Session() as sess:

        # prepare the input/output placeholders
        x = tf.placeholder(tf.float32, [None, input_dim, input_dim, 3])
        y = tf.placeholder(tf.float32, [None, 1])

        # Model/Graph
        if type == 'End2End':
            _, tf_model = \
                prepare_GBP_End2End(num_classes, inputT=x, checkpoint_dir="./{}_{}".format(dataset, type), sess=sess)
        else:
            _, tf_model = \
                prepare_Resnet(num_classes, inputT=x, checkpoint_dir="./{}_{}".format(dataset, type), sess=sess)

        # create an attackable model for the cleverhans lib
        # we are doing a wrapping
        model = CallableModelWrapper(lambda ph_of_ph: tf_model.logits, 'logits')
        attack = FastGradientMethod(model, sess=sess)
        adv_x = attack.generate(x)

        # this is where we actually generate the adversarial examples
        adv_vals = sess.run(adv_x, feed_dict={x: x_test[:size]})

        accuracy = model_eval(sess, x, y, tf_model.logits, x_test[:size], y_test[:size],
                              args=eval_params)
        print('Test accuracy on normal examples: %0.4f' % accuracy)

        accuracy = model_eval(sess, x, y, tf_model.logits, adv_vals, y_test[:size],
                              args=eval_params)
        print('Test accuracy on adversarial examples: %0.4f' % accuracy)

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--type", required=True, help="Resnet or End2End")
    ap.add_argument("-d", "--dataset", required=True, help="CIFAR10, CIFAR100 or SVHN")
    args = vars(ap.parse_args())

    main(type=args["type"], dataset=["dataset"])