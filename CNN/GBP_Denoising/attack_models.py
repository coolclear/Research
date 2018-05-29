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

model_types = ['End2End', 'Resnet']
datasets = ['CIFAR10', 'CIFAR100', 'SVHN']
attacks = ['FGM']

model_type = "Resnet"
data_set = "CIFAR10"

eval_params = {'batch_size': 128}
size = 100

def graph(input_ph):

    print("Model Type = {}, Data Set = {}".format(model_type, data_set))

    if data_set == "CIFAR100":
        output_dim = 100
    else:
        output_dim = 10

    checkpoint_dir = "Models/{}_{}".format(data_set, model_type)

    if model_type == 'End2End':
        _, tf_model = prepare_GBP_End2End(output_dim, inputT=input_ph, checkpoint_dir=checkpoint_dir, reuse=True)
    else:
        _, tf_model = prepare_Resnet(output_dim, inputT=input_ph, checkpoint_dir=checkpoint_dir, reuse=True)

    return tf_model.logits

def main():

    for type in model_types:
        model_type = type

        for set in datasets:
            data_set = set

            if data_set == 'CIAFR10':
                (x_train, y_train), (x_test, y_test) = prepare_CIFAR10()
                num_classes = 10
                input_dim = 32
            elif data_set == 'CIFAR100':
                (x_train, y_train), (x_test, y_test) = prepare_CIFAR100()
                num_classes = 100
                input_dim = 32
            else:
                (x_train, y_train), (x_test, y_test) = prepare_SVHN("./")
                num_classes = 10
                input_dim = 32

            tf.reset_default_graph()  # erase whatever the previous graph

            # prepare the input/output placeholders
            x = tf.placeholder(tf.float32, [None, input_dim, input_dim, 3])
            y = tf.placeholder(tf.float32, [None, num_classes])

            # create an attackable model for the cleverhans lib
            # we are doing a wrapping
            preds = graph(x)
            model = CallableModelWrapper(graph, 'logits')


            with tf.Session() as sess:

                # apply the attacks
                for attack in attacks:

                    attack = FastGradientMethod(model, sess=sess)
                    adv_x = attack.generate(x)
                    preds_adv = graph(adv_x)

                    accuracy = model_eval(sess, x, y, preds, x_test[:size], y_test[:size],
                                          args=eval_params)
                    print('Test accuracy on legitimate examples: %0.4f' % accuracy)

                    accuracy = model_eval(sess, x, y, preds_adv, x_test[:size], y_test[:100],
                                          args=eval_params)
                    print('Test accuracy on adversarial examples: %0.4f' % accuracy)



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()