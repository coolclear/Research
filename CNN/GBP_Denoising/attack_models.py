import os
import sys
import argparse

dirname = os.path.dirname(__file__)
path = os.path.join(dirname, './../')
sys.path.append(path)

from Prepare_Data import pickle_load, prepare_CIFAR10, prepare_CIFAR100, prepare_SVHN
from Prepare_Model import prepare_GBP_End2End, prepare_Resnet

import numpy as np
import tensorflow as tf
import keras

from cleverhans.model import CallableModelWrapper
from cleverhans.utils_tf import model_eval
from cleverhans.attacks import\
    FastGradientMethod,\
    CarliniWagnerL2,\
    DeepFool,\
    ElasticNetMethod,\
    FastFeatureAdversaries,\
    LBFGS,\
    MadryEtAl,\
    MomentumIterativeMethod,\
    VirtualAdversarialMethod,\
    SaliencyMapMethod,\
    SPSA,\
    vatm

size = 256
eval_params = {'batch_size': 128}

def main(type="Resnet", dataset="CIFAR10", attack_type="FGM"):

    if dataset == 'CIAFR10':
        (_, _), (x_test, y_test) = prepare_CIFAR10()
        num_classes = 10
        input_dim = 32
    elif dataset == 'CIFAR100':
        (_, _), (x_test, y_test) = prepare_CIFAR100()
        num_classes = 100
        input_dim = 32
    else:
        (_, _), (x_test, y_test) = prepare_SVHN("./")
        num_classes = 10
        input_dim = 32

    x_test = x_test / 255.
    y_test = keras.utils.to_categorical(y_test, num_classes)

    with tf.Session() as sess:

        # scopes = []
        input_output = []

        # prepare the output placeholders
        x = tf.placeholder(tf.float32, [None, 32, 32, 3])
        y = tf.placeholder(tf.float32, [None, num_classes])

        def helper1(x, num_classes, dataset, type, sess, input_output):

            if len(input_output) == 0:

                reuse = False

                # Model/Graph
                if type == 'End2End':
                    _, tf_model = \
                        prepare_GBP_End2End(num_classes,
                                            inputT=x, checkpoint_dir='./{}_{}/'.format(dataset, type),
                                            sess=sess, keepprob=1.0, reuse=reuse)
                else:
                    _, tf_model = \
                        prepare_Resnet(num_classes,
                                       inputT=x, checkpoint_dir='./{}_{}/'.format(dataset, type),
                                       sess=sess, keepprob=1.0, reuse=reuse)

                input_output.append(x)
                input_output.append(tf_model.logits)

            else:

                reuse = True

                # Model/Graph
                if type == 'End2End':
                    _, tf_model = \
                        prepare_GBP_End2End(num_classes,
                                            inputT=x, checkpoint_dir=None,
                                            sess=None, keepprob=1.0, reuse=reuse)
                else:
                    _, tf_model = \
                        prepare_Resnet(num_classes,
                                       inputT=x, checkpoint_dir=None,
                                       sess=None, keepprob=1.0, reuse=reuse)

                input_output.append(x)
                input_output.append(tf_model.logits)


            return tf_model.logits

        # create an attackable model for the cleverhans lib
        # we are doing a wrapping
        model = CallableModelWrapper(lambda placeholder: helper1(placeholder, num_classes, dataset, type, sess, input_output), 'logits')

        if attack_type == "FGM": # pass
            attack = FastGradientMethod(model, back='tf', sess=sess)
            params = {
                'eps' : 0.06,
                'clip_min': 0.,
                'clip_max': 1.
            }
        elif attack_type == "CWL2": # pass
            attack = CarliniWagnerL2(model, back='tf', sess=sess)
            params = {
                'confidence': 0.9,
                'batch_size': 128,
                'learning_rate': 0.005,
            }
        elif attack_type == "DF": # pass
            attack = DeepFool(model, back='tf', sess=sess)
            params = {
            }
        elif attack_type == "ENM": # configurations checked, quickly tested
            attack = ElasticNetMethod(model, back='tf', sess=sess)
            params = {
                'confidence': 0.9,
                'batch_size': 128,
                'learning_rate': 0.005,
            }
        elif attack_type == "FFA": # configuration checked
            attack = FastFeatureAdversaries(model, back='tf', sess=sess)
            params = {
                'eps': 0.06,
                'eps_iter': 0.005,
                'clip_min': 0.,
                'clip_max': 1.
            }
        elif attack_type == "LBFGS":
            attack = LBFGS(model, back='tf', sess=sess)
            params = {
                'eps': 0.06,
                'clip_min': 0.,
                'clip_max': 1.
            }
        elif attack_type == "MEA":
            attack = MadryEtAl(model, back='tf', sess=sess)
            params = {
                'eps': 0.06,
                'clip_min': 0.,
                'clip_max': 1.
            }
        elif attack_type == "MIM":
            attack = MomentumIterativeMethod(model, back='tf', sess=sess)
            params = {
                'eps': 0.06,
                'clip_min': 0.,
                'clip_max': 1.
            }
        elif attack_type == "SMM":
            attack = SaliencyMapMethod(model, back='tf', sess=sess)
            params = {
                'eps': 0.06,
                'clip_min': 0.,
                'clip_max': 1.
            }
        elif attack_type == "SPSA":
            attack = SPSA(model, back='tf', sess=sess)
            params = {
                'eps': 0.06,
                'clip_min': 0.,
                'clip_max': 1.
            }
        elif attack_type == "VATM":
            attack = vatm(model, back='tf', sess=sess)
            params = {
                'eps': 0.06,
                'clip_min': 0.,
                'clip_max': 1.
            }
        elif attack_type == "VAM":
            attack = VirtualAdversarialMethod(model, back='tf', sess=sess)
            params = {
                'eps': 0.06,
                'clip_min': 0.,
                'clip_max': 1.
            }
        else:
            raise Exception("I don't recognize {} this attack type. I will use FGM instead.".format(attack_type))

        adv_x = attack.generate(x, **params)
        adv_vals = sess.run(adv_x, feed_dict={x: x_test[:size]})

        # Notice that adv_vals may contain NANs because of the failure of the attack
        # Also the input may not be perturbed at all because of the failure of the attack
        to_delete = []
        for idx, adv in enumerate(adv_vals):
            # for nan
            if np.isnan(adv).any():
                to_delete.append(idx)
            # for no perturbation
            if np.array_equiv(adv, x_test[idx]):
                to_delete.append(idx)

        # cleanings
        adv_vals_cleaned = np.delete(adv_vals, to_delete, axis=0)
        ori_cleaned = np.delete(x_test[:size], to_delete, axis=0)
        y_cleaned = np.delete(y_test[:size], to_delete, axis=0)

        if len(adv_vals_cleaned) == 0:
            print("No adversarial example is generated!")
            return

        print("{} out of {} adversarial examples are generated.".format(len(adv_vals_cleaned), size))

        print("The average L_inf distortion is {}".format(
            np.mean([np.max(np.abs(adv - ori_cleaned[idx])) for idx, adv in enumerate(adv_vals_cleaned)])))

        # TODO: visualize the adv_vals

        accuracy = model_eval(sess, input_output[0], y, tf.nn.softmax(input_output[1]), x_test[:size], y_test[:size],
                              args=eval_params)
        print('Test accuracy on normal examples: %0.4f' % accuracy)

        accuracy = model_eval(sess, input_output[0], y, tf.nn.softmax(input_output[1]), adv_vals_cleaned, y_cleaned,
                              args=eval_params)
        print('Test accuracy on adversarial examples: %0.4f' % accuracy)

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--type", required=True, help="Resnet or End2End")
    ap.add_argument("-d", "--dataset", required=True, help="CIFAR10, CIFAR100 or SVHN")
    ap.add_argument("-a", "--attack", required=True, help="Specify the Attack Type")
    args = vars(ap.parse_args())

    main(type=args["type"], dataset=args["dataset"], attack_type=args["attack"])
