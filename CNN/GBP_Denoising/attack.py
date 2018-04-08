import os
import sys
sys.path.append('/home/yang/Research/CNN/')
from Prepare_Model import prepare_GBPdenoising_end2end
sys.path.append('/home/yang/Research/CNN/Tools')
from Plot import simple_plot
from logging import warning
import numpy as np
import tensorflow as tf
import keras
from keras.datasets import cifar10

import foolbox
from foolbox.models import TensorFlowModel
from foolbox.criteria import\
    Misclassification,\
    TopKMisclassification,\
    OriginalClassProbability,\
    TargetClass,\
    TargetClassProbability

trainable = False
batch_size = 128

def softmax_np(x, axis=None):
    return np.exp(x) / np.sum(np.exp(x), axis=axis)


def main():

    # load in the data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    with tf.Session() as sess:

        # load in the trained model
        tf_model = prepare_GBPdenoising_end2end(sess=sess,
                                                trainable=trainable,
                                                saved='./Models/End2End_Trainable_{}.ckpt'.format(trainable))

        # foolbox - construct a tensorflow model
        fool_model = TensorFlowModel(tf_model.inputs, tf_model.output, bounds=(0, 255))

        # calculate the adversarial examples on some testing images
        for index, image in enumerate(x_test[:30]):

            # define the criterion
            # criterion = TargetClass((y_test[index] + 3) % 10) # target on a wrong label
            criterion = Misclassification() # as long as misclassify

            attack_one_image(image, 'TEST_{}'.format(index), y_test[index], 'LBFG', criterion, fool_model)


def attack_one_image(image, name, label, attack_type, criterion, fool_model):

        print('True Label: {}'.format(label))

        preds = fool_model.predictions(image)
        label_pre = np.argmax(preds)
        prob_pre = np.max(softmax_np(preds))
        print('Prediction : {} ({:.2f})'.format(label_pre, prob_pre))

        if label_pre != label:

            print('The model predicts wrong. No need to attack.')
            return None

        else:

            print('Ok, let us attack this image ... ')

            if attack_type == "FGSM":
                attack = foolbox.attacks.FGSM(fool_model)

            elif attack_type == "IterGS":
                attack = foolbox.attacks.IterativeGradientSignAttack(fool_model)

            elif attack_type == "SalMap":
                attack = foolbox.attacks.SaliencyMapAttack(fool_model)

            elif attack_type == "LBFG":
                attack = foolbox.attacks.LBFGSAttack(fool_model)

            else:
                print("Unknown attack type! Using FGSM")
                attack = foolbox.attacks.FGSM(fool_model)

            # attack happens here
            adversarial = attack(image, int(label))

            preds_adv = fool_model.predictions(adversarial)
            label_pre_adv = np.argmax(preds_adv)
            prob_pre_adv = np.max(softmax_np(preds_adv))
            print('(ADV) Prediction : {} ({:.2f})'.format(label_pre_adv, prob_pre_adv))

            if label_pre_adv != label:

                print('The attack is successed!')

                simple_plot(adversarial, 'ADV' + name, './Adversarial_Examples/')

                print('Saved!')

            else:

                print('The attack failed!')




if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()