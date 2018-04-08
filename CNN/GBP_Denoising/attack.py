import os
import sys
sys.path.append('/home/yang/Research/CNN/')
from Prepare_Model import prepare_GBPdenoising_end2end
sys.path.append('/home/yang/Research/CNN/Tools')
from Plot import simple_plot
from logging import warning
import numpy as np
import tensorflow as tf
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

    sess = tf.Session()

    # load in the trained model
    tf_model = prepare_GBPdenoising_end2end(sess=sess,
                                            trainable=trainable,
                                            saved='./Models/End2End_Trainable_{}.ckpt'.format(trainable))

    # what's the testing accuracy?
    test_accu = 0.
    for i in range(int(len(x_test) / batch_size)):

        # prepare batch
        test_X_batch = x_test[batch_size * i: batch_size * i + batch_size]
        test_y_batch = y_test[batch_size * i: batch_size * i + batch_size]

        # accumulate
        test_accu += \
            sess.run(tf_model.accuracy,
                     feed_dict={tf_model.inputs: test_X_batch,
                                tf_model.labels: keras.utils.to_categorical(test_y_batch, 10)}) * batch_size

    msg = "Test Accuracy = {:.4f}".format(test_accu / len(x_test))
    print(msg)

    # foolbox - construct a tensorflow model
    fool_model = TensorFlowModel(tf_model.inputs, tf_model.output, bounds=(0, 255))

    # calculate the adversarial examples on some testing images
    for index, image in enumerate(x_test[:10]):

        # define the criterion
        criterion = TargetClass((y_test[index] + 3) % 10) # target on a wrong label
        # criterion1 = Misclassification() # as long as misclassify

        attack_one_image(image, 'TEST_{}'.format(index), y_test[index], 'FGSM', criterion, fool_model)


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
                attack = foolbox.attacks.FGSM(fmodel, criterion=criterion)

            elif attack_type == "IterGS":
                attack = foolbox.attacks.IterativeGradientSignAttack(fmodel, criterion=criterion)

            elif attack_type == "SalMap":
                attack = foolbox.attacks.SaliencyMapAttack(fmodel, criterion=criterion)

            else:
                print("Unknown attack type! Using FGSM")
                attack = foolbox.attacks.FGSM(fmodel, criterion=criterion)

            # attack happens here
            adversarial = attack(image, label)

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