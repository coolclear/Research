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

        input_pl = tf_model.inputs
        logits = tf_model.output

        # # predict one by one
        # # double check the testing accuracy
        # # comment out this part if necessary
        # test_accu = 0.
        # for index, image in enumerate(x_test):
        #     batch_image = np.expand_dims(image, 0)
        #     logits_val = sess.run(logits, feed_dict={input_pl: batch_image})
        #     if np.argmax(logits_val) == y_test[index]:
        #         test_accu += 1
        # msg = "Test Accuracy = {:.4f}".format(test_accu / len(x_test))
        # print(msg)

        # foolbox - construct a tensorflow model
        fool_model = TensorFlowModel(input_pl, logits, bounds=(0, 255))

        # criterion
        # criterion = TargetClass((y_test[index] + 3) % 10) # target on a wrong label
        criterion = Misclassification() # as long as misclassify

        # attack type
        attack_type = "Boundary"

        # image index
        index = 23

        attack_one_image(x_test[index], 'TEST_{}'.format(index), y_test[index], attack_type, criterion, fool_model)


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

            ############################ Gradient-based Attacks ########################################################
            ############################################################################################################

            if attack_type == "FGSM":
                attack = foolbox.attacks.FGSM(fool_model, criterion)

            elif attack_type == "IterGS":
                attack = foolbox.attacks.IterativeGradientSignAttack(fool_model, criterion)

            elif attack_type == "IterG":
                attack = foolbox.attacks.IterativeGradientAttack(fool_model, criterion)

            elif attack_type == "LBFG":
                attack = foolbox.attacks.LBFGSAttack(fool_model, criterion)

            elif attack_type == "DeepFool":
                attack = foolbox.attacks.DeepFoolAttack(fool_model, criterion)

            elif attack_type == "SalMap":
                attack = foolbox.attacks.SaliencyMapAttack(fool_model, criterion)

            ############################################################################################################
            ############################ Gradient-based Attacks ########################################################



            ############################ Score-based Attacks ###########################################################
            ############################################################################################################

            elif attack_type == "SinPix":
                attack = foolbox.attacks.SinglePixelAttack(fool_model, criterion)

            elif attack_type == "LocalSearch":
                attack = foolbox.attacks.LocalSearchAttack(fool_model, criterion)

            ############################################################################################################
            ############################ Score-based Attacks ###########################################################



            ############################ Decision-based Attacks ###########################################################
            ############################################################################################################

            elif attack_type == "Boundary":
                attack = foolbox.attacks.BoundaryAttack(fool_model, criterion)

            elif attack_type == "Blur":
                attack = foolbox.attacks.GaussianBlurAttack(fool_model, criterion)

            elif attack_type == "Contrast":
                attack = foolbox.attacks.ContrastReductionAttack(fool_model, criterion)

            elif attack_type == "Noise":
                attack = foolbox.attacks.AdditiveUniformNoiseAttack(fool_model, criterion)

            ############################################################################################################
            ############################ Score-based Attacks ###########################################################

            else:
                print("Unknown attack type! Using FGSM")
                attack = foolbox.attacks.FGSM(fool_model)

            # attack happens here
            adversarial = attack(image, label[0])

            """
            Notice that for a given input image if we run the model multiple times the predictions
            could be different because of the random logits in the GBP Reconstruction. 
            """

            if adversarial is None:

                # if the attack above fails, it will return None and we catch it here
                print('The attack failed!')

            elif np.array_equal(adversarial, image):

                # the prediction for this image is not stable
                #  because of the random logit in the GBP Reconstruction
                print('No attack at all, the prediction itself is not stable')

            else :

                preds_adv = fool_model.predictions(adversarial)
                label_pre_adv = np.argmax(preds_adv)
                prob_pre_adv = np.max(softmax_np(preds_adv))
                print('(ADV) Prediction : {} ({:.2f})'.format(label_pre_adv, prob_pre_adv))

                if label_pre_adv != label:

                    print('The attack is successed!')

                    simple_plot(adversarial, 'ADV' + name, './Adversarial_Examples/')

                    print('Saved!')






if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    main()