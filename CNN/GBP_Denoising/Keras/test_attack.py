## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

## Modified by Yang Zhang <yz78@rice.edu>

import tensorflow as tf
import numpy as np
import time

from setup_cifar import CIFAR, CIFARModel
from l2_attack import CarliniL2

def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.
    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1, 1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            for j in seq:
                if (j == np.argmax(data.test_labels[start + i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start + i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start + i])
            targets.append(data.test_labels[start + i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets


if __name__ == "__main__":

    with tf.Session() as sess:

        data = CIFAR("ORI")

        Model = CIFARModel(restore="Models/CIFAR10_End2End_Trainable", end2end=True)

        attack = CarliniL2(sess, Model, batch_size=9, max_iterations=1000, confidence=0)

        inputs, targets = generate_data(data, samples=1, targeted=True,
                                        start=0, inception=False)

        timestart = time.time()
        adv = attack.attack(inputs, targets)
        timeend = time.time()

        print("Took", timeend - timestart, "seconds to run", len(inputs), "samples.")

        for i in range(len(adv)):

            print("Originally Prediction : ", Model.model.predict(inputs[i]))

            print("Adversarial Prediction : ", Model.model.predict(adv[i]))

            print("Total Distortion : ", np.sum((adv[i] - inputs[i]) ** 2) ** .5)