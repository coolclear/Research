import os
import sys
sys.path.append('/home/yang/Research/CNN/')
sys.path.append('/home/yang/Research/CNN/Tools')
from Prepare_Model import prepare_GBPdenoising_end2end, prepare_resnet
from Prepare_Data import prepare_CIFAR10, prepare_CIFAR100, prepare_SVHN, pickle_load

import numpy as np
from setup_cifar import CIFAR, CIFARModel

def main():

    tags = ["GBP_0",
            "GBP_1",
            "GBP_2",
            "GBP_3",
            "GBP_4",
            "ORI"]

    model = prepare_resnet(load_weights='random', sess=None, num_classes=100)

    for tag in tags:

        data = CIFAR(tag)

        print('Accuracy on {} - Training : '.format(tag),
              np.mean(np.argmax(model.predict(data.train_data), axis=1) == np.argmax(data.train_labels,
                                                                                         axis=1)))

        print('Accuracy on {} - Testing : '.format(tag),
              np.mean(np.argmax(model.predict(data.test_data), axis=1) == np.argmax(data.test_labels,
                                                                                         axis=1)))

if __name__ == "__main__":
    main()