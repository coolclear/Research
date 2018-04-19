import numpy as np
from setup_cifar import CIFAR, CIFARModel

def main():

    tags = ["GBP_0",
            "GBP_1",
            "GBP_2",
            "GBP_3",
            "GBP_4",
            "ORI"]

    model = CIFARModel(restore="Models/CIFAR10_GBP_0").model

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