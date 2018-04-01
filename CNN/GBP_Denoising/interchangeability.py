import numpy as np
from setup_cifar import CIFAR, CIFARModel

def main():

    tag = "GBP"
    model = CIFARModel("Models/CIFAR10_{}".format(tag)).model
    data_gbp = CIFAR("GBP")
    data_ori = CIFAR("ORI")

    # trained on ORI, test on GBP
    if tag == "ORI":

        print('Testing Accuracy on ORI:',
              np.mean(np.argmax(model.predict(data_ori.test_data), axis=1) == np.argmax(data_ori.test_labels,
                                                                                         axis=1)))

        print('Trained on ORI and Test on GBP(Test):',
              np.mean(np.argmax(model.predict(data_gbp.test_data), axis=1) == np.argmax(data_gbp.test_labels, axis=1)))

    # trained on GBP, test on ORI
    if tag == "GBP":

        print('Testing Accuracy on GBP:',
              np.mean(np.argmax(model.predict(data_gbp.test_data), axis=1) == np.argmax(data_gbp.test_labels,
                                                                                        axis=1)))

        print('Trained on GBP and Test on ORI(Test):',
              np.mean(
                  np.argmax(model.predict(data_ori.test_data), axis=1) == np.argmax(data_ori.test_labels, axis=1)))

if __name__ == "__main__":
    main()