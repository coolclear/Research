import numpy as np
from setup_cifar import CIFAR, CIFARModel

np.random.seed(13)
tf.set_random_seed(13)

def main():

    tag = "End2End"
    model = CIFARModel(restore="Models/CIFAR10_{}_Trainable".format(tag), end2end=True).model

    data_gbp = CIFAR("GBP")
    data_ori = CIFAR("ORI")

    # trained on ORI, test on GBP
    if tag == "ORI":

        print('Training Accuracy on ORI:',
              np.mean(np.argmax(model.predict(data_ori.train_data), axis=1) == np.argmax(data_ori.train_labels,
                                                                                         axis=1)))

        print('Testing Accuracy on ORI:',
              np.mean(np.argmax(model.predict(data_ori.test_data), axis=1) == np.argmax(data_ori.test_labels,
                                                                                         axis=1)))

        print('Trained on ORI and Test on GBP(Train):',
              np.mean(np.argmax(model.predict(data_gbp.train_data), axis=1) == np.argmax(data_gbp.train_labels,
                                                                                         axis=1)))

        print('Trained on ORI and Test on GBP(Test):',
              np.mean(np.argmax(model.predict(data_gbp.test_data), axis=1) == np.argmax(data_gbp.test_labels, axis=1)))

    # trained on GBP, test on ORI
    if tag == "GBP":

        print('Training Accuracy on GBP:',
              np.mean(np.argmax(model.predict(data_gbp.train_data), axis=1) == np.argmax(data_gbp.train_labels,
                                                                                         axis=1)))

        print('Testing Accuracy on GBP:',
              np.mean(np.argmax(model.predict(data_gbp.test_data), axis=1) == np.argmax(data_gbp.test_labels,
                                                                                        axis=1)))

        print('Trained on GBP and Test on ORI(Train):',
              np.mean(np.argmax(model.predict(data_ori.train_data), axis=1) == np.argmax(data_ori.train_labels,
                                                                                         axis=1)))
        print('Trained on GBP and Test on ORI(Test):',
              np.mean(
                  np.argmax(model.predict(data_ori.test_data), axis=1) == np.argmax(data_ori.test_labels, axis=1)))

    # trained on ORI, our End2End system
    if tag == "End2End":

        print('End2End Testing Accuracy on ORI:',
              np.mean(np.argmax(model.predict(data_ori.test_data), axis=1) == np.argmax(data_ori.test_labels,
                                                                                        axis=1)))

if __name__ == "__main__":
    main()