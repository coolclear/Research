# trained on ORI, test on GBP
if tag == "ORI":
    data_gbp = CIFAR("GBP")

    print('Trained on ORI and Test on GBP(Train):',
          np.mean(np.argmax(model.predict(data_gbp.train_data), axis=1) == np.argmax(data_gbp.train_labels,
                                                                                     axis=1)))

    print('Trained on ORI and Test on GBP(Test):',
          np.mean(np.argmax(model.predict(data_gbp.test_data), axis=1) == np.argmax(data_gbp.test_labels, axis=1)))

# trained on GBP, test on ORI
if tag == "GBP":
    data_ori = CIFAR("ORI")

    print('Trained on GBP and Test on ORI(Train):',
          np.mean(np.argmax(model.predict(data_ori.train_data), axis=1) == np.argmax(data_ori.train_labels,
                                                                                     axis=1)))
    print('Trained on GBP and Test on ORI(Test):',
          np.mean(
              np.argmax(model.predict(data_ori.test_data), axis=1) == np.argmax(data_ori.test_labels, axis=1)))
