import pickle

from sklearn.cross_validation import train_test_split

with open("../../dataset/04_tr_filled_data.p", 'rb') as f:
    training_dataset = pickle.load(f)
    y = training_dataset['target']

    del training_dataset['ID']
    del training_dataset['target']

    xtr, xts, ytr, yts = train_test_split(training_dataset, y, test_size=0.20, random_state=13)
    #
    # print(xtr)
    # print(xts)
    # print(ytr)
    # print(yts)
    #

    with open('../../dataset/05_split_xtr.p', 'wb') as handle:
        pickle.dump(xtr, handle)
    with open('../../dataset/05_split_xts.p', 'wb') as handle:
        pickle.dump(xts, handle)
    with open('../../dataset/05_split_ytr.p', 'wb') as handle:
        pickle.dump(ytr, handle)
    with open('../../dataset/05_split_yts.p', 'wb') as handle:
        pickle.dump(yts, handle)
