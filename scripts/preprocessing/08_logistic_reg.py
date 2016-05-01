import pickle
import os.path
from sklearn.linear_model import LogisticRegression as lreg
from sklearn.metrics import log_loss as ll


def log_reg(standardize=False):
    model_file_name = "08_log_reg_model_standardized.p" if standardize else "08_log_reg_model.p"

    print("Loading Training and validation sets\n")
    with open("../../dataset/05_split_xtr.p", 'rb') as h:
        xtr = pickle.load(h)

    with open("../../dataset/05_split_xts.p", 'rb') as h:
        xts = pickle.load(h)

    with open("../../dataset/05_split_ytr.p", 'rb') as h:
        ytr = pickle.load(h)

    with open("../../dataset/05_split_yts.p", 'rb') as h:
        yts = pickle.load(h)

    with open("../../dataset/06_standardizer.p", 'rb') as h:
        scaler = pickle.load(h)

    if standardize:
        print("Perform standardization\n")
        print(xtr.shape)
        xtr = scaler.transform(xtr)
        xts = scaler.transform(xts)

    if os.path.isfile("../../dataset/" + model_file_name):
        print("Found trained model, loading...")
        with open("../../dataset/" + model_file_name, 'rb') as h:
            model = pickle.load(h)
    else:
        model = lreg(n_jobs=7, max_iter=200, random_state=13)

        print("Start Training model\n")
        model.fit(xtr, ytr)

        print("Save model")

        with open("../../dataset/" + model_file_name, 'wb') as h:
            pickle.dump(model, h)

    xtr_pred = model.predict_proba(xtr)
    print("First column is output: 0, second column is output:1")

    # # Debug
    # print(xtr_pred)
    # print(xtr_pred.shape)

    xts_pred = model.predict_proba(xts)

    print("Log Loss fn results")

    err_xtr = ll(ytr, xtr_pred[:, [1]], eps=1e-14)
    err_xts = ll(yts, xts_pred[:, [1]], eps=1e-14)

    print("Training Set Error: " + str(err_xtr))
    print("Validation Set Error: " + str(err_xts))


if __name__ == "__main__":
    log_reg(standardize=True)
