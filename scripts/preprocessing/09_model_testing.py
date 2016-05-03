import pickle
from sklearn.metrics import log_loss


def run_model_prediction(standardize=False, model_type='rf'):
    standardized = "_standardized" if standardize else ""
    model_file_name = "08_reg_model_" + model_type + standardized + ".p"

    print("Loading Training and validation sets\n")
    with open("../../dataset/05_split_xtr.p", 'rb') as h:
        xtr = pickle.load(h)

    with open("../../dataset/05_split_xts.p", 'rb') as h:
        xts = pickle.load(h)

    with open("../../dataset/05_split_ytr.p", 'rb') as h:
        ytr = pickle.load(h)

    with open("../../dataset/05_split_yts.p", 'rb') as h:
        yts = pickle.load(h)

    if standardize:
        print("Perform standardization\n")
        with open("../../dataset/06_standardizer.p", 'rb') as h:
            scaler = pickle.load(h)

        # print(xtr.shape)

        xtr = scaler.transform(xtr)
        xts = scaler.transform(xts)

    print("Loading Model\n")
    with open("../../dataset/" + model_file_name) as handle:
        model = pickle.load(handle)

    xtr_pred = model.predict_proba(xtr)

    # # Debug
    # print(xtr_pred)
    # print(xtr_pred.shape)

    xts_pred = model.predict_proba(xts)
    if model_type == 'rf':
        print("Predict using Random Forest Classifier")
    elif model_type == 'svc':
        print("Predict using SVM Classifier")
    elif model_type == 'log_reg':
        print("Predict using Logistic Regressor")
    elif model_type == 'extra_trees_classifier':
        print("Predict using Extra Decision Trees Classifier")

    print("Log Loss fn results")

    err_xtr = log_loss(ytr, xtr_pred[:, [1]], eps=1e-14)
    err_xts = log_loss(yts, xts_pred[:, [1]], eps=1e-14)

    print("Training Set Error: " + str(err_xtr))
    print("Validation Set Error: " + str(err_xts))


if __name__ == "__main__":
    # Model options:
    # "log_reg",
    # "rf"(Random Forest),
    # "svc"(Support Vector Classification)
    # "extra_trees_classifier" (Extra Decision Trees Classifier)

    model_option = "rf"

    run_model_prediction(standardize=True, model_type=model_option)
