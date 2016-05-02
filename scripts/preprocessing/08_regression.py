import pickle
import os.path
from sklearn.linear_model import LogisticRegression as lreg
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier


def get_datasets(for_submission):
    if for_submission:
        print("Loading Training and test sets\n")
        with open("../../dataset/04_tr_filled_data.p", 'rb') as h:
            xtr = pickle.load(h)

        ytr = xtr['target']
        del (xtr['ID'])
        del (xtr['target'])

        # # Debug
        # print(xtr)
        # print(ytr)

        with open("../../dataset/04_ts_filled_data.p", 'rb') as h:
            xts = pickle.load(h)
            del (xts['ID'])

            # # Debug
            # print(xts)
        return xtr, xts, ytr, None
    else:
        print("Loading Training and validation sets\n")
        with open("../../dataset/05_split_xtr.p", 'rb') as h:
            xtr = pickle.load(h)

        with open("../../dataset/05_split_xts.p", 'rb') as h:
            xts = pickle.load(h)

        with open("../../dataset/05_split_ytr.p", 'rb') as h:
            ytr = pickle.load(h)

        with open("../../dataset/05_split_yts.p", 'rb') as h:
            yts = pickle.load(h)
        return xtr, xts, ytr, yts


def get_model(model_typ):
    def random_forest():
        print("Creating Random Forest Classifier")
        return RFC(max_features="log2", max_depth=10)

    #              n_estimators=10,)
    #              tors=10,
    #              criterion="gini",
    #              max_depth=None,
    #              min_samples_split=2,
    #              min_samples_leaf=1,
    #              min_weight_fraction_leaf=0.,
    #              max_features="auto",
    #              max_leaf_nodes=None,
    #              bootstrap=True,
    #              oob_score=False,
    #              n_jobs=1,
    #              random_state=None,
    #              verbose=0,
    #              warm_start=False,
    #              class_weight=None):

    def log_regression():
        print("Creating Logistic Regression Model for Classification")
        return lreg(n_jobs=7, C=10, verbose=1, max_iter=300)
        # penalty='l2', dual=False, tol=1e-4, C=1.0,
        #         fit_intercept=True, intercept_scaling=1, class_weight=None,
        #         random_state=None, solver='liblinear', max_iter=100,
        #         multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):

    def svc():
        print("Creating Support Vector Machine(SVM) Model for classification")
        return SVC(kernel='rbf', verbose=True, max_iter=500, probability=True)

    # elf, C=1.0, kernel='rbf', degree=3, gamma='auto',
    #                  coef0=0.0, shrinking=True, probability=False,
    #                  tol=1e-3, cache_size=200, class_weight=None,
    #                  verbose=False, max_iter=-1, decision_function_shape=None,
    #                  random_state=None):

    def extra_trees_classifier():
        print("Creating Extra Decision Trees Classifier")
        return ExtraTreesClassifier(n_estimators=850,
                                     max_features=60,
                                     criterion='entropy',
                                     min_samples_split=4,
                                     max_depth=40,
                                     min_samples_leaf=2,
                                     n_jobs=-1)


    options = {
        'rf': random_forest,
        'log_reg': log_regression,
        'svc': svc,
        'extra_trees_classifier': extra_trees_classifier
    }

    return options[model_typ]()


def regression(standardize=False, model_type='rf', for_submission=True):
    submission = "_submission" if for_submission else ""
    standardized = "_standardized" if standardize else ""
    model_file_name = "08_reg_model_" + model_type + standardized + submission + ".p"

    print("Creating model....\n")
    xtr, xts, ytr, yts = get_datasets(for_submission)
    if standardize:
        print("Perform standardization\n")
        with open("../../dataset/06_standardizer.p", 'rb') as h:
            scaler = pickle.load(h)

        # print(xtr.shape)

        xtr = scaler.transform(xtr)
        xts = scaler.transform(xts)

    model = get_model(model_type)

    print("Start fitting our model. This may take a while...\n")
    model.fit(xtr, ytr)

    print("Saving model...\n")

    with open("../../dataset/" + model_file_name, 'wb') as h:
        pickle.dump(model, h)


if __name__ == "__main__":
    # Model options:
    # "log_reg",
    # "rf"(Random Forest),
    # "svc"(Support Vector Classification)
    # "extra_trees_classifier" (Extra Decision Trees Classifier)

    model_option = "extra_trees_classifier"

    regression(standardize=True, model_type=model_option, for_submission=False)
