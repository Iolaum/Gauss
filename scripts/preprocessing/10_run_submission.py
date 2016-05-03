import pickle
import os.path
import numpy as np
import pandas as pd


def replace_probs(pval):
    return max(min(pval, (1 - 1e-14)), 1e-14)


def submission(standardize, model_type='rf', for_submission=True):
    submission = "_submission" if for_submission else ""
    standardized = "_standardized" if standardize else ""
    model_file_name = "08_reg_model_" + model_type + standardized + submission + ".p"

    print("Loading Test Set..\n")
    with open("../../dataset/04_ts_filled_data.p", 'rb') as h:
        test_set_df = pickle.load(h)

        # Keep ID col for CSV
        id_col = np.array(test_set_df['ID'])

        # Remove ID col
        del (test_set_df['ID'])
        test_set = np.array(test_set_df)

        # # Debug
        print(id_col)
        print(test_set.shape)
        # print(test_set)
        # print(test_set[0, :])
        # print(type(test_set))

        # x = np.delete(x, 0, axis=0)

    if standardize:
        print("Loading Scaler\n")
        with open("../../dataset/06_standardizer.p", 'rb') as h:
            scaler = pickle.load(h)
        print("Perform standardization\n")
        print(test_set.shape)
        test_set = scaler.transform(test_set)

    # # Debug
    # print(xtr_pred)
    # print(xtr_pred.shape)

    print("Log Loss fn results")
    if os.path.isfile("../../dataset/" + model_file_name):
        print("Found trained model, loading...")
        with open("../../dataset/" + model_file_name, 'rb') as h:
            model = pickle.load(h)

        test_set_pred = model.predict_proba(test_set)
        if model_type == 'rf':
            print("Predict using Random Forest Classifier")
        elif model_type == 'svc':
            print("Predict using SVM Classifier")
        elif model_type == 'log_reg':
            print("Predict using Logistic Regressor")

        # # Debug
        # print(test_set_pred.shape)
        # print(test_set_pred)
        # print(test_set_pred[:, [1]])

        new_arr = np.column_stack((id_col.astype(int), test_set_pred[:, [1]]))
        print("Creating pandas dataframe with ID and Predicted Probability")
        subm_dataframe = pd.DataFrame(new_arr,
                                      columns=[
                                          'ID',
                                          'PredictedProb'
                                      ],
                                      index=id_col.tolist()
                                      )

        subm_dataframe.ID = subm_dataframe.ID.astype(int)
        # # Debug
        # print(subm_dataframe.shape)
        # print(subm_dataframe)

        subm = "_full_tr_set" if for_submission else ""
        subm_fn = "subm_file_" + model_type + standardized + subm + ".csv"
        print("Saving into " + subm_fn)
        subm_dataframe.to_csv("../../dataset/" + subm_fn, index=False)
    else:
        print("Did not found trained model...")


if __name__ == "__main__":
    # Model options:
    # "log_reg",
    # "rf"(Random Forest),
    # "svc"(Support Vector Classification)
    # "extra_trees_classifier" (Extra Decision Trees Classifier)

    model_option = "extra_trees_classifier"

    submission(standardize=True, model_type=model_option, for_submission=False)
