import pickle
import operator


def explore_model(standardize, model_type='rf', for_submission=True):
	submission = "_submission" if for_submission else ""
	standardized = "_standardized" if standardize else ""
	model_file_name = "08_reg_model_" + model_type + standardized + submission + ".p"

	print("Opening " + model_file_name + "...")

	with open("../../dataset/" + model_file_name, 'rb') as h:
		model = pickle.load(h)

	if model_type == 'rf':
		feature_importances = model.feature_importances_
		print(feature_importances)

	with open("../../dataset/04_tr_filled_data.p", 'rb') as h:
		xtr = pickle.load(h)

	del (xtr['ID'])
	del (xtr['target'])
	feature_names = list(xtr.columns.values)

	feat_imps = dict(zip(feature_names, feature_importances))
	feat_imps = sorted(feat_imps.items(), key=operator.itemgetter(1), reverse=True)

	print(feat_imps)

if __name__ == "__main__":
	# Model options:
	# "log_reg",
	# "rf"(Random Forest),
	# "svc"(Support Vector Classification)
	# "extra_trees_classifier" (Extra Decision Trees Classifier)

	model_option = "rf"

	explore_model(standardize=True, model_type=model_option, for_submission=False)
