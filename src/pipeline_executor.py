import joblib
from sklearn.model_selection import ShuffleSplit, cross_val_score, KFold, GridSearchCV


class PipelineExecutor():
	def __init__(self):
		pass

	def save_model(self, best_pipeline, group_key):
		model_filename = f"../models/pipe_{group_key}.joblib"
		joblib.dump(best_pipeline, model_filename)


	def evaluate_pipeline(self, group_key, best_pipeline, best_score, X_train, y_train):
		kfold = KFold(n_splits=5, shuffle=True, random_state=0)
		scores = cross_val_score(
			best_pipeline, X_train, 
			y_train, 
			scoring='accuracy', 
			cv=kfold
		)

		print(scores)
		print(f"\033[92mAverage accuracy with cross-validation for group: {group_key}: {scores.mean():.2f}\033[0m")
