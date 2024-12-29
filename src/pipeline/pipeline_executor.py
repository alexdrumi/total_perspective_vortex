import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, KFold

class PipelineExecutor():
	def __init__(self):
		pass



	def save_model(self, best_pipeline: BaseEstimator, group_key:str ) -> None:
		"""
		Saves the best pipeline to the location: ../../models/pipe_{group_key}.joblib .

		Args:
			best_pipeline (BaseEstimator Pipeline object): the best pipeline passed after running experiments.
			group_key (string): name of the experiment group.
		"""
		model_filename = f"../../models/pipe_{group_key}.joblib"
		joblib.dump(best_pipeline, model_filename)



	def evaluate_pipeline(self, group_key:str, best_pipeline: BaseEstimator, X_train: np.ndarray, y_train: np.ndarray) -> None:
		"""
		Evaluates the pipeline using Kfold cross-validation. Prints the accuracy scores.

		Args:
			group_key (string): name of the experiment group.
			best_pipeline (BaseEstimator Pipeline object): the best pipeline passed after running experiments.
			X_train (np.ndarray): Training features.
			y_train (np.ndarray): Training labels.
		"""
		kfold = KFold(n_splits=5, shuffle=True, random_state=0)
		scores = cross_val_score(
			best_pipeline, X_train, 
			y_train, 
			scoring='accuracy', 
			cv=kfold
		)

		print(scores)
		print(f"\033[92mAverage accuracy with cross-validation for group: {group_key}: {scores.mean():.2f}\033[0m")
