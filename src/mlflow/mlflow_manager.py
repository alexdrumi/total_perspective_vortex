
import mlflow
import mlflow.sklearn
import joblib
import subprocess
import time
import numpy as np
from mlflow.models.signature import infer_signature
from sklearn.base import BaseEstimator

class MlflowManager():
	"""
	A class to manage MLflow operations, including starting the MLflow server 
	and logging experiment details, parameters, and models.
	"""
	def __init__(self):
		pass


	def start_mlflow_server(self):
		"""
		Starts the MLflow tracking server locally and sets the tracking URI.

		This method starts the MLflow UI on the default port (http://localhost:5000) 
		using a subprocess and configures MLflow to use this URI as the tracking server.

		It prints a message indicating that MLflow is running and the URL where it can be accessed.
		"""
		subprocess.Popen(["mlflow", "ui"])
		mlflow.set_tracking_uri("http://localhost:5000")  #uri
		print('mlfow is running on http://localhost:5000", here you can follow the model metrics.')
		time.sleep(2)



	def log_mlflow_experiment(self, group_key:str, best_params: dict, best_score: float, best_pipeline: BaseEstimator, X_train: np.ndarray, y_train: np.ndarray):
		"""
		Logs an MLflow experiment, including parameters, metrics, and the trained pipeline.

		Args:
			group_key (str): A unique identifier for the experiment group.
			best_params (dict): The best hyperparameters found during model tuning.
			best_score (float): The best cross-validation accuracy achieved.
			best_pipeline (Pipeline/BaseEstimator): The trained scikit-learn pipeline. 
			X_train (np.ndarray): The training input data used to infer the model's signature.
			y_train (np.ndarray): The training target data used to infer the model's signature.

		Logs:
			The group key as a parameter.
			The best hyperparameters.
			The best cross-validation accuracy.

		Saves:
			The trained pipeline to a file in the `../../models/` directory.

		Registers:
			The trained pipeline as a model in the MLflow tracking server.

		Prints:
			The best parameters and the best cross-validation accuracy.

		Notes:
			The model is logged with a signature inferred from the training data.
			The pipeline is registered with MLflow under a name derived from the group key.
		"""
		mlflow.set_experiment(f"{group_key}")
		
		#log info
		mlflow.log_param('group_key', group_key)
		mlflow.log_params(best_params)
		mlflow.log_metric('best_cross_val_accuracy', best_score)

		#print results
		print("Best Parameters:")
		print(best_params)
		print(f"Best Cross-Validation Accuracy: {best_score:.2f}")

		#save model
		signature = infer_signature(X_train, y_train)
		#best_pipeline = grid_search.best_estimator_
		model_filename = f"../../models/pipe_{group_key}.joblib"
		joblib.dump(best_pipeline, model_filename)

		#log model to mlflow
		mlflow.sklearn.log_model(
			sk_model=best_pipeline, 
			artifact_path='models', 
			signature=signature, 
			registered_model_name=f"model_{group_key}"
		)




