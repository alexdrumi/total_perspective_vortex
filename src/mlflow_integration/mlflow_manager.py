
import mlflow
import mlflow.sklearn
import joblib
import subprocess
import time
from mlflow.models.signature import infer_signature


class MlflowManager():
	# mlflow_enabled = self.command_line_parser.arg_parser.parse_arguments()
	# if (mlflow_enabled == True):
	# 	print(f'MLFLOW enabled: go to localhost:5000 to see model metrics.') #green color?
	# 	self.start_mlflow_server()
	def __init__(self):
		pass


	def start_mlflow_server(self):
		subprocess.Popen(["mlflow", "ui"])
		mlflow.set_tracking_uri("http://localhost:5000")  #uri
		print('mlfow is running on http://localhost:5000", here you can follow the model metrics.')
		time.sleep(2)



	def log_mlflow_experiment(self, group_key, best_params, best_score, best_pipeline, X_train, y_train):
		
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
		model_filename = f"../models/pipe_{group_key}.joblib"
		joblib.dump(best_pipeline, model_filename)

		#log model to mlflow
		mlflow.sklearn.log_model(
			sk_model=best_pipeline, 
			artifact_path='models', 
			signature=signature, 
			registered_model_name=f"model_{group_key}"
		)




