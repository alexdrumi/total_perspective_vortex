from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit, cross_val_score, KFold, GridSearchCV
from sklearn.base import BaseEstimator

import yaml
import numpy as np

class GridSearchManager():
	"""
	Manages the setup and execution of grid search for hyperparameter tuning.

	This class integrates with various classifiers, reads configurations from a YAML file,
	and uses GridSearchCV to perform hyperparameter optimization.

	Attributes:
		classifier_mapping (dict): A dictionary mapping classifier names to scikit-learn classifier objects.
	"""
	def __init__(self) -> None:
		"""
		Initializes the GridSearchManager with a predefined mapping of classifiers.
		"""
		self.classifier_mapping = {
				'MLPClassifier': MLPClassifier(max_iter=10000,early_stopping=True,n_iter_no_change=50,verbose=False),
				'SVC': SVC(),
				'RandomForestClassifier': RandomForestClassifier(),
				'LogisticRegression': LogisticRegression(),
				'DecisionTreeClassifier': DecisionTreeClassifier()
			}


	#this could be an external function as well, using it in dataset preprocessor
	def load_config(self) -> dict:
		"""
		Loads the grid search parameters from a YAML configuration file.

		Returns:
			dict: A dictionary containing the grid search parameters.
		"""
		with open('../../config/grid_search_parameters.yaml', 'r') as f:
			config = yaml.safe_load(f)
		return config



	def create_grid_search_parameters(self) -> list:
		"""
		Prepares the grid search parameters based on the config file and classifier mapping.

		Returns:
			list: A list of parameter grids for grid search.
		"""
		config = self.load_config()
		
		grid_search_params = []
		for param_set in config['grid_search_params']:
			classifier_name = param_set['classifier']
			if classifier_name in self.classifier_mapping:
				param_set['classifier'] = [self.classifier_mapping[classifier_name]] 
				print(f'{param_set} is gonna be the paramset now')
				grid_search_params.append(param_set)

		return grid_search_params



	def create_grid_search(self, pipeline: BaseEstimator) -> GridSearchCV:
		"""
		Creates a GridSearchCV object with the prepared parameter grid.

		Args:
			pipeline: The pipeline or model to be optimized.

		Returns:
			GridSearchCV: Configured GridSearchCV object.
		"""
		grid_search_params = self.create_grid_search_parameters()
		grid_search = GridSearchCV(
			estimator=pipeline,
				param_grid=grid_search_params,
				cv=9,  #9fold cross-val
				scoring='accuracy',  #evalmetric
				n_jobs=-1,  #util all all available cpu cores
				verbose=1,  #2 would be for detailed output
				refit=True #this fits it automatically to the best estimator, just to emphasize here, its True by default
		)
		return grid_search
	


	def get_grid_search_results(self, grid_search: GridSearchCV) -> tuple[dict, float, BaseEstimator]:
		"""
		Extracts the best parameters, score, and pipeline from a completed GridSearchCV.

		Args:
			grid_search (GridSearchCV): The completed GridSearchCV object.

		Returns:
			Tuple: A Tuple containing:
				best_params (dict): The best parameters from the grid search.
				best_score (float): The best accuracy score.
				best_pipeline (Pipeline): The pipeline corresponding to the best parameters.
		"""
		best_params = {k: (float(v) if isinstance(v, (np.float64, np.float32)) else v) for k, v in grid_search.best_params_.items()}
		best_score = float(grid_search.best_score_)  # Ensure it's a Python float
		best_pipeline = grid_search.best_estimator_

		return best_params, best_score, best_pipeline


	def run_grid_search(self, pipeline: BaseEstimator, X_train: np.ndarray, y_train: np.ndarray) -> tuple[dict, float, BaseEstimator]:
		"""
		Executes grid search on the given pipeline and training data.

		Args:
			pipeline: The pipeline or model to optimize.
			X_train (np.ndarray): Training feature data.
			y_train (np.ndarray): Training target data.

		Returns:
			Tuple: A Tuple containing:
				- best_params (dict): The best parameters from the grid search.
				- best_score (float): The best accuracy score.
				- best_pipeline (Pipeline): The pipeline corresponding to the best parameters.
		"""
		grid_search = self.create_grid_search(pipeline)
		grid_search.fit(X_train, y_train)

		best_params, best_score, best_pipeline = self.get_grid_search_results(grid_search)		
		return best_params, best_score, best_pipeline