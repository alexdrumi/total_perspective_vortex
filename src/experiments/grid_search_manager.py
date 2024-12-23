from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit, cross_val_score, KFold, GridSearchCV

import yaml
import numpy as np

class GridSearchManager():
	def __init__(self):
		self.classifier_mapping = {
				'MLPClassifier': MLPClassifier(max_iter=10000,early_stopping=True,n_iter_no_change=50,verbose=False),
				'SVC': SVC(),
				'RandomForestClassifier': RandomForestClassifier(),
				'LogisticRegression': LogisticRegression(),
				'DecisionTreeClassifier': DecisionTreeClassifier()
			}



	def load_config(self):
		with open('../../configs/grid_search_parameters.yaml', 'r') as f:
			config = yaml.safe_load(f)
		return config



	def create_grid_search_parameters(self):
		config = self.load_config()
		
		grid_search_params = []
		for param_set in config['grid_search_params']:
			classifier_name = param_set['classifier']
			if classifier_name in self.classifier_mapping:
				param_set['classifier'] = [self.classifier_mapping[classifier_name]] 
				print(f'{param_set} is gonna be the paramset now')
				grid_search_params.append(param_set)

		return grid_search_params



	def create_grid_search(self, pipeline):
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
	


	def get_grid_search_results(self, grid_search):
		best_params = {k: (float(v) if isinstance(v, (np.float64, np.float32)) else v) for k, v in grid_search.best_params_.items()}
		best_score = float(grid_search.best_score_)  # Ensure it's a Python float
		best_pipeline = grid_search.best_estimator_

		return best_params, best_score, best_pipeline


	def run_grid_search(self, pipeline, X_train, y_train):
		grid_search = self.create_grid_search(pipeline)
		grid_search.fit(X_train, y_train)

		best_params, best_score, best_pipeline = self.get_grid_search_results(grid_search)
		
		return best_params, best_score, best_pipeline