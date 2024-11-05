#!/usr/bin/python
import numpy as np
import mne
import sys
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score, KFold

from dataset_preprocessor import Preprocessor
from feature_extractor import FeatureExtractor


import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# from epoch_extractor import epoch_extractooor, extract_epochs
# from feature_extractor import feature_extractor, create_feature_vectors, calculate_mean_power_energy

from pca import My_PCA
from sklearn.preprocessing import FunctionTransformer
from epoch_extractor import EpochExtractor


from custom_scaler import CustomScaler
from reshaper import Reshaper

import logging
from pathlib import Path

#configure the logger
#set up logging to both file and terminal (console)
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

#file handler - Logs to a file
file_handler = logging.FileHandler('../logs/error_log.log', mode='w')
file_handler.setLevel(logging.ERROR)

#stream handler - Logs to terminal (console)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.ERROR)

#format for log messages
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

#add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)




mne.set_log_level(verbose='WARNING')
# channels = ["Fc3.", "Fcz.", "Fc4.", "C3..", "C1..", "Cz..", "C2..", "C4.."]
# channels = ["Fc1.","Fc2.", "Fc3.", "Fcz.", "Fc4.", "C3..", "C1..", "Cz..", "C2..", "C4.."]
            # "CP3",
            # "CP1",
            # "CPz",
            # "CP2",
            # "CP4",
            # "Fpz",
# predict = [
# "data/S018/S018R11.edf",
# "data/S042/S042R07.edf",
# "data/S042/S042R03.edf",
# "data/S104/S104R11.edf",
# # "data/S104/S104R07.edf",
# # "data/S090/S090R11.edf",
# # "data/S086/S086R11.edf",
# # "data/S086/S086R03.edf",
# # "data/S086/S086R07.edf",
# # "data/S017/S017R11.edf",
# # "data/S017/S017R07.edf",
# # "data/S017/S017R03.edf",
# # "data/S013/S013R07.edf",
# # "data/S013/S013R11.edf",
# # "data/S013/S013R03.edf",
# # "data/S055/S055R11.edf",
# # "data/S055/S055R07.edf",
# # "data/S055/S055R03.edf",
# # "data/S016/S016R03.edf",
# # "data/S016/S016R07.edf",
# # "data/S016/S016R11.edf",
# #"/data/S103/S103R11.edf",
# ]

train = [
		"../data/S018/S018R03.edf",
		"../data/S018/S018R07.edf",
		"../data/S018/S018R11.edf",

		# "../data/S028/S028R03.edf",
		# "../data/S028/S028R07.edf",
		# "../data/S028/S028R11.edf",

		# "../data/S038/S038R03.edf",
		# "../data/S038/S038R07.edf",
		# "../data/S038/S038R11.edf",


		# "../data/S048/S048R03.edf",
		# "../data/S048/S048R07.edf",
		# "../data/S048/S048R11.edf",


		# "../data/S058/S058R03.edf",
		# "../data/S058/S058R07.edf",
		# "../data/S058/S058R11.edf",


		# "../data/S068/S068R03.edf",
		# "../data/S068/S068R07.edf",
		# "../data/S068/S068R11.edf",


		# "../data/S078/S078R03.edf",
		# "../data/S078/S078R07.edf",
		# "../data/S078/S078R11.edf",


		# "../data/S088/S088R03.edf",
		# "../data/S088/S088R07.edf",
		# "../data/S088/S088R11.edf",


		# "../data/S098/S098R03.edf",
		# "../data/S098/S098R07.edf",
		# "../data/S098/S098R11.edf",


		# "../data/S018/S018R03.edf",
		# "../data/S018/S018R07.edf",
		# "../data/S018/S018R11.edf",


		# "../data/S018/S018R03.edf",
		# "../data/S018/S018R07.edf",
		# "../data/S018/S018R11.edf",



		# "../data/S018/S018R04.edf",
		# "../data/S018/S018R08.edf",
		# "../data/S018/S018R12.edf",

		# "../data/S018/S018R05.edf",
		# "../data/S018/S018R09.edf",
		# "../data/S018/S018R13.edf",

		# "../data/S018/S018R06.edf",
		# "../data/S018/S018R10.edf",
		# "../data/S018/S018R14.edf",

		"../data/S028/S028R03.edf",
		"../data/S028/S028R07.edf",
		"../data/S028/S028R11.edf",

		"../data/S028/S028R04.edf",
		"../data/S028/S028R08.edf",
		"../data/S028/S028R12.edf",

		"../data/S028/S028R05.edf",
		"../data/S028/S028R09.edf",
		"../data/S028/S028R13.edf",

		"../data/S028/S028R06.edf",
		"../data/S028/S028R10.edf",
		"../data/S028/S028R14.edf",



		"../data/S038/S038R03.edf",
		# "../data/S038/S038R07.edf",
		# "../data/S038/S038R11.edf",
		# "../data/S038/S038R04.edf",
		# "../data/S038/S038R08.edf",
		# "../data/S038/S038R12.edf",
		# "../data/S038/S038R05.edf",
		# "../data/S038/S038R09.edf",
		# "../data/S038/S038R13.edf",
		# "../data/S038/S038R06.edf",
		# "../data/S038/S038R10.edf",
		# "../data/S038/S038R14.edf",

		# "../data/S048/S048R03.edf",
		# "../data/S048/S048R07.edf",
		# "../data/S048/S048R11.edf",
		# "../data/S048/S048R04.edf",
		# "../data/S048/S048R08.edf",
		# "../data/S048/S048R12.edf",
		# "../data/S048/S048R05.edf",
		# "../data/S048/S048R09.edf",
		# "../data/S048/S048R13.edf",
		# "../data/S048/S048R06.edf",
		# "../data/S048/S048R10.edf",
		# "../data/S048/S048R14.edf",

		# "../data/S104/S104R03.edf",
		# "../data/S091/S091R11.edf",
		# "../data/S091/S091R03.edf",
		# "../data/S091/S091R07.edf",
		# "../data/S082/S082R11.edf",
		# "../data/S082/S082R03.edf",
		# "../data/S082/S082R07.edf",
		# "../data/S048/S048R03.edf",
		# "../data/S048/S048R11.edf",
		# "../data/S048/S048R07.edf",
		# "../data/S038/S038R11.edf",
		# "../data/S038/S038R07.edf",
		# "../data/S038/S038R03.edf",
		# "../data/S040/S040R03.edf",
		# "../data/S040/S040R07.edf",
		# "../data/S040/S040R11.edf",
		# "../data/S093/S093R07.edf",
		# "../data/S093/S093R11.edf",
		# "../data/S093/S093R03.edf",
		# "../data/S047/S047R11.edf",
		# "../data/S047/S047R07.edf",
		# "../data/S047/S047R03.edf",
		# "../data/S102/S102R07.edf",
		# "../data/S102/S102R03.edf",
		# "../data/S102/S102R11.edf",
		# "../data/S083/S083R11.edf",
		# "../data/S083/S083R03.edf",
		# "../data/S083/S083R07.edf",
		# "../data/S034/S034R07.edf",
		# "../data/S034/S034R03.edf",
		# "../data/S034/S034R11.edf",
		# "../data/S041/S041R07.edf",
		# "../data/S041/S041R03.edf",
		# "../data/S041/S041R11.edf",
		# "../data/S035/S035R07.edf",
		# "../data/S035/S035R11.edf",
		# "../data/S035/S035R03.edf",
		# "../data/S060/S060R07.edf",
		# "../data/S060/S060R11.edf",
		# "../data/S060/S060R03.edf",
		# "../data/S009/S009R11.edf",
		# "../data/S009/S009R07.edf",
		# "../data/S009/S009R03.edf",
		# "../data/S045/S045R11.edf",
		# "../data/S045/S045R07.edf",
		# "../data/S045/S045R03.edf",
		# "../data/S044/S044R03.edf",
		# "../data/S044/S044R11.edf",
		# "../data/S044/S044R07.edf",
		# "../data/S029/S029R11.edf",
		# "../data/S029/S029R03.edf",
		# "../data/S029/S029R07.edf",
		# "../data/S056/S056R03.edf",
		# "../data/S056/S056R11.edf",
		# "../data/S056/S056R07.edf",
		# "../data/S076/S076R07.edf",
		# "../data/S076/S076R03.edf",
		# "../data/S076/S076R11.edf",
		# "../data/S105/S105R07.edf",
		# "../data/S105/S105R11.edf",
		# "../data/S105/S105R03.edf",
		# "../data/S106/S106R07.edf",
		# "../data/S106/S106R03.edf",
		# "../data/S106/S106R11.edf",
		# "../data/S050/S050R07.edf",
		# "../data/S050/S050R03.edf",
		# "../data/S050/S050R11.edf",
		# "../data/S099/S099R07.edf",
		# "../data/S099/S099R03.edf",
		# "../data/S099/S099R11.edf",
		# "../data/S031/S031R03.edf",
		# "../data/S031/S031R11.edf",
		# "../data/S031/S031R07.edf",
		# "../data/S061/S061R03.edf",
		# "../data/S061/S061R07.edf",
		# "../data/S061/S061R11.edf",
		# "../data/S059/S059R07.edf",
		# "../data/S059/S059R11.edf",
		# "../data/S059/S059R03.edf",
		# "../data/S072/S072R07.edf",
		# "../data/S072/S072R03.edf",
		# "../data/S072/S072R11.edf",
		# "../data/S023/S023R03.edf",
		# "../data/S023/S023R11.edf",
		# "../data/S023/S023R07.edf",
		# "../data/S043/S043R11.edf",
		# "../data/S043/S043R07.edf",
		# "../data/S043/S043R03.edf",
		# "../data/S073/S073R07.edf",
		# "../data/S073/S073R11.edf",
		# "../data/S073/S073R03.edf",
		# "../data/S046/S046R11.edf",
		# "../data/S046/S046R07.edf",
		# "../data/S046/S046R03.edf",
		# "../data/S075/S075R07.edf",
		# "../data/S075/S075R11.edf",
		# "../data/S075/S075R03.edf",
		# "../data/S011/S011R03.edf",
		# "../data/S011/S011R07.edf",
		# "../data/S011/S011R11.edf",
		# "../data/S066/S066R03.edf",
		# "../data/S066/S066R07.edf",
		# "../data/S066/S066R11.edf",
		# "../data/S006/S006R11.edf",
		# "../data/S006/S006R03.edf",
		# "../data/S006/S006R07.edf",
		# "../data/S021/S021R11.edf",
		# "../data/S021/S021R03.edf",
		# "../data/S021/S021R07.edf",
		# "../data/S010/S010R03.edf",
		# "../data/S010/S010R07.edf",
		# "../data/S010/S010R11.edf",
		# "../data/S008/S008R07.edf",
		# "../data/S008/S008R03.edf",
		# "../data/S008/S008R11.edf",
		# "../data/S089/S089R03.edf",
		# "../data/S089/S089R07.edf",
		# "../data/S089/S089R11.edf",
		# "../data/S058/S058R07.edf",
		# "../data/S058/S058R11.edf",
		# "../data/S058/S058R03.edf",
		# "../data/S090/S090R03.edf",
		# "../data/S090/S090R07.edf",
]

# ica = mne.preprocessing.ICA(method="infomax")
#--------------------------------------------------------------------------------------------------------------------------
def main():
	try:
		dataset_preprocessor_instance = Preprocessor()
		loaded_raw_data = dataset_preprocessor_instance.load_raw_data(data_path=train) #RETURN DOESNT WORK, IT RETURNS AFTER 1 FILE
		# print(dataset_preprocessor_instance.raw_data)
		filtered_data = dataset_preprocessor_instance.filter_raw_data(loaded_raw_data) #this returns a triplet now

		# for data in filtered_data:
		# 	print(data[0], data[1], data[2])
		# sys.exit(1)
		# print(experiment)
		epoch_extractor_instance = EpochExtractor()
		epochs, labels = epoch_extractor_instance.extract_epochs_and_labels(filtered_data)
		# print(labels)
		# sys.exit(1)

		# for idx, epoch in enumerate(epochs):
		# 	print(epoch)
		# 	print(labels[idx])

		feature_extractor_instance = FeatureExtractor()
		trained_extracted_features = feature_extractor_instance.extract_features(epochs) #callable


		#https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.FunctionTransformer.html
		custom_scaler = CustomScaler()
		reshaper = Reshaper()
		my_pca = My_PCA(n_comps=100)
		mlp_classifier = MLPClassifier(hidden_layer_sizes=(20,10),
									max_iter=16000,
									random_state=42
		)

	#for customscaler 3d shape check
	


		pipeline = Pipeline([
			('scaler', custom_scaler),
			('reshaper', reshaper),
			('pca', my_pca),
			('classifier', mlp_classifier) #mlp will be replaced in grid search
		])

	# pipeline.fit(trained_extracted_features, labels)


	# #------------------------------------------------------------------------------------------------------------

	# predict_raw = dataset_preprocessor_instance.load_raw_data(data_path=predict)
	# predict_filtered = dataset_preprocessor_instance.filter_raw_data()
	# epochs_predict, labels_predict = epoch_extractor_instance.extract_epochs_and_labels(predict_filtered)

	# test_extracted_features = feature_extractor_instance.extract_features(epochs_predict) #callable


		shuffle_split_validation = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

		# # scoring = ['accuracy', 'precision', 'f1_micro'] this only works for: scores = cross_validate(pipeline_custom, x_train, y_train, scoring=scoring, cv=k_fold_cross_val)
		# # scores = cross_val_score(pipeline_custom, x_train, y_train, scoring='accuracy', cv=shuffle_split_validation)
		scores = cross_val_score(
			pipeline, trained_extracted_features, 
			labels,  
			scoring='accuracy', 
			cv=shuffle_split_validation
		)
		
		print(scores)
		print(f'Average accuracy: {scores.mean()}')



		grid_search_params = [
			#MLP
			{
				'classifier': [MLPClassifier(
					max_iter=16000,
					early_stopping=True,
					n_iter_no_change=100, #if it doesnt improve for 10 epochs
					verbose=True)],
				'pca__n_comps': [20,30,42,50],
				#hidden layers of multilayer perceptron class
				'classifier__hidden_layer_sizes': [(20, 10), (50, 20), (100, 50)],
				#relu->helps mitigate vanishing gradients, faster convergence
				#tanh->hyperbolic tangent, outputs centered around zero
				'classifier__activation': ['relu', 'tanh'],
				#adam, efficient for large datasets, adapts learning rates
				#stochastic gradient, generalize better, slower convergence
				'classifier__solver': ['adam', 'sgd'],
				'classifier__learning_rate_init': [0.001, 0.01, 0.1]

			},
			#SVC
			{
				'classifier': [SVC()],
				'pca__n_comps': [20, 30, 42, 50],
				'classifier__C': [0.1, 1, 10],
				'classifier__kernel': ['linear', 'rbf']
			},
			
			#RANDOM FOREST
			{
				'classifier': [RandomForestClassifier()],
				'pca__n_comps': [20,30,42,50],
				'classifier__n_estimators': [50, 100, 200],
				'classifier__max_depth': [None, 10, 20]
			},
			#DECISION TREE
			{
				'pca__n_comps': [20, 30, 42, 50],
				'classifier': [DecisionTreeClassifier()],
				'classifier__max_depth': [None, 10, 20],
				'classifier__min_samples_split': [2, 5, 10]
			},
			# Logistic Regression
			{
				'classifier': [LogisticRegression()],
				'pca__n_comps': [20, 30, 42, 50],
				'classifier__C': [0.1, 1, 10],
				'classifier__penalty': ['l1', 'l2'],
				'classifier__solver': ['liblinear'],  # 'liblinear' supports 'l1' penalty
				'classifier__multi_class': ['auto'],
				'classifier__max_iter': [1000, 5000]
			}
		]

		from sklearn.model_selection import GridSearchCV

		grid_search = GridSearchCV(
			estimator=pipeline,
			param_grid=grid_search_params,
			cv=9,  #9fold cross-val
			scoring='accuracy',  #evalmetric
			n_jobs=-1,  #util all all available cpu cores
			verbose=2,  # For detailed output
			refit=True #this fits it automatically to the best estimator, just to emphasize here, its True by default
		)

		#just to use standard variables
		X_train = trained_extracted_features
		y_train = labels 
		grid_search.fit(X_train, y_train)

		print("Best Parameters:")
		print(grid_search.best_params_)
		print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.2f}")


		best_pipeline = grid_search.best_estimator_
		joblib.dump(best_pipeline, '../models/pipe.joblib')


	except FileNotFoundError as e:
		logging.error(f"File not found: {e}")
	except PermissionError as e:
		logging.error(f"Permission on the file denied: {e}")
	except IOError as e:
		logging.error(f"Error reading the data file: {e}")
	except ValueError as e:
		logging.error(f"Invalid EDF data: {e}")
	except TypeError as e:
			logging.error(f"{e}")

if __name__ == '__main__':
	main()