#!/usr/bin/python
import numpy as np
import mne
import sys
import matplotlib.pyplot as plt
import joblib
import logging
from pathlib import Path
import time
import yaml
import os

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(src_path)

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from mne.decoding import CSP

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import ShuffleSplit, cross_val_score, KFold, GridSearchCV

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from src.mlflow.mlflow_manager import MlflowManager
from src.data_processing.preprocessor import Preprocessor
from src.pipeline.feature_extractor import FeatureExtractor
from src.pipeline.pca import My_PCA
from src.data_processing.extract_epochs import EpochExtractor
from src.pipeline.custom_scaler import CustomScaler
from src.pipeline.reshaper import Reshaper
from src.utils.command_line_parser import CommandLineParser
from src.experiments.trainer import ExperimentTrainerFacade

import subprocess
from pathlib import Path

#logger config
#logging for both file and console

log_file_path = Path(__file__).resolve().parent.parent.parent / 'logs' / 'error_log.log'
log_file_path.parent.mkdir(parents=True, exist_ok=True) #checks if there is this folder existing

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

file_handler = logging.FileHandler(log_file_path, mode='w')
file_handler.setLevel(logging.ERROR)

#stream handler - Logs to terminal (console)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.ERROR)

#formatfor log messages
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

#handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

mne.set_log_level(verbose='WARNING')

#eeg channels to read data from->in the paper
channels = ["Fc3.", "Fcz.", "Fc4.", "C3..", "C1..", "Cz..", "C2..", "C4.."]


#-------------------------------------------------------

def main():
	try:
		trainer = ExperimentTrainerFacade()
		import time
		start = time.time()
		trainer.run_experiment()
		end = time.time()

		print(f"The time the script took : {end - start} ms")

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
	try:
		main()
	except KeyboardInterrupt:
		logging.info("CTRL+C pressed. Cleaning up...")
		#what should I cleanup here? ports closed from mlflow no?
		sys.exit(0)