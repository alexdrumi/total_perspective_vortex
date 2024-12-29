import numpy as np
import mne
import sys
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import ShuffleSplit, cross_val_score, KFold
from sklearn.model_selection import KFold


import joblib
import logging
import time
import yaml
import os

# Add the `src` directory to the system path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(src_path)

from myapp import PredictOrchestrator
from src.pipeline.custom_scaler import CustomScaler
from src.pipeline.reshaper import Reshaper
from src.utils.command_line_parser import CommandLineParser
from src.data_processing.preprocessor import Preprocessor
from src.pipeline.feature_extractor import FeatureExtractor
from src.pipeline.pca import My_PCA
from src.data_processing.extract_epochs import EpochExtractor

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

file_handler = logging.FileHandler('../../logs/error_log.log', mode='w')
file_handler.setLevel(logging.ERROR)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)



channels = ["Fc3.", "Fcz.", "Fc4.", "C3..", "C1..", "Cz..", "C2..", "C4.."]


#-------------------------------------------------------


def main():
	try:
		predict_data_path = '../../config/predict_data.yaml'
		app = PredictOrchestrator()
		app.run(predict_data_path)

	except FileNotFoundError as e:
		logging.error(f"File not found: {e}")
	except PermissionError as e:
		logging.error(f"Permission error: {e}")
	except IOError as e:
		logging.error(f"IO error: {e}")
	except ValueError as e:
		logging.error(f"Value error: {e}")

if __name__ == "__main__":
	main()
