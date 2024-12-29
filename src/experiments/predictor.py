from src.pipeline.pipeline_builder import PipelineBuilder
from src.experiments.grid_search import GridSearchManager
from src.pipeline.pipeline_executor import PipelineExecutor
from src.utils.command_line_parser import CommandLineParser
from src.pipeline.feature_extractor import FeatureExtractor
from src.mlflow_integration.mlflow_manager import MlflowManager
from src.data_processing.preprocessor import Preprocessor
from src.data_processing.extract_epochs import EpochExtractor
from src.utils.eeg_plotter import EEGPlotter
from src.data_processing.concatenate_epochs import EpochConcatenator

import numpy as np
import sys
import matplotlib.pyplot as plt
import joblib
import logging
import time

from sklearn.model_selection import  cross_val_score, KFold
from sklearn.model_selection import KFold



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




class ExperimentPredictor:
	"""
	Handles loading the model, preprocessing data, making predictions, 
	and calculating statistics (accuracy, cross-validation, etc.).
	"""
	def __init__(self, plot_eeg=False):
		self.plot_eeg = plot_eeg
		self.data_preprocessor = Preprocessor()
		self.extract_epochs = EpochExtractor()
		self.feature_extractor = FeatureExtractor()
		self.eeg_plotter = EEGPlotter(epoch_duration=7.1)



	def load_and_filter_data(self, predict):
		"""
		Loads raw data from a path, filters it, and extracts epochs/labels.
		"""
		logging.info("Loading raw data...")
		loaded_raw_data = self.data_preprocessor.load_raw_data(data_path=predict)
		logging.info("Filtering raw data...")
		filtered_data = self.data_preprocessor.filter_raw_data(loaded_raw_data)
		logging.info("Extracting epochs and labels...")
		epochs_dict, labels_dict = self.extract_epochs.extract_epochs_and_labels(filtered_data)
		run_groups = self.extract_epochs.experiments_list
		return epochs_dict, labels_dict, run_groups



	def load_model(self, model_path):
		"""
		Tries to load a model (pipeline) from a given path.
		"""
		try:
			pipeline = joblib.load(model_path)
			logging.info(f"Pipeline loaded successfully from {model_path}")
			return pipeline
		except FileNotFoundError:
			logging.warning(f'Pipeline file not found at {model_path}.')
			return None



	def predict_chunk(self, pipeline, test_features):
		"""
		Runs inference on a chunk of features using a trained pipeline.
		"""
		start_time = time.time()
		predictions = pipeline.predict(test_features)
		end_time = time.time()
		inference_time = end_time - start_time
		logging.debug(f"Prediction time for current chunk: {inference_time:.4f} seconds")
		print(f'Prediction time for current chunk is: {inference_time:.4f} seconds')
		return predictions, inference_time



	def evaluate_experiment(self, epochs_predict, labels_predict, pipeline, group, run_key, chunk_size=7):
		"""
		Evaluates predictions on an entire run group (split into chunks), 
		and optionally plots EEG signals if self.plot_eeg is True.
		"""
		feature_extraction_method = 'baseline' if (group['runs'][0] in [1, 2]) else 'events'
		
		#extract features
		test_extracted_features = self.feature_extractor.extract_features(
			epochs_predict[run_key], feature_extraction_method
		)
		flattened_epochs = epochs_predict[run_key]
		flattened_labels = labels_predict[run_key]

		total_chunks = len(flattened_epochs) // chunk_size
		total_correct = 0

		fig, ax = None, None
		if self.plot_eeg:
			fig, ax = plt.subplots(figsize=(15, 6))
			plt.ion()

		for chunk_idx in range(total_chunks):
			start = chunk_idx * chunk_size
			end = start + chunk_size
			current_features = test_extracted_features[start:end]
			current_labels = flattened_labels[start:end]
			current_epochs = flattened_epochs[start:end]

			current_pred, inference_time = self.predict_chunk(pipeline, current_features)
			correct_predictions = np.sum(current_pred == current_labels)
			total_correct += correct_predictions

			current_accuracy = correct_predictions / len(current_labels)
			logging.info(f"Chunk {chunk_idx}: Accuracy={current_accuracy:.2f}, Inference time={inference_time:.3f}s")

			# print(f'{chunk_idx} is chunk idx, {total_chunks} is total_chunks')
			if self.plot_eeg and total_chunks - 4 == chunk_idx: #print one before the last chunk
				label_names = group['mapping']
				self.eeg_plotter.plot_eeg_epochs_chunk(
					current_batch_idx=chunk_idx,
					epochs_chunk=current_epochs,
					labels_chunk=current_labels,
					predictions_chunk=current_pred,
					label_names=label_names,
					ax=ax,
					alpha=0.3,
					linewidth=0.7
				)
				time.sleep(1)

		overall_accuracy = total_correct / (total_chunks * chunk_size)
		logging.info(f"Overall accuracy for run_key {run_key}: {overall_accuracy:.3f}")

		# Cross-validate
		kfold = KFold(n_splits=5, shuffle=True, random_state=0)
		scores = cross_val_score(
			pipeline, 
			test_extracted_features, 
			flattened_labels,
			scoring='accuracy', 
			cv=kfold
		)
		logging.info(f"Cross-val scores: {scores}")
		logging.info(f"Mean cross-val accuracy: {scores.mean():.2f}")

		return overall_accuracy, scores.mean()
