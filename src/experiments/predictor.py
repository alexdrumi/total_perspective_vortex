import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.pipeline_builder import PipelineBuilder
from src.experiments.grid_search import GridSearchManager
from src.pipeline.pipeline_executor import PipelineExecutor
from src.utils.command_line_parser import CommandLineParser
from src.pipeline.feature_extractor import FeatureExtractor
from src.mlflow.mlflow_manager import MlflowManager
from src.data_processing.preprocessor import Preprocessor
from src.data_processing.extract_epochs import EpochExtractor
from src.utils.eeg_plotter import EEGPlotter
from src.data_processing.concatenate_epochs import EpochConcatenator
from sklearn.base import BaseEstimator
from typing import Optional
from src.utils.data_and_model_checker import check_data, check_models

import numpy as np
import matplotlib.pyplot as plt
import joblib
import logging
import time
import os

from sklearn.model_selection import  cross_val_score, KFold



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


#main orchestrator for predictions
class PredictOrchestrator:
	"""
	The main orchestrator: parses CLI args, runs the entire experiment.
	"""
	def __init__(self):
		"""
		Initializes the PredictOrchestrator with command-line argument configuration.

		Returns:
			None
		"""
		argument_config = [
			{
				'name': '--plot_eeg_predictions',
				'type': str,
				'default': 'false',
				'choices': ['true', 'false'],
				'help': 'Enable (True) or disable (False) the visual representation of EEG predictions.'
			}
		]
		self.arg_parser = CommandLineParser(argument_config)


	def run(self, predict):
		"""
		Runs the prediction pipeline: checks data and models, parses arguments, loads and preprocesses data,
		and evaluates experiment predictions.

		Args:
			predict (str): A string parameter used for prediction (e.g., a path to data).

		Returns:
			None
		"""
		check_data()
		check_models()

		plot_eeg_predictions_enabled = self.arg_parser.parse_arguments()
		plot_eeg = (plot_eeg_predictions_enabled == True)

		predictor = ExperimentPredictor(plot_eeg=plot_eeg)

		#load and preprocess data
		epochs_dict, labels_dict, run_groups = predictor.load_and_filter_data(predict)

		total_mean_accuracy_events = []
		for group in run_groups:
			groups_runs = group['runs']
			group_key = f"runs_{'_'.join(map(str, groups_runs))}"
			model_path = f"../../models/pipe_{group_key}.joblib"

			run_keys = [k for k in epochs_dict.keys() if int(k[-2:]) in groups_runs]
			if not run_keys:
				continue
			
			pipeline = predictor.load_model(model_path)
			if pipeline is None:
				continue
			
			run_key = run_keys[0]
			accuracy, crossval_mean = predictor.evaluate_experiment(
				epochs_dict, labels_dict, pipeline, group, run_key, chunk_size=7
			)
			print(f"\033[1;32mAccuracy of current experiment {run_key} is: {accuracy}\nCross-validation with Kfold mean is: {crossval_mean}\n\033[0m")
			time.sleep(2)

			#if events are not baseline->with baseline the rating will be heavily scewed, doesnt make sense.
			#if group['runs'][0] not in [1, 2]: #this would be with all 6 experiments, but just a better outcome in any case.
			total_mean_accuracy_events.append(accuracy)

		if total_mean_accuracy_events:
			print("\033[1;32mMean accuracy of the six different experiments for all 109 subjects:\033[0m")
			for i, accuracy in enumerate(total_mean_accuracy_events):
				print(f"\033[1;32mExperiment {i}: accuracy = {accuracy:.3f}\033[0m")

			print(f"\033[1;32mMean accuracy of 6 event-based experiments: {np.mean(total_mean_accuracy_events):.3f}\033[0m")
			print(f"\033[1;32mMean accuracy of 4 event-based experiments without open/closed eyes baseline: {np.mean(total_mean_accuracy_events[2:]):.3f}\033[0m")

			time.sleep(1)
		else:
			print("No event-based experiments processed.")



class ExperimentPredictor:
	"""
	Handles loading the model, preprocessing data, making predictions, and calculating statistics.
	"""
	def __init__(self, plot_eeg: bool=False) -> None:
		"""
		Initializes the ExperimentPredictor with optional EEG plotting and required components.

		Args:
			plot_eeg (bool): Whether to enable EEG plotting. Default is False.

		Returns:
			None
		"""
		self.plot_eeg = plot_eeg
		self.data_preprocessor = Preprocessor()
		self.extract_epochs = EpochExtractor()
		self.feature_extractor = FeatureExtractor()
		self.eeg_plotter = EEGPlotter(epoch_duration=7.1)



	def load_and_filter_data(self, predict: str) -> tuple[dict, dict, list]:
		"""
		Loads raw data, applies filtering, and extracts epochs and labels.

		Args:
			predict (str): Path or parameter for raw EEG data.

		Returns:
			Tuple:
				epochs_dict (dict): Dictionary of extracted epochs.
				labels_dict (dict): Dictionary of associated labels.
				run_groups (list): List of experimental run groups.
		"""
		logging.info("Loading raw data...")
		loaded_raw_data = self.data_preprocessor.load_raw_data(data_path=predict)
		logging.info("Filtering raw data...")
		filtered_data = self.data_preprocessor.filter_raw_data(loaded_raw_data)
		logging.info("Extracting epochs and labels...")
		epochs_dict, labels_dict = self.extract_epochs.extract_epochs_and_labels(filtered_data)
		run_groups = self.extract_epochs.experiments_list
		return epochs_dict, labels_dict, run_groups



	def load_model(self, model_path: str) -> Optional[BaseEstimator]:
		"""
		Loads a trained model pipeline from the specified path.

		Args:
			model_path (str): Path to the saved model file.

		Returns:
			Optional[BaseEstimator]: The loaded machine learning pipeline, or None if not found.
		"""
		try:
			pipeline = joblib.load(model_path)
			logging.info(f"Pipeline loaded successfully from {model_path}")
			return pipeline
		except FileNotFoundError:
			logging.warning(f'Pipeline file not found at {model_path}.')
			return None



	def predict_chunk(self, pipeline: BaseEstimator, test_features: np.ndarray) -> tuple[np.ndarray, float]:
		"""
		Makes predictions on a chunk of features using the trained pipeline.

		Args:
			pipeline (BaseEstimator): The trained model pipeline.
			test_features (np.ndarray): Features for the current chunk.

		Returns:
			Tuple:
				predictions (np.ndarray): Predicted labels for the chunk.
				inference_time (float): Time taken for inference.
		"""
		start_time = time.time()
		predictions = pipeline.predict(test_features)
		end_time = time.time()
		inference_time = end_time - start_time
		logging.debug(f"Prediction time for current chunk: {inference_time:.4f} seconds")
		# print(f'Prediction time for current chunk is: {inference_time:.4f} seconds')
		return predictions, inference_time



	def evaluate_experiment(self, epochs_predict:dict, labels_predict:dict, pipeline: BaseEstimator, group:dict, run_key:str, chunk_size: int=7) -> tuple[float, float]:
		"""
		Evaluates the model's performance on an entire experimental run group.

		Args:
			epochs_predict (dict): Dictionary of epochs for prediction.
			labels_predict (dict): Dictionary of labels for prediction.
			pipeline (BaseEstimator): The trained machine learning pipeline.
			group (dict): Experimental group information.
			run_key (str): Identifier for the current run.
			chunk_size (int): Number of samples per chunk. Default is 7.

		Returns:
			Tuple:
				overall_accuracy (float): Overall accuracy for the run.
				mean_cross_val_accuracy (float): Mean accuracy from cross-validation.
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

		#cross-validate
		kfold = KFold(n_splits=5, shuffle=True, random_state=0)
		scores = cross_val_score(
			pipeline, 
			test_extracted_features, 
			flattened_labels,
			scoring='accuracy', 
			cv=kfold
		)
		# logging.info(f"Cross-val scores: {scores}")
		# logging.info(f"Mean cross-val accuracy: {scores.mean():.2f}")

		return overall_accuracy, scores.mean()
