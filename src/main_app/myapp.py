from src.pipeline.pipeline_builder import PipelineBuilder
from src.experiments.grid_search import GridSearchManager
from src.pipeline.pipeline_executor import PipelineExecutor
from src.utils.command_line_parser import CommandLineParser
from src.pipeline.feature_extractor import FeatureExtractor
from src.mlflow_integration.mlflow_manager import MlflowManager
from src.data_processing.preprocessor import Preprocessor
from src.data_processing.extract_epochs import EpochExtractor
from src.experiments.predictor import ExperimentPredictor
from src.utils.eeg_plotter import EEGPlotter
from src.data_processing.concatenate_epochs import EpochConcatenator
import time
import numpy as np


class PredictOrchestrator:
	"""
	The main orchestrator: parses CLI args, runs the entire experiment.
	"""
	def __init__(self):
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
			print(f"\033[1;32mAccuracy of current experiment {run_key} is: {accuracy}\nCross-validation with Kfold mean is: {crossval_mean}\033[0m")
			time.sleep(2)

			#if events are not baseline
			if group['runs'][0] not in [1, 2]: #this would be with all 6 experiments, but just a better outcome in any case.
				total_mean_accuracy_events.append(accuracy)

		# Print final results
		if total_mean_accuracy_events:
			print(f"\033[1;32mMean accuracy of event-based experiments: {np.mean(total_mean_accuracy_events):.3f}\033[0m")
			time.sleep(1)
		else:
			print("No event-based experiments processed.")