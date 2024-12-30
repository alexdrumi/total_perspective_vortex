
from src.pipeline.pipeline_builder import PipelineBuilder
from src.experiments.grid_search import GridSearchManager
from src.pipeline.pipeline_executor import PipelineExecutor
from src.utils.command_line_parser import CommandLineParser
from src.pipeline.feature_extractor import FeatureExtractor
from src.mlflow.mlflow_manager import MlflowManager
from src.data_processing.preprocessor import Preprocessor
from src.data_processing.extract_epochs import EpochExtractor
import mlflow

#create a facade which interacts with all the subsystems
class ExperimentTrainerFacade:
	"""
	A facade class to manage the end-to-end training and evaluation of the selected machine learning models.

	This class integrates multiple components such as data preprocessing, epoch extraction,
	feature extraction, pipeline building, grid search, and MLflow logging. It provides
	a streamlined interface for running experiments and processing groups of experimental runs.

	Attributes:
		command_line_parser (CommandLineParser): Handles command-line arguments.
		mlflow_manager (MlflowManager): Manages MLflow operations such as starting the server and logging experiments.
		data_preprocessor (Preprocessor): Handles raw data loading and filtering.
		extract_epochs (EpochExtractor): Extracts epochs and associated labels from the filtered data.
		feature_extractor (FeatureExtractor): Extracts features from epochs.
		pipeline_executor (PipelineExecutor): Executes and evaluates trained pipelines.
		pipeline_builder (PipelineBuilder): Builds machine learning pipelines.
		grid_search (GridSearchManager): Performs hyperparameter tuning using sklearn's GridSearch search.
		mlflow_enabled (bool): Flag indicating whether MLflow logging is enabled.

	Args:
		config_path (str): Path to the configuration file for grid search parameters. Default is `'../config/grid_search_parameters.yaml'`.
		mlflow_enabled (bool): Flag to enable or disable MLflow server. Default is `False`.

	Methods:
		run_experiment():
			Executes the experiment pipeline, including data preprocessing,
			feature extraction, pipeline building, and grid search. If enabled, logs metrics to MLflow.

		process_run_groups(run_groups, epochs_dict, labels_dict, mlflow_enabled):
			Processes each group of experimental runs, performs feature extraction,
			grid search, and logs results or saves the trained models.
	"""


	def __init__(self, config_path: str ='../config/grid_search_parameters.yaml', mlflow_enabled: bool =False) -> None:
		self.command_line_parser = CommandLineParser( #we can make this more elegant with passing another config?
			[{
					'name': '--mlflow',
					'type': str,
					'default': 'false',
					'choices': ['true', 'false'],
					'help':'Enable (True) or disable (False) the mlflow server for tracking model analysis. Default is False.\n'
			}]
		)
		self.mlflow_manager = MlflowManager()
		self.data_preprocessor = Preprocessor()
		self.extract_epochs = EpochExtractor()
		self.feature_extractor = FeatureExtractor()
		self.pipeline_executor = PipelineExecutor()
		self.pipeline_builder = PipelineBuilder(n_components=40)
		self.grid_search = GridSearchManager()
		self.mlflow_enabled = mlflow_enabled


	
	def run_experiment(self) -> None:
		"""
		Executes the full experiment pipeline.

		This method handles the following steps:
			Parses command-line arguments to determine if MLflow should be enabled.
			Starts the MLflow server if enabled.
			Loads and preprocesses raw data.
			Extracts epochs and associated labels.
			Processes experimental run groups to train and evaluate models.

		It prints status messages indicating progress through the pipeline.

		Raises:
			Exception: If any component fails during execution.
		"""
		self.mlflow_enabled = self.command_line_parser.parse_arguments()
		if (self.mlflow_enabled == True):
			self.mlflow_manager.start_mlflow_server()

		#load data
		raw_data = self.data_preprocessor.load_raw_data('../../config/train_data.yaml')
		filtered_data = self.data_preprocessor.filter_raw_data(raw_data) #this returns a triplet now

		#extract epochs and associated labels
		epochs_dict, labels_dict = self.extract_epochs.extract_epochs_and_labels(filtered_data)
		run_groups = self.extract_epochs.experiments_list

		#run the experiments on different groups
		self.process_run_groups(run_groups, epochs_dict, labels_dict, self.mlflow_enabled)



	def process_run_groups(self, run_groups: dict, epochs_dict: dict, labels_dict: dict, mlflow_enabled: bool) -> None:
		"""
		Processes each group of experimental runs to train and evaluate models.

		Args:
			run_groups (list): List of groups with experimental runs to process.
			epochs_dict (dict): Dictionary containing epochs for each experimental run.
			labels_dict (dict): Dictionary containing labels for each experimental run.
			mlflow_enabled (bool): Flag to enable or disable MLflow logging.

		It prints status messages and results for each processed group.

		Raises:
			ValueError: If no available runs exist for a group.
		"""
		for groups in run_groups:
			groups_runs = groups['runs']
			group_key = f'runs_{"_".join(map(str, groups_runs))}'
			print(f"\nProcessing group: {group_key} with runs {groups_runs[0]}")

			run_keys = [run_key for run_key in epochs_dict.keys() if int(run_key[-2:]) in groups_runs]
			available_runs = [run_key for run_key in run_keys if run_key in epochs_dict]

			if not available_runs:
				print(f"No available runs for group '{group_key}', skipping.")
				continue
			
			feature_extraction_method = 'baseline' if groups_runs[0] in [1,2] else 'events'
			
			#feature extraction
			X_train = self.feature_extractor.extract_features(epochs_dict[run_keys[0]], feature_extraction_method) #trained_extracted_features, for now groups runs[0] is ok but at 13 etc it wont be
			y_train = labels_dict[run_keys[0]]
			
			#build a pipeline
			pipeline = self.pipeline_builder.build_pipeline()

			#run the grid search
			best_params, best_score, best_pipeline = self.grid_search.run_grid_search(pipeline, X_train, y_train)
			print(f'types of bestparams {type(best_params)}, bestscore {type(best_score)}, bestpipelines {type(best_pipeline)}')

			if self.mlflow_enabled == True:
				#log metrics to mlflow
				with mlflow.start_run(run_name=group_key):
					#we could use the pipeline executor as an external function to save pipeline metrics?
					self.mlflow_manager.log_mlflow_experiment(group_key, best_params, best_score, best_pipeline, X_train, y_train) #this also dumps model
			else:
				#save model
				self.pipeline_executor.save_model(best_pipeline, group_key)
				#print cross val scores
				self.pipeline_executor.evaluate_pipeline(group_key, best_pipeline, X_train, y_train)


