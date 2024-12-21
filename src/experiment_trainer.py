
from pipeline_builder import PipelineBuilder
from grid_search_manager import GridSearchManager
from pipeline_executor import PipelineExecutor
from command_line_parser import CommandLineParser

#create a facade which has all the subsystems
class ExperimentTrainerFacade:
	def __init__(config_path='../configs/grid_search_parameters.yaml', mlflow_enabled=False):
		self.command_line_parser = CommandLineParser({
					'name': '--mlfow',
					'type': str,
					'default': 'false',
					'choices': ['true', 'false'],
					'help':'Enable (True) or disable (False) the mlflow server for tracking model analysis. Default is False.\n'
				})
		self.mlflow_manager = MlflowManager()
		self.data_preprocessor = Preprocessor()
		self.epoch_extractor = EpochExtractor()
		self.pipeline_executor = PipelineExecutor()
		self.pipeline_builder = PipelineBuilder(n_components=40)
		self.grid_search_manager = GridSearchManager()


	def run_experiment(self):
		mlflow_enabled = self.command_line_parser.arg_parser.parse_arguments()
		if (mlflow_enabled == True):
			self.mlflow_manager.start_mlflow_server()

		#load data
		raw_data = self.data_preprocessor.load_raw_data(data_path=train)
		filtered_data = dataset_preprocessor_instance.filter_raw_data(raw_data) #this returns a triplet now

		#extract epochs and associated labels
		epochs_dict, labels_dict = epoch_extractor_instance.extract_epochs_and_labels(filtered_data)
		run_groups = epoch_extractor_instance.experiments_list
		
		#run the experiments on different groups
		self.process_run_groups(run_groups, mlflow_enabled)



	def process_run_groups(self, run_groups, mlflow_enabled):
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
			feature_extractor_instance = FeatureExtractor()
			X_train = feature_extractor_instance.extract_features(epochs_dict[run_keys[0]], feature_extraction_method) #trained_extracted_features, for now groups runs[0] is ok but at 13 etc it wont be
			y_train = labels_dict[run_keys[0]] #trained_extracted_labels
			
			#build a pipeline
			pipeline = pipeline_builder.build_pipeline()

			#run the grid search
			best_params, best_score, best_pipeline = self.grid_search_manager.run_grid_search(pipeline, X_train, y_train)
			
			if mlflow_enabled == True:
				# log metrics to mlflow
				with mlflow.start_run(run_name=group_key):
					#we could use the pipeline executor as an external function to save pipeline metrics?
					self.mlflow_manager.log_mlflow_experiment(group_key, best_params, best_score, best_pipeline, X_train, y_train) #this also dumps model
			else:
				#save model
				self.pipeline_executor.save_model(best_pipeline, group_key)
				#print cross val scores
				self.pipeline_executor.evaluate_pipeline(group_key, best_pipeline, best_score)


