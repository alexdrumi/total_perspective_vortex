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
# from epoch_processor import EpochProcessor
from feature_extractor import FeatureExtractor

import joblib
import matplotlib.pyplot as plt

import logging
# from feature_extractor import feature_extractor, create_feature_vectors, calculate_mean_power_energy

from pca import My_PCA
from sklearn.preprocessing import FunctionTransformer

from epoch_extractor import EpochExtractor

import time

from custom_scaler import CustomScaler
from reshaper import Reshaper


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



channels = ["Fc3.", "Fcz.", "Fc4.", "C3..", "C1..", "Cz..", "C2..", "C4.."]
predict = [

	#3,7,11
	"../data/S060/S060R03.edf",
	"../data/S060/S060R11.edf",
	"../data/S060/S060R07.edf",
	# "../data/S061/S061R03.edf",
	# "../data/S061/S061R11.edf",
	# "../data/S061/S061R07.edf",
	# "../data/S062/S062R03.edf",
	# "../data/S062/S062R11.edf",
	# "../data/S062/S062R07.edf",
	# "../data/S063/S063R03.edf",
	# "../data/S063/S063R11.edf",
	# "../data/S063/S063R07.edf",
	# "../data/S064/S064R03.edf",
	# "../data/S064/S064R11.edf",
	# "../data/S064/S064R07.edf",
	# "../data/S065/S065R03.edf",
	# "../data/S065/S065R11.edf",
	# "../data/S065/S065R7.edf",
	# "../data/S066/S066R03.edf",
	# "../data/S066/S066R11.edf",
	# "../data/S066/S066R7.edf",
	# "../data/S067/S067R03.edf",
	# "../data/S067/S067R11.edf",
	# "../data/S067/S067R7.edf",
	# "../data/S068/S068R03.edf",
	# "../data/S068/S068R11.edf",
	# "../data/S068/S068R7.edf",
	# "../data/S069/S069R03.edf",
	# "../data/S069/S069R11.edf",
	# "../data/S069/S069R7.edf",
	# "../data/S070/S070R03.edf",
	# "../data/S070/S070R11.edf",
	# "../data/S070/S070R7.edf",
	# "../data/S071/S071R03.edf",
	# "../data/S071/S071R11.edf",
	# "../data/S071/S071R7.edf",
	# "../data/S072/S072R03.edf",
	# "../data/S072/S072R11.edf",
	# "../data/S072/S072R7.edf",
	# "../data/S073/S073R03.edf",
	# "../data/S073/S073R11.edf",
	# "../data/S073/S073R7.edf",
	# "../data/S074/S074R03.edf",
	# "../data/S074/S074R11.edf",
	# "../data/S074/S074R7.edf",
	# "../data/S075/S075R03.edf",
	# "../data/S075/S075R11.edf",
	# "../data/S075/S075R7.edf",
	# "../data/S076/S076R03.edf",
	# "../data/S076/S076R11.edf",
	# "../data/S076/S076R7.edf",
	# "../data/S077/S077R03.edf",
	# "../data/S077/S077R11.edf",
	# "../data/S077/S077R7.edf",
	# "../data/S078/S078R03.edf",
	# "../data/S078/S078R11.edf",
	# "../data/S078/S078R7.edf",
	# "../data/S079/S079R03.edf",
	# "../data/S079/S079R11.edf",
	# "../data/S079/S079R7.edf",

	#4,8,12
	# "../data/S060/S060R04.edf",
	# "../data/S060/S060R08.edf",
	# "../data/S060/S060R12.edf",
	# "../data/S061/S061R04.edf",
	# "../data/S061/S061R08.edf",
	# "../data/S061/S061R12.edf",
	# "../data/S062/S062R04.edf",
	# "../data/S062/S062R08.edf",
	# "../data/S062/S062R12.edf",
	# "../data/S063/S063R04.edf",
	# "../data/S063/S063R08.edf",
	# "../data/S063/S063R12.edf",
	# "../data/S064/S064R04.edf",
	# "../data/S064/S064R08.edf",
	# "../data/S064/S064R12.edf",
	# "../data/S065/S065R04.edf",
	"../data/S065/S065R08.edf",
	"../data/S065/S065R12.edf",
	"../data/S066/S066R04.edf",
	# "../data/S066/S066R08.edf",
	# "../data/S066/S066R12.edf",
	# "../data/S067/S067R04.edf",
	# "../data/S067/S067R08.edf",
	# "../data/S067/S067R12.edf",
	# "../data/S068/S068R04.edf",
	# "../data/S068/S068R08.edf",
	# "../data/S068/S068R12.edf",
	# "../data/S069/S069R04.edf",
	# "../data/S069/S069R08.edf",
	# "../data/S069/S069R12.edf",
	# "../data/S070/S070R04.edf",
	# "../data/S070/S070R08.edf",
	# "../data/S070/S070R12.edf",
	# "../data/S071/S071R04.edf",
	# "../data/S071/S071R08.edf",
	# "../data/S071/S071R12.edf",
	# "../data/S072/S072R04.edf",
	# "../data/S072/S072R08.edf",
	# "../data/S072/S072R12.edf",
	# "../data/S073/S073R04.edf",
	# "../data/S073/S073R08.edf",
	# "../data/S073/S073R12.edf",
	# "../data/S074/S074R04.edf",
	# "../data/S074/S074R08.edf",
	# "../data/S074/S074R12.edf",
	# "../data/S075/S075R04.edf",
	# "../data/S075/S075R08.edf",
	# "../data/S075/S075R12.edf",
	# "../data/S076/S076R04.edf",
	# "../data/S076/S076R08.edf",
	# "../data/S076/S076R12.edf",
	# "../data/S077/S077R04.edf",
	# "../data/S077/S077R08.edf",
	# "../data/S077/S077R12.edf",
	# "../data/S078/S078R04.edf",
	# "../data/S078/S078R08.edf",
	# "../data/S078/S078R12.edf",
	# "../data/S079/S079R04.edf",
	# "../data/S079/S079R08.edf",
	# "../data/S079/S079R12.edf",




	#5,9,13
	"../data/S060/S060R05.edf",
	"../data/S060/S060R09.edf",
	"../data/S060/S060R13.edf",
	# "../data/S061/S061R05.edf",
	# "../data/S061/S061R09.edf",
	# "../data/S061/S061R13.edf",
	# "../data/S062/S062R05.edf",
	# "../data/S062/S062R09.edf",
	# "../data/S062/S062R13.edf",
	# "../data/S063/S063R05.edf",
	# "../data/S063/S063R09.edf",
	# "../data/S063/S063R13.edf",
	# "../data/S064/S064R05.edf",
	# "../data/S064/S064R09.edf",
	# "../data/S064/S064R13.edf",
	# "../data/S065/S065R05.edf",
	# "../data/S065/S065R09.edf",
	# "../data/S065/S065R13.edf",
	# "../data/S066/S066R05.edf",
	# "../data/S066/S066R09.edf",
	# "../data/S066/S066R13.edf",
	# "../data/S067/S067R05.edf",
	# "../data/S067/S067R09.edf",
	# "../data/S067/S067R13.edf",
	# "../data/S068/S068R05.edf",
	# "../data/S068/S068R09.edf",
	# "../data/S068/S068R13.edf",
	# "../data/S069/S069R05.edf",
	# "../data/S069/S069R09.edf",
	# "../data/S069/S069R13.edf",
	# "../data/S070/S070R05.edf",
	# "../data/S070/S070R09.edf",
	# "../data/S070/S070R13.edf",
	# "../data/S071/S071R05.edf",
	# "../data/S071/S071R09.edf",
	# "../data/S071/S071R13.edf",
	# "../data/S072/S072R05.edf",
	# "../data/S072/S072R09.edf",
	# "../data/S072/S072R13.edf",
	# "../data/S073/S073R05.edf",
	# "../data/S073/S073R09.edf",
	# "../data/S073/S073R13.edf",
	# "../data/S074/S074R05.edf",
	# "../data/S074/S074R09.edf",
	# "../data/S074/S074R13.edf",
	# "../data/S075/S075R05.edf",
	# "../data/S075/S075R09.edf",
	# "../data/S075/S075R13.edf",
	# "../data/S076/S076R05.edf",
	# "../data/S076/S076R09.edf",
	# "../data/S076/S076R13.edf",
	# "../data/S077/S077R05.edf",
	# "../data/S077/S077R09.edf",
	# "../data/S077/S077R13.edf",
	# "../data/S078/S078R05.edf",
	# "../data/S078/S078R09.edf",
	# "../data/S078/S078R13.edf",
	# "../data/S079/S079R05.edf",
	# "../data/S079/S079R09.edf",
	# "../data/S079/S079R13.edf",


	#6,10,14
	# "../data/S060/S060R06.edf",
	# "../data/S060/S060R10.edf",
	# "../data/S060/S060R14.edf",
	# "../data/S061/S061R06.edf",
	# "../data/S061/S061R10.edf",
	# "../data/S061/S061R14.edf",
	# "../data/S062/S062R06.edf",
	# "../data/S062/S062R10.edf",
	# "../data/S062/S062R14.edf",
	# "../data/S063/S063R06.edf",
	# "../data/S063/S063R10.edf",
	# "../data/S063/S063R14.edf",
	# "../data/S064/S064R06.edf",
	# "../data/S064/S064R10.edf",
	# "../data/S064/S064R14.edf",
	# "../data/S065/S065R06.edf",
	# "../data/S065/S065R10.edf",
	"../data/S065/S065R14.edf",
	"../data/S066/S066R06.edf",
	"../data/S066/S066R10.edf",
	"../data/S066/S066R14.edf",
	"../data/S067/S067R06.edf",
	"../data/S067/S067R10.edf",
	# "../data/S067/S067R14.edf",
	# "../data/S068/S068R06.edf",
	# "../data/S068/S068R10.edf",
	# "../data/S068/S068R14.edf",
	# "../data/S069/S069R06.edf",
	# "../data/S069/S069R10.edf",
	# "../data/S069/S069R14.edf",
	# "../data/S070/S070R06.edf",
	# "../data/S070/S070R10.edf",
	# "../data/S070/S070R14.edf",
	# "../data/S071/S071R06.edf",
	# "../data/S071/S071R10.edf",
	# "../data/S071/S071R14.edf",
	# "../data/S072/S072R06.edf",
	# "../data/S072/S072R10.edf",
	# "../data/S072/S072R14.edf",
	# "../data/S073/S073R06.edf",
	# "../data/S073/S073R10.edf",
	# "../data/S073/S073R14.edf",
	# "../data/S074/S074R06.edf",
	# "../data/S074/S074R10.edf",
	# "../data/S074/S074R14.edf",
	# "../data/S075/S075R06.edf",
	# "../data/S075/S075R10.edf",
	# "../data/S075/S075R14.edf",
	# "../data/S076/S076R06.edf",
	# "../data/S076/S076R10.edf",
	# "../data/S076/S076R14.edf",
	# "../data/S077/S077R06.edf",
	# "../data/S077/S077R10.edf",
	# "../data/S077/S077R14.edf",
	# "../data/S078/S078R06.edf",
	# "../data/S078/S078R10.edf",
	# "../data/S078/S078R14.edf",
	# "../data/S079/S079R06.edf",
	# "../data/S079/S079R10.edf",
	# "../data/S079/S079R14.edf",
]

#-------------------------------------------------------
def concatenate_all_epochs(epochs_chunk, labels_chunk, predictions_chunk):
	'''
	Return concatenated epochs and alsoo event times
	'''
	n_epochs = len(epochs_chunk)
	epoch_duration = 7.1  #seconds -2.1+5 (whole range for features per epoch)
	total_duration = n_epochs * epoch_duration

	concatenated_data = []
	event_times = []  #to mark epoch boundaries
	concatenated_labels = []  #to store labels for annotations
	concatenated_predictions = []  #to store predictions for annotations

	for idx, (epoch, label, pred) in enumerate(zip(epochs_chunk, labels_chunk, predictions_chunk)):
		# epoch is an np array of shape (n_channels, n_times)
		mean_data = epoch.mean(axis=0)  #mean across channels; shape: (n_times,)
		concatenated_data.append(mean_data)
		event_times.append(idx * epoch_duration)
		concatenated_labels.append(label)
		concatenated_predictions.append(pred)

	#concat all epochs data
	concatenated_data = np.concatenate(concatenated_data)  #shape: (n_epochs * n_times,)
	return concatenated_data, event_times



def plot_eeg_epochs_chunk(current_batch_idx, epochs_chunk, labels_chunk, predictions_chunk, ax, alpha=0.3, linewidth=0.7):
	"""
	Plots a chunk of 21 epochs on a single continuous plot with annotations indicating correctness.
	#still buggy after a lot of tries but eventually i ll figure out how to make it better 
	Parameters:
	- epochs_chunk: list of NumPy arrays (21 epochs), each array shape (n_channels, n_times)
	- labels_chunk: NumPy array of true labels corresponding to each epoch (21 labels)
	- predictions_chunk: NumPy array of predicted labels corresponding to each epoch (21 predictions)
	- ax: Matplotlib Axes object to plot on
	- alpha: Transparency level for the plot lines (default is 0.3)
	- linewidth: Width of the EEG signal line (default is 0.7)
	"""
	n_epochs = len(epochs_chunk)
	epoch_duration = 7.1  # seconds
	total_duration = n_epochs * epoch_duration
	concatenated_data, event_times = concatenate_all_epochs(epochs_chunk, labels_chunk, predictions_chunk)   #shape: (n_epochs * n_times,) 	->concatenate all epochs data, for efficiency also extract the event times within the same loop

	#create a time array
	times = np.linspace(0, total_duration, concatenated_data.shape[0])

	ax.clear() #everytime we clear the plot, like this there will be no popup in each loop
	ax.plot(times, concatenated_data, label='EEG Signal', alpha=alpha, linewidth=linewidth) #plot the concatenated data with adjusted alpha and linewidth


	#add vertical dashed lines to separate epoch boundaries
	for event_time in event_times:
		ax.axvline(x=event_time, color='gray', linestyle='--', linewidth=0.5)

	y_min, y_max = ax.get_ylim() #determine y-pos for annotations
	annotation_y = y_max - 0.05 * (y_max - y_min)  #slightly below the top

	#annotate each epoch with true and predicted labels
	for idx, event_time in enumerate(event_times):
		true_label = 'Left' if labels_chunk[idx] == 0 else 'Right'
		predicted_label = 'Left' if predictions_chunk[idx] == 0 else 'Right'
		is_correct = (labels_chunk[idx] == predictions_chunk[idx])
		# print(is_correct)
		annotation = f"{true_label} / {predicted_label}"
		color = 'green' if is_correct else 'red'
		# print(color)
		#color is pissing me off, this should be red sometimes.
		ax.text(
			event_time + epoch_duration / 2,  #position at the center of the epoch
			annotation_y,                     #fixed y-position for alignment
			annotation,
			horizontalalignment='center',
			verticalalignment='bottom',        #align text to bottom
			fontsize=5,
			fontweight='bold',
			color=color,
			bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1)
		)

	#optional: adjust y-limits to provide space for annotations
	padding = 0.1 * (y_max - y_min)
	ax.set_ylim(y_min, y_max + padding)

	ax.set_title(f'Continuous EEG Data for 21 Epochs in Batch index {current_batch_idx}')
	ax.set_xlabel('Time (s)')
	ax.set_ylabel('Amplitude (µV)')
	ax.grid(True)
	ax.legend(loc='upper right', fontsize=8)
	plt.draw()
	plt.pause(1) #this was the problem, the pause didnt give it enough time to update ax



def display_epoch_stats(start, chunk_size, current_pred, current_labels):
		for i in range(chunk_size):
			epoch_num = start + i
			prediction = current_pred[i]
			truth = current_labels[i]
			is_correct = prediction == truth
			outcome = 'Left' if truth == 0 else 'Right'
			# print(f'epoch {epoch_num}: Prediction={prediction} | Truth={truth} ({outcome}) | Correct={is_correct}')


def main():
	try:
		dataset_preprocessor_instance = Preprocessor()
		loaded_raw_data = dataset_preprocessor_instance.load_raw_data(data_path=predict) #RETURN DOESNT WORK, IT RETURNS AFTER 1 FILE
		filtered_data = dataset_preprocessor_instance.filter_raw_data(loaded_raw_data) #THIS WILL BE INITIAL FILTER TRANSFORMER

		epoch_extractor_instance = EpochExtractor()
		epochs_predict, labels_predict = epoch_extractor_instance.extract_epochs_and_labels(filtered_data)

		# print(f"{len(epochs_predict['3'])} is epochs predict len, {len(labels_predict['3'])} is labels predict len")
		run_groups = epoch_extractor_instance.experiments_list
		i = 0
		for group in run_groups:
			# pipeline_name = 
			group_runs = group['runs']
			group_key = f"runs_{'_'.join(map(str, group_runs))}"
			models_to_load = f"../models/pipe_{group_key}.joblib" #names of the models to be loaded
			print(f'{models_to_load} are the models to load, type: {type(models_to_load)}')
			run_keys = list(set([run_key for run_key in epochs_predict.keys() if int(run_key[-2:]) in group_runs]))
			print(run_keys)
			time.sleep(5)
			# sys.exit(1)
			available_runs = list[set([run_key for run_key in run_keys if run_key in epochs_predict])]
			# sys.exit(1)

			if len(run_keys) == 0:
				continue
			
			# print(run_keys)
			# # print(available_runs)
			model_location = models_to_load
			print(model_location)
			# pipeline = joblib.load(model_location)
			# if pipeline == None:
			# 	print('pipeline hasnt been found, continue searching another.')
			try:
				pipeline = joblib.load(model_location)
				print('Pipeline loaded successfully.')
				# Proceed with using the pipeline
			except FileNotFoundError:
				print(f'Pipeline file not found at {model_location}. Continuing to search another.')
				pipeline = None
				continue


			feature_extractor_instance = FeatureExtractor()
			test_extracted_features = feature_extractor_instance.extract_features(epochs_predict[run_keys[0]]) 
			print(f'{test_extracted_features} are test extracted features')
			print(f'epoch nb:	[prediction]	[truth]		equal?')
			# sys.exit(1)

			i = 0 
			true_predictions_per_chunks = []
			# flattened_epochs = [epoch for file_epochs in epochs_predict[run_keys[0]] for epoch in file_epochs]
			flattened_epochs = epochs_predict[run_keys[0]]

			# print(f'{flattened_epochs} ARE FLATTENED EPOCHS, {len(flattened_epochs)} are the length')
			# sys.exit(1)

			#this is wrong here, we have the epochs from epochs predict. this is just 3,4 at the moment
			chunk_size = 21  #number of epochs per plot (per datafile)
			total_chunks = len(flattened_epochs) // chunk_size #chunk is at the moment all the epochs per datafile (21)
			# sys.exit(1)

			flattened_labels = labels_predict[run_keys[0]]  #already concatenated->for clarity that its flattened
			# print(f'{flattened_labels} are the flattened labels, {len(flattened_labels)} are the len')
			# sys.exit(1)
			chunk_size = 21  # Number of epochs per plot (per file)
			total_chunks = len(flattened_epochs) // chunk_size  # Should be 8

			true_predictions_per_chunks = []
			total_correct = 0
			print(f'epoch nb:	[prediction]	[truth]		equal?')

			#init live plot
			fig, ax = plt.subplots(figsize=(15, 6))
			plt.ion()  #turn on interactive mode
			print(f'{total_chunks} are total chunks')
			# sys.exit(1)

			for chunk_idx in range(total_chunks):
				print('INSIDE CHUNK INDEX')
				start = chunk_idx * chunk_size
				end = start + chunk_size
				current_features = test_extracted_features[start:end]
				current_labels = flattened_labels[start:end]
				current_epochs = flattened_epochs[start:end]

				#predict in batch
				start_time = time.time()
				current_pred = pipeline.predict(current_features)
				# sys.exit(1)
				print(current_pred)
				# print(current_pred)
				# print(current_labels)
				correct_predictions = np.sum(current_pred == current_labels)
				true_predictions_per_chunks.append(correct_predictions)
				total_correct += correct_predictions

				display_epoch_stats(start, chunk_size, current_pred, current_labels)

				epochs_data = current_epochs #list of (n_channels, n_times)-just for clarity, for now
				current_batch_accuracy = correct_predictions/len(current_labels)
				end_time = time.time()
				total_time_for_current_batch = end_time - start_time
				print(f'Current accuracy after processing epoch {start}-{end}: {current_batch_accuracy}.\nPrediction of this batch took: {total_time_for_current_batch} seconds of time.')
				# Plot the current chunk of 21 epochs with true labels and predictions
				plot_eeg_epochs_chunk(
					current_batch_idx=chunk_idx,
					epochs_chunk=epochs_data,
					labels_chunk=current_labels,
					predictions_chunk=current_pred,
					ax=ax,
					alpha=0.3,          #trnsparency as needed (e.g., 0.3 for higher transparency)
					linewidth=0.7       #linewidth for thinner lines
				)
				time.sleep(3)  #pause for real time plot, if this is too small, the plot will be buggy


			total_accuracy_on_this_test_set = np.sum(true_predictions_per_chunks)/len(test_extracted_features)
			print(f'{total_accuracy_on_this_test_set} is the total accuracy on this test set. Now we test with cross validation.')
			shuffle_split_validation = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
			scores = cross_val_score(
				pipeline, test_extracted_features, 
				labels_predict[run_keys[0]], 
				scoring='accuracy', 
				cv=shuffle_split_validation
			)

			print(scores)
			print(f'Average accuracy with cross-validation: {scores.mean()}')


	except FileNotFoundError as e:
		logging.error(f"File not found: {e}")
	except PermissionError as e:
		logging.error(f"Permission on the file denied: {e}")
	except IOError as e:
		logging.error(f"Error reading the data file: {e}")
	except ValueError as e:
		logging.error(f"Invalid EDF data: {e}")
	


if __name__ == "__main__":
	main()








'''


def main():
	try:
		dataset_preprocessor_instance = Preprocessor()
		loaded_raw_data = dataset_preprocessor_instance.load_raw_data(data_path=predict) #RETURN DOESNT WORK, IT RETURNS AFTER 1 FILE
		filtered_data = dataset_preprocessor_instance.filter_raw_data(loaded_raw_data) #THIS WILL BE INITIAL FILTER TRANSFORMER

		epoch_extractor_instance = EpochExtractor()
		epochs_predict, labels_predict = epoch_extractor_instance.extract_epochs_and_labels(filtered_data)
		pipeline = joblib.load('../models/pipe.joblib')
		
		feature_extractor_instance = FeatureExtractor()
		test_extracted_features = feature_extractor_instance.extract_features(epochs_predict) 
		
		print(f'epoch nb:	[prediction]	[truth]		equal?')
		
		i = 0 
		true_predictions_per_chunks = []
		flattened_epochs = [epoch for file_epochs in epochs_predict for epoch in file_epochs]
		chunk_size = 21  #number of epochs per plot (per datafile)
		total_chunks = len(flattened_epochs) // chunk_size #chunk is at the moment all the epochs per datafile (21)

		flattened_labels = labels_predict['3']  #already concatenated->for clarity that its flattened
		print(flattened_labels)
		chunk_size = 21  # Number of epochs per plot (per file)
		total_chunks = len(flattened_epochs) // chunk_size  # Should be 8

		true_predictions_per_chunks = []
		total_correct = 0
		print(f'epoch nb:	[prediction]	[truth]		equal?')

		#init live plot
		fig, ax = plt.subplots(figsize=(15, 6))
		plt.ion()  #turn on interactive mode
		for chunk_idx in range(total_chunks):
			start = chunk_idx * chunk_size
			end = start + chunk_size
			current_features = test_extracted_features[start:end]
			current_labels = flattened_labels[start:end]
			current_epochs = flattened_epochs[start:end]

			#predict in batch
			start_time = time.time()
			current_pred = pipeline.predict(current_features)
			# sys.exit(1)
			print(current_pred)
			# print(current_pred)
			# print(current_labels)
			correct_predictions = np.sum(current_pred == current_labels)
			true_predictions_per_chunks.append(correct_predictions)
			total_correct += correct_predictions

			display_epoch_stats(start, chunk_size, current_pred, current_labels)

			epochs_data = current_epochs #list of (n_channels, n_times)-just for clarity, for now
			current_batch_accuracy = correct_predictions/len(current_labels)
			end_time = time.time()
			total_time_for_current_batch = end_time - start_time
			print(f'Current accuracy after processing epoch {start}-{end}: {current_batch_accuracy}.\nPrediction of this batch took: {total_time_for_current_batch} seconds of time.')
			# Plot the current chunk of 21 epochs with true labels and predictions
			plot_eeg_epochs_chunk(
				current_batch_idx=chunk_idx,
				epochs_chunk=epochs_data,
				labels_chunk=current_labels,
				predictions_chunk=current_pred,
				ax=ax,
				alpha=0.3,          #trnsparency as needed (e.g., 0.3 for higher transparency)
				linewidth=0.7       #linewidth for thinner lines
			)
			time.sleep(3)  #pause for real time plot, if this is too small, the plot will be buggy


		total_accuracy_on_this_test_set = np.sum(true_predictions_per_chunks)/len(test_extracted_features)
		print(f'{total_accuracy_on_this_test_set} is the total accuracy on this test set. Now we test with cross validation.')
		shuffle_split_validation = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
		scores = cross_val_score(
			pipeline, test_extracted_features, 
			labels_predict['3'], 
			scoring='accuracy', 
			cv=shuffle_split_validation
		)

		print(scores)
		print(f'Average accuracy with cross-validation: {scores.mean()}')


	except FileNotFoundError as e:
		logging.error(f"File not found: {e}")
	except PermissionError as e:
		logging.error(f"Permission on the file denied: {e}")
	except IOError as e:
		logging.error(f"Error reading the data file: {e}")
	except ValueError as e:
		logging.error(f"Invalid EDF data: {e}")
	


if __name__ == "__main__":
	main()

'''