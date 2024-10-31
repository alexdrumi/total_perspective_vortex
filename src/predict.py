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
# from epoch_extractor import epoch_extractooor, extract_epochs
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
"../data/S018/S018R11.edf",
"../data/S042/S042R07.edf",
"../data/S042/S042R03.edf",
# "data/S104/S104R11.edf",
# "data/S104/S104R07.edf",
# "data/S090/S090R11.edf",
# "data/S086/S086R11.edf",
# "data/S086/S086R03.edf",
#"/data/S086/S086R07.edf",
#"/data/S017/S017R11.edf",
#"/data/S017/S017R07.edf",
#"/data/S017/S017R03.edf",
#"/data/S013/S013R07.edf",
#"/data/S013/S013R11.edf",
#"/data/S013/S013R03.edf",
#"/data/S055/S055R11.edf",
#"/data/S055/S055R07.edf",
#"/data/S055/S055R03.edf",
#"/data/S016/S016R03.edf",
#"/data/S016/S016R07.edf",
#"/data/S016/S016R11.edf",
#"/data/S103/S103R11.edf",
]



def extract_epochs(data):
	event_id = {"T1": 1, "T2": 2}
	events, _ = mne.events_from_annotations(data)
	# events_other = mne.find_events(data)
	# print(f'{events_other} are OTHER EVENTS')
	print(f'{events} are possible events')
	sfreq = data.info["sfreq"] #this is 160 but we could create a custom dataclass to pass this along, transform only expects an X output
	epochs = mne.Epochs(data, events, event_id=event_id, tmin=-2, tmax=5.1,
						baseline=None, preload=True) #epochs are 7.1 secs each in extracted epoch (this results in 21 identified event, each 7.1 secs long)
	print(f'{len(epochs)} IS THE AMOUNT OF EPOCH LEN')
	return epochs, sfreq


#feature extractor have self,y and self feature matrix
#is ica filter hopeless?
def extract_epochs_and_labelsf(eeg_data):
	'''
	Input: X->filtered eeg data, several eeg data thus we need a loop
	output1: Filtered epochs on which we will run feature extraction (based on different timeframes and associated high/low frequencies)
	output2: the labels of the epoch events which we will use as y_train 
	'''
	epochs_list = [] #len of eeg data, each consists of 21 events, corresponding to (8*21=168 events of 7.1 seconds long)
	labels_list = [] #len of eeg data, each consists of 21 events, corresponding to (8*21=168 events of 7.1 seconds long)
	for filtered_eeg_data in eeg_data:
		# print('TRANSFORM IN EXTRACT EPOCHS')
		epochs, sfreq = extract_epochs(filtered_eeg_data)
		epochs_list.append(epochs)
		labels = epochs.events[:, 2]- 1 #these are all the same, prob enough jsut [2][1]
		labels_list.append(labels)
	# print(f'{epochs_list} is the epochs list from transform')
	# print(f'!!!!!!!!!!{len(epochs_list)} is len epoch list, {len(labels_list)} is len label list')
	# sys.exit(0)
	return epochs_list, np.concatenate(labels_list)



def feature_extractor_small(filtered_epochs):
	'''
	Input: current filtered epoch from EpochExtractor
	Output: a (x,y,z)d np array of created features based on mean, energy, power
	NO LABELS HERE, WILL DO SEPARATE
	'''

	analysis = {
		'mrcp': {'tmin': -2, 'tmax': 0, 'lofreq': 3, 'hifreq': 30},
		'erd': {'tmin': -2, 'tmax': 0, 'lofreq': 8, 'hifreq': 30},
		'ers': {'tmin': 4.1, 'tmax': 5.1, 'lofreq': 8, 'hifreq': 30}
	}

	feature_matrices = []
	labels = []
	for analysis_name, parameters in analysis.items():
		cropped_epochs = filtered_epochs.copy().crop(tmin=parameters['tmin'], tmax=parameters['tmax'])
		filtered_epoch = cropped_epochs.filter(h_freq=parameters['hifreq'],
												l_freq=parameters['lofreq'],
												method='iir')
		
		feature_matrix, y = create_feature_vectors(filtered_epoch, 160.0, compute_y=None)
		feature_matrices.append(feature_matrix)

		#check samples for consistent counts
		sample_counts = [fm.shape[0] for fm in feature_matrices]
		if not all(count == sample_counts[0] for count in sample_counts):
			raise ValueError("Inconsistent number of samples across analyses. Ensure all have the same number of epochs.")
	
	ret = np.concatenate(feature_matrices, axis=0) #this is now (59 epoch list, 21 epochs inside, 9*8 feature combinations) thus we need to concat them 
	return ret






#-------------------------------------------------------


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

	# print(labels_chunk)
	# print('Is the LABEL CHUNK')
	# print(predictions_chunk)
	# print('Is the PREDICXTIONS CHUNK')
	# sys.exit(0)
	n_epochs = len(epochs_chunk)
	epoch_duration = 7.1  # seconds
	total_duration = n_epochs * epoch_duration

	concatenated_data = []
	event_times = []  # To mark epoch boundaries
	concatenated_labels = []  # To store labels for annotations
	concatenated_predictions = []  # To store predictions for annotations

	for idx, (epoch, label, pred) in enumerate(zip(epochs_chunk, labels_chunk, predictions_chunk)):
		# epoch is a NumPy array of shape (n_channels, n_times)
		mean_data = epoch.mean(axis=0)  # Average across channels; shape: (n_times,)
		concatenated_data.append(mean_data)
		event_times.append(idx * epoch_duration)
		concatenated_labels.append(label)
		concatenated_predictions.append(pred)

	# Concatenate all epochs' data
	concatenated_data = np.concatenate(concatenated_data)  # Shape: (n_epochs * n_times,)

	# Create a time array
	times = np.linspace(0, total_duration, concatenated_data.shape[0])

	ax.clear()
	# Plot the concatenated data with adjusted alpha and linewidth
	ax.plot(times, concatenated_data, label='EEG Signal', alpha=alpha, linewidth=linewidth)

	# Add vertical dashed lines to demarcate epoch boundaries
	for event_time in event_times:
		ax.axvline(x=event_time, color='gray', linestyle='--', linewidth=0.5)

	# Determine y-position for annotations
	y_min, y_max = ax.get_ylim()
	annotation_y = y_max - 0.05 * (y_max - y_min)  # Slightly below the top

	# Annotate each epoch with true and predicted labels
	# print(len(event_times))
	# print('IS THE LENGTH OF EVENT TIMES')
	for idx, event_time in enumerate(event_times):
		# print(f'{labels_chunk[idx]} is the label chunk')
		# print(f'{predictions_chunk[idx]} is the predictions_chunk')

		true_label = 'Left' if labels_chunk[idx] == 0 else 'Right'
		predicted_label = 'Left' if predictions_chunk[idx] == 0 else 'Right'
		# print(concatenated_labels)
		# print('\n')
		# print(predicted_label)
		# print('\n')
		is_correct = (labels_chunk[idx] == predictions_chunk[idx])
		# print(is_correct)
		annotation = f"{true_label} / {predicted_label}"
		color = 'green' if is_correct else 'red'
		# print(color)
		#color is pissing me off, this should be red sometimes.
		ax.text(
			event_time + epoch_duration / 2,  # Position at the center of the epoch
			annotation_y,                     # Fixed y-position for alignment
			annotation,
			horizontalalignment='center',
			verticalalignment='bottom',        # Align text to bottom
			fontsize=5,
			fontweight='bold',
			color=color,
			bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1)
		)

	# Optional: Adjust y-limits to provide space for annotations
	padding = 0.1 * (y_max - y_min)
	ax.set_ylim(y_min, y_max + padding)

	ax.set_title(f'Continuous EEG Data for 21 Epochs in Batch index {current_batch_idx}')
	ax.set_xlabel('Time (s)')
	ax.set_ylabel('Amplitude (µV)')
	ax.grid(True)
	ax.legend(loc='upper right', fontsize=8)
	plt.draw()
	plt.pause(1) #this was the problem, the pause didnt give it enough time to update ax





def main():

	try:
		dataset_preprocessor_instance = Preprocessor()
		dataset_preprocessor_instance.load_raw_data(data_path=predict) #RETURN DOESNT WORK, IT RETURNS AFTER 1 FILE
		filtered_data = dataset_preprocessor_instance.filter_raw_data() #THIS WILL BE INITIAL FILTER TRANSFORMER

		epoch_extractor_instance = EpochExtractor()
		epochs_predict, labels_predict = epoch_extractor_instance.extract_epochs_and_labels(filtered_data)

	
	
		pipeline = joblib.load('../models/pipe.joblib')
		# dataset_preprocessor = Preprocessor()

		# predict_raw = dataset_preprocessor.load_raw_data(data_path=predict)
		# predict_filtered = dataset_preprocessor.filter_raw_data()

		# epochs_predict, labels_predict = extract_epochs_and_labelsf(predict_filtered)
		# feature_transformer = FunctionTransformer(feature_extractor)
		feature_extractor_instance = FeatureExtractor()
		test_extracted_features = feature_extractor_instance.extract_features(epochs_predict) 
		# # print(f'{len(epochs_predict)} is the len of the EPOCHS extracted from filtered, {len(labels_predict)} is the len of the labels predicted\n')
		# # print(f'{predict_raw.ch_names} ARE THE CH NAMES!!!!!')
		



		idx = 0
		print(f'epoch nb:	[prediction]	[truth]		equal?')
		chunk_range = 1 #len(test_extracted_features) // len(test_extracted_features)  #per 4 epochs lets say
		i = 0 
		true_predictions_per_chunks = []

		# print(f'{len(epochs_predict)} is the len of the predicted epochs')
		# print(f'{len(test_extracted_features)} is the len of the extracted features')
		flattened_epochs = [epoch for file_epochs in epochs_predict for epoch in file_epochs]
		# print(f'{len(flattened_epochs)} is the len of the flattened_epochs')
		# sys.exit(0)

		chunk_size = 21  # Number of epochs per plot (per file)
		total_chunks = len(flattened_epochs) // chunk_size  # Should be 8

		# Initialize live plot
		fig, ax = plt.subplots(figsize=(15, 6))
		plt.ion()  # Turn on interactive mode
		# while i < len(test_extracted_features):
		# 	# current_epoch = epochs_predict[i]
		# 	# current_epoch['T1'].plot_psd(picks='eeg')
		# 	# current_epoch['T1'].plot_psd_topomap()
			
		# 	start_time = time.time()
		# 	print(f'start time: {start_time}')
		# 	current_pred = pipeline.predict(test_extracted_features[i:i+chunk_range])
		# 	true_predictions = np.sum(current_pred == labels_predict[i:i+chunk_range])
		# 	true_predictions_per_chunks.append(true_predictions)
			
		# 	for compare_idx in range(len(current_pred)):
		# 		print(f'epoch nb:	[prediction]	[truth]		equal?')
		# 		is_prediction_true = (current_pred[compare_idx] == labels_predict[i+compare_idx])
		# 		print(f'epoch {compare_idx+idx}:	{current_pred[compare_idx]}	 {labels_predict[i+compare_idx]}	{is_prediction_true} \n')

		# 		if (i+compare_idx < len(test_extracted_features)):
		# 			print(f'{i+compare_idx} is the current idx')
		# 			current_epoch = epochs_predict[i + compare_idx].get_data()[0]  # Assuming 1 epoch at a time
		# 			print(type(current_epoch))
		# 			plot_eeg_epoch(current_epoch, ax)  # Update the plot with the new epoch data

		flattened_labels = labels_predict  # Already concatenated
		chunk_size = 21  # Number of epochs per plot (per file)
		total_chunks = len(flattened_epochs) // chunk_size  # Should be 8


		true_predictions_per_chunks = []
		total_correct = 0
		print(f'epoch nb:	[prediction]	[truth]		equal?')

		for chunk_idx in range(total_chunks):
			start = chunk_idx * chunk_size
			end = start + chunk_size
			current_features = test_extracted_features[start:end]
			current_labels = flattened_labels[start:end]
			current_epochs = flattened_epochs[start:end]

			#predict in batch
			start_time = time.time()
			current_pred = pipeline.predict(current_features)
			# print(current_pred)
			# print(current_labels)
			correct_predictions = np.sum(current_pred == current_labels)
			true_predictions_per_chunks.append(correct_predictions)
			total_correct += correct_predictions


			for i in range(chunk_size):
				epoch_num = start + i
				prediction = current_pred[i]
				truth = current_labels[i]
				is_correct = prediction == truth
				outcome = 'Left' if truth == 0 else 'Right'
				print(f'epoch {epoch_num}: Prediction={prediction} | Truth={truth} ({outcome}) | Correct={is_correct}')

			epochs_data = current_epochs# List of (n_channels, n_times)
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
				alpha=0.3,          # Adjust alpha for transparency as needed (e.g., 0.3 for higher transparency)
				linewidth=0.7       # Adjust linewidth for thinner lines
			)

			# Plot the current chunk of 21 epochs with true labels and predictions
			# plot_eeg_epochs_chunk(current_epochs, current_labels, current_pred, ax)

			# Optional: Pause to simulate real-time plotting
			time.sleep(3)  # Adjust as needed
			
		total_accuracy_on_this_test_set = np.sum(true_predictions_per_chunks)/len(test_extracted_features)
		print(f'{total_accuracy_on_this_test_set} is the total accuracy on this test set. Now we test with cross validation.')
		shuffle_split_validation = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
		scores = cross_val_score(
			pipeline, test_extracted_features, 
			labels_predict, 
			scoring='accuracy', 
			cv=shuffle_split_validation
		)

		print(scores)
		print(f'Average accuracy: {scores.mean()}')


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