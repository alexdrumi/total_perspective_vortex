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
# "../data/S018/S018R11.edf",
# "../data/S042/S042R07.edf",
# "../data/S042/S042R03.edf",
# "../data/S104/S104R11.edf",
# "../data/S104/S104R07.edf",
# "../data/S090/S090R11.edf",
# "../data/S086/S086R11.edf",
# "../data/S086/S086R03.edf",
# "../data/S086/S086R07.edf",
# "../data/S017/S017R11.edf",
# "../data/S017/S017R07.edf",


"../data/S098/S098R03.edf",
"../data/S098/S098R07.edf",
"../data/S098/S098R11.edf",

"../data/S051/S051R03.edf",
"../data/S051/S051R07.edf",
"../data/S051/S051R11.edf",

"../data/S052/S052R03.edf",
"../data/S052/S052R07.edf",
"../data/S052/S052R11.edf",

"../data/S053/S053R03.edf",
"../data/S053/S053R07.edf",
"../data/S053/S053R11.edf",



"../data/S054/S054R03.edf",
"../data/S054/S054R07.edf",
"../data/S054/S054R11.edf",



"../data/S061/S061R03.edf",
"../data/S061/S061R07.edf",
"../data/S061/S061R11.edf",

"../data/S062/S062R03.edf",
"../data/S062/S062R07.edf",
"../data/S062/S062R11.edf",

"../data/S063/S063R03.edf",
"../data/S063/S063R07.edf",
"../data/S063/S063R11.edf",



"../data/S064/S064R03.edf",
"../data/S064/S064R07.edf",
"../data/S064/S064R11.edf",

#"../data/S017/S017R03.edf",
#..//data/S013/S013R07.edf",
#"../data/S013/S013R11.edf",
#"/data/S013/S013R03.edf",
#"/data/S055/S055R11.edf",
#"/data/S055/S055R07.edf",
#"/data/S055/S055R03.edf",
#"/data/S016/S016R03.edf",
#"/data/S016/S016R07.edf",
#"/data/S016/S016R11.edf",
#"/data/S103/S103R11.edf",
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
			print(f'epoch {epoch_num}: Prediction={prediction} | Truth={truth} ({outcome}) | Correct={is_correct}')


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