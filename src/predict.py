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

from dataset_preprocessor import Preprocessor
from feature_extractor import FeatureExtractor

import joblib
import logging

from pca import My_PCA
from epoch_extractor import EpochExtractor

import time

from custom_scaler import CustomScaler
from reshaper import Reshaper


logger = logging.getLogger()
logger.setLevel(logging.ERROR)

file_handler = logging.FileHandler('../logs/error_log.log', mode='w')
file_handler.setLevel(logging.ERROR)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)



channels = ["Fc3.", "Fcz.", "Fc4.", "C3..", "C1..", "Cz..", "C2..", "C4.."]
predict = [

	#3,7,11
	"../data/S020/S020R03.edf",
	"../data/S020/S020R07.edf",
	"../data/S020/S020R11.edf",
	# "../data/S021/S021R03.edf",
	# "../data/S021/S021R07.edf",
	# "../data/S021/S021R11.edf",
	# "../data/S022/S022R03.edf",
	# "../data/S022/S022R07.edf",
	# "../data/S022/S022R11.edf",

	#4,8,12
	"../data/S011/S011R08.edf",
	"../data/S011/S011R12.edf",
	"../data/S012/S012R04.edf",
	# "../data/S012/S012R08.edf",
	# "../data/S012/S012R12.edf",
	# "../data/S013/S013R04.edf",
	# "../data/S013/S013R08.edf",
	# "../data/S013/S013R12.edf",


	#5,9,13
	"../data/S011/S011R09.edf",
	"../data/S011/S011R13.edf",
	"../data/S012/S012R05.edf",
	# "../data/S012/S012R09.edf",
	# "../data/S012/S012R13.edf",
	# "../data/S013/S013R05.edf",
	# "../data/S013/S013R09.edf",
	# "../data/S013/S013R13.edf",


	#6,10,14
	"../data/S027/S027R06.edf",
	"../data/S027/S027R10.edf",
	"../data/S027/S027R14.edf",
	# "../data/S028/S028R06.edf",
	# "../data/S028/S028R10.edf",
	# "../data/S028/S028R14.edf",
	# "../data/S029/S029R06.edf",
]
#-------------------------------------------------------

def concatenate_all_epochs(epochs_chunk, labels_chunk, predictions_chunk):
	'''
	return concatenated epochs and alsoo event times
	'''
	n_epochs = len(epochs_chunk)
	epoch_duration = 7.1
	total_duration = n_epochs * epoch_duration

	concatenated_data = []
	event_times = []
	concatenated_labels = []
	concatenated_predictions = []

	for idx, (epoch, label, pred) in enumerate(zip(epochs_chunk, labels_chunk, predictions_chunk)):
		mean_data = epoch.mean(axis=0)
		concatenated_data.append(mean_data)
		event_times.append(idx * epoch_duration)
		concatenated_labels.append(label)
		concatenated_predictions.append(pred)

	concatenated_data = np.concatenate(concatenated_data)
	return concatenated_data, event_times



def plot_eeg_epochs_chunk(current_batch_idx, epochs_chunk, labels_chunk, predictions_chunk, label_names, ax, alpha=0.3, linewidth=0.7):
	"""
	plots a chunk of 21 epochs on a single continuous plot with annotations indicating correctness.
	#still buggy after a lot of tries but eventually i ll figure out how to make it better 
	parameters:
	- epochs_chunk: list of NumPy arrays (21 epochs), each array shape (n_channels, n_times)
	- labels_chunk: NumPy array of true labels corresponding to each epoch (21 labels)
	- predictions_chunk: NumPy array of predicted labels corresponding to each epoch (21 predictions)
	- ax: Matplotlib Axes object to plot on
	- alpha: Transparency level for the plot lines (default is 0.3)
	- linewidth: Width of the EEG signal line (default is 0.7)
	"""
	n_epochs = len(epochs_chunk)
	epoch_duration = 7.1
	total_duration = n_epochs * epoch_duration
	concatenated_data, event_times = concatenate_all_epochs(epochs_chunk, labels_chunk, predictions_chunk)

	times = np.linspace(0, total_duration, concatenated_data.shape[0])

	ax.clear()
	ax.plot(times, concatenated_data, label='EEG Signal', alpha=alpha, linewidth=linewidth)

	for event_time in event_times:
		ax.axvline(x=event_time, color='gray', linestyle='--', linewidth=0.5)

	y_min, y_max = ax.get_ylim()
	annotation_y = y_max - 0.05 * (y_max - y_min)

	for idx, event_time in enumerate(event_times):
		true_label = label_names[1] if labels_chunk[idx] == 0 else label_names[2]
		predicted_label = label_names[1] if predictions_chunk[idx] == 0 else label_names[2]
		is_correct = (labels_chunk[idx] == predictions_chunk[idx])
		annotation = f"{true_label}\n----------------\n{predicted_label}"
		color = 'green' if is_correct else 'red'
		ax.text(
			event_time + epoch_duration / 2,
			annotation_y,
			annotation,
			horizontalalignment='center',
			verticalalignment='bottom',
			fontsize=4,
			fontweight='bold',
			color=color,
			bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1)
		)

	padding = 0.1 * (y_max - y_min)
	ax.set_ylim(y_min, y_max + padding)

	ax.set_title(f'Continuous EEG Data for 15 Epochs in Batch index {current_batch_idx}')
	ax.set_xlabel('Time (s)')
	ax.set_ylabel('Amplitude (µV)')
	ax.grid(True)
	ax.legend(loc='upper right', fontsize=8)
	plt.draw()
	plt.pause(1)



def display_epoch_stats(start, chunk_size, current_pred, current_labels):
	for i in range(chunk_size):
		epoch_num = start + i
		prediction = current_pred[i]
		truth = current_labels[i]
		is_correct = prediction == truth
		outcome = 'Left' if truth == 0 else 'Right'


def main():
	try:
		dataset_preprocessor_instance = Preprocessor()
		loaded_raw_data = dataset_preprocessor_instance.load_raw_data(data_path=predict)
		filtered_data = dataset_preprocessor_instance.filter_raw_data(loaded_raw_data)

		epoch_extractor_instance = EpochExtractor()
		epochs_predict, labels_predict = epoch_extractor_instance.extract_epochs_and_labels(filtered_data)
		run_groups = epoch_extractor_instance.experiments_list

		i = 0
		for group in run_groups:
			group_runs = group['runs']
			group_key = f"runs_{'_'.join(map(str, group_runs))}"
			models_to_load = f"../models/pipe_{group_key}.joblib"
			run_keys = list(set([run_key for run_key in epochs_predict.keys() if int(run_key[-2:]) in group_runs]))
			time.sleep(5)
			available_runs = list[set([run_key for run_key in run_keys if run_key in epochs_predict])]

			if len(run_keys) == 0:
				continue
			model_location = models_to_load
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

			i = 0 
			flattened_epochs = epochs_predict[run_keys[0]]
			flattened_labels = labels_predict[run_keys[0]]

			chunk_size = 15
			true_predictions_per_chunks = []
			total_chunks = len(flattened_epochs) // chunk_size
			true_predictions_per_chunks = []
			total_correct = 0
			print(f'epoch nb:	[prediction]	[truth]		equal?')

			fig, ax = plt.subplots(figsize=(15, 6))
			plt.ion()

			for chunk_idx in range(total_chunks):
				start = chunk_idx * chunk_size
				end = start + chunk_size
				current_features = test_extracted_features[start:end]
				current_labels = flattened_labels[start:end]
				current_epochs = flattened_epochs[start:end]

				start_time = time.time()
				current_pred = pipeline.predict(current_features)
				
				correct_predictions = np.sum(current_pred == current_labels)
				true_predictions_per_chunks.append(correct_predictions)
				total_correct += correct_predictions

				epochs_data = current_epochs
				current_batch_accuracy = correct_predictions/len(current_labels)
				end_time = time.time()
				total_time_for_current_batch = end_time - start_time
				print(f'Current accuracy after processing epoch {start}-{end}: {current_batch_accuracy}.\nPrediction of this batch took: {total_time_for_current_batch} seconds of time.')

				label_names = group['mapping']
				plot_eeg_epochs_chunk(
					current_batch_idx=chunk_idx,
					epochs_chunk=epochs_data,
					labels_chunk=current_labels,
					predictions_chunk=current_pred,
					label_names=label_names,
					ax=ax,
					alpha=0.3,
					linewidth=0.7
				)
				time.sleep(3)


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
			print(f'Average accuracy with cross-validation for group:{group_runs}: {scores.mean()}')


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
