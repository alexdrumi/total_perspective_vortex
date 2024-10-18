import mne
import numpy as np
from feature_extractor import FeatureExtractor
from sklearn.base import BaseEstimator, TransformerMixin


class EpochExtractor(BaseEstimator, TransformerMixin):
	def __init__(self):
		# self.feature_extractor = feature_extractor_instance 
		pass


	def extract_epochs(self, data):
		event_id = {"T1": 1, "T2": 2}
		events, _ = mne.events_from_annotations(data)
		sfreq = data.info["sfreq"] #this is 160 but we could create a custom dataclass to pass this along, transform only expects an X output
		epochs = mne.Epochs(data, events, event_id=event_id, tmin=-2, tmax=5.1,
							baseline=None, preload=True)
		return epochs, sfreq


	def epoch_extractooor(self, X):
			'''
			Input: X->filtered eeg data, several eeg files thus we need a loop
			output: Filtered epochs (based on different timeframes and associated high/low frequencies)
			'''
			epochs_list = []

			for filtered_eeg_data in X:
				print('TRANSFORM IN EXTRACT EPOCHS')
				epochs, sfreq = self.extract_epochs(filtered_eeg_data)
				epochs_list.append(epochs)
			# print(f'{epochs_list} is the epochs list from transform')
			return epochs_list
	


def extract_epochs(data):
	event_id = {"T1": 1, "T2": 2}
	events, _ = mne.events_from_annotations(data)
	sfreq = data.info["sfreq"] #this is 160 but we could create a custom dataclass to pass this along, transform only expects an X output
	epochs = mne.Epochs(data, events, event_id=event_id, tmin=-2, tmax=5.1,
						baseline=None, preload=True)
	return epochs, sfreq


def epoch_extractooor(X):
		'''
		Input: X->filtered eeg data, several eeg files thus we need a loop
		output: Filtered epochs (based on different timeframes and associated high/low frequencies)
		'''
		epochs_list = []

		for filtered_eeg_data in X:
			print('TRANSFORM IN EXTRACT EPOCHS')
			epochs, sfreq = extract_epochs(filtered_eeg_data)
			epochs_list.append(epochs)
		# print(f'{epochs_list} is the epochs list from transform')
		return epochs_list