import mne
import numpy as np
from typing import List, Tuple


#we dont need the baseestimator and transformermixin here actually
class EpochExtractor:
	def __init__(self):
		self.experiments_list = [
			# {
			# 	"runs": [3,7,11],
			# 	"mapping": {0: "rest", 1: "left fist", 2: "right fist"},
			# 	"event_id": {"left fist": 2, "right fist": 3},
			# },
			{
				"runs": [4,8,12],
				"mapping": {0: "rest", 1: "left imagined", 2: "right imagined"},
				"event_id": {"left fist imagined": 4, "right fist imagined": 5},
			},
			{
				"runs": [5,9,13],
				"mapping": {0: "rest", 1: "both fists", 2: "both feet"},
				"event_id": {"both fists": 6, "both feet": 7},
			},
			# {
			# 	"runs": [6, 10, 14],
			# 	"mapping": {0: "rest", 1: "both fists imagined", 2: "both feet imagined"},
			# 	"event_id": {"imagine both fists": 8, "imagine both feet": 9},
			# },
		]



	def extract_epochs(self, data: mne.io.Raw) -> Tuple[mne.epochs.Epochs, float]:
		# event_id = {"T1": 1, "T2": 2} #here include T0
		event_id = {"T0": 1, "T1": 2, "T2": 3} #here include T0
		events, _ = mne.events_from_annotations(data)
		# print(f'{_} are event ids')
		sfreq = data.info["sfreq"]
		epochs = mne.Epochs(data, events, event_id=event_id, tmin=-2, tmax=5.1,
							baseline=None, preload=True)
		
		return epochs, sfreq



	def extract_epochs_and_labels(self, filtered_eeg_data: mne.io.Raw) -> Tuple[list, np.ndarray]:
			'''
			Input: X->filtered eeg data, several eeg data thus we need a loop
			output: Filtered epochs (based on different timeframes and associated high/low frequencies)
			'''
			epochs_for_multiple_files = {}
			labels_for_multiple_files = {}
			for key, raw_data in filtered_eeg_data.items():
				if raw_data is not None:
					current_epochs, _ = self.extract_epochs(raw_data)
					print(f'{_} is sfreq')
					epochs_for_multiple_files[key] = current_epochs 

					current_labels = current_epochs.events[:, 2] -1
					print(f"current labels are {current_labels}")
					labels_for_multiple_files[key] = current_labels

			return epochs_for_multiple_files, labels_for_multiple_files
