import mne
import numpy as np
from typing import List, Tuple


#we dont need the baseestimator and transformermixin here actually
class EpochExtractor:
	def __init__(self):
		self.experiments_list = [
			{
				"runs": [3,7,11],
				"mapping": {0: "rest", 1: "left fist", 2: "right fist"},
				"event_id": {"left fist": 1, "right fist": 2},
			},
			{
				"runs": [4,8,12],
				"mapping": {0: "rest", 1: "left fist imagined", 2: "right fist imagined"},
				"event_id": {"left fist imaginedt": 1, "right fist imagined": 2},
			},
			{
				"runs": [5,9,13],
				"mapping": {0: "rest", 1: "both fists", 2: "both feet"},
				"event_id": {"both fists": 1, "both feet": 2},
			},
			 {
				"runs": [6, 10, 14],
				"mapping": {0: "rest", 1: "both fists imagined", 2: "both feet imagined"},
				"event_id": {"imagine both fists": 1, "imagine both feets": 2},
			},
		]
		self.subject_count = 109 #in dataset, its up to S109 folders with subject data respectively





	def extract_epochs(self, data: mne.io.Raw) -> Tuple[mne.epochs.Epochs, float]:
		event_id = {"T1": 1, "T2": 2}
		events, _ = mne.events_from_annotations(data)

		sfreq = data.info["sfreq"] #this is 160 but we could create a custom dataclass to pass this along, transform only expects an X output
		epochs = mne.Epochs(data, events, event_id=event_id, tmin=-2, tmax=5.1,
							baseline=None, preload=True)
		
		return epochs, sfreq



	def extract_epochs_and_labels(self, filtered_eeg_data: mne.io.Raw) -> Tuple[list, np.ndarray]:
			'''
			Input: X->filtered eeg data, several eeg data thus we need a loop
			output: Filtered epochs (based on different timeframes and associated high/low frequencies)
			'''
			epochs_list = []
			label_list = []
			for data in filtered_eeg_data:
				epochs, _ = self.extract_epochs(data) #could store frequency somewhere if its really needed, for this project its 160 across
				epochs_list.append(epochs)
				print(epochs.events)
				label_list.append(epochs.events[:, 2] - 1) #1 or 2 essentially for T1, T2
			
			return epochs_list, np.concatenate(label_list)







'''
	def extract_epochs(self, data: mne.io.Raw) -> Tuple[mne.epochs.Epochs, float]:
		event_id = {"T1": 1, "T2": 2}
		events, _ = mne.events_from_annotations(data)

		sfreq = data.info["sfreq"] #this is 160 but we could create a custom dataclass to pass this along, transform only expects an X output
		epochs = mne.Epochs(data, events, event_id=event_id, tmin=-2, tmax=5.1,
							baseline=None, preload=True)
		
		return epochs, sfreq



	def extract_epochs_and_labels(self, filtered_eeg_data: mne.io.Raw) -> Tuple[list, np.ndarray]:
			
			epochs_list = []
			label_list = []
			for data in filtered_eeg_data:
				epochs, _ = self.extract_epochs(data) #could store frequency somewhere if its really needed, for this project its 160 across
				epochs_list.append(epochs)
				print(epochs.events)
				label_list.append(epochs.events[:, 2] - 1) #1 or 2 essentially for T1, T2
			
			return epochs_list, np.concatenate(label_list)

'''