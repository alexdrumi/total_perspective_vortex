import mne
import numpy as np
from typing import List, Tuple

class EpochExtractor:
	def __init__(self):
		#3-Task 1 (open and close left or right fist) #run 3-7-11-T1:left real, T2:right real
		#4-Task 2 (imagine opening and closing left or right fist) run 4-8-12-T1:left imagined, T2:right imagined
		#5-Task 3 (open and close both fists or both feet) run 5-9-13-T1:both fists T2:both feet real
		#6-Task 4 (imagine opening and closing both fists or both feet) run 6-10-14-T1:both fists imagined T2:both feet imagined
		self.experiments_list = [
			{
				"runs": [1],
				"mapping": {0: "rest", 1: "open", 2: "closed"},
				"event_id": {"open eyes": 2, "closed eyes": 3},
			},
			{
				"runs": [2],
				"mapping": {0: "rest", 1: "closed", 2: "open"},
				"event_id": {"closed eyes": 2, "open eyes": 3},
			},
			{
				"runs": [3,7,11],
				"mapping": {0: "rest", 1: "left fist", 2: "right fist"},
				"event_id": {"left fist": 2, "right fist": 3},
			},
			
			{
				"runs": [4,8,12],
				"mapping": {0: "rest", 1: "left imagined", 2: "right imagined"},
				"event_id": {"left fist imagined": 2, "right fist imagined": 3},
			},
			{
				"runs": [5,9,13],
				"mapping": {0: "rest", 1: "both fists", 2: "both feet"},
				"event_id": {"both fists": 2, "both feet": 3},
			},
			{
				"runs": [6, 10, 14],
				"mapping": {0: "rest", 1: "both fists imagined", 2: "both feet imagined"},
				"event_id": {"imagine both fists": 2, "imagine both feet": 3},
			},
		]



	def extract_epochs(self, data: mne.io.Raw) -> Tuple[mne.epochs.Epochs, float]:
		# event_id = {"T0": 1, "T1": 2, "T2": 3} #here include T0
		# event_id = {"T1": 1, "T2": 2} #this is fine in general
		#only t0 = 1 in the baseline eye open
		events, event_id = mne.events_from_annotations(data)
		print(f'{event_id} are event ids')
		sfreq = data.info["sfreq"]
		# event_times = events[:, 0] / sfreq
		# print(event_times)
		# tmin, tmax = 0

		#WE COULD MAKE THIS PRETTIER with a dict or so!!!!!!!!
		if (len(event_id) == 1): #single event, baseline open or closed eyes
			t_min = 0
			t_max = 0.793
		else:
			event_id = {"T1": 1, "T2": 2} #this is fine in general
			t_min = -2
			t_max= 5.1
		print(f'{t_min}: tmin, {t_max}: tmax')
		epochs = mne.Epochs(data, events, event_id=event_id, tmin=t_min, tmax=t_max,
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

					current_labels = current_epochs.events[:, 2] - 1
					print(f"current labels are {current_labels}")
					labels_for_multiple_files[key] = current_labels

			return epochs_for_multiple_files, labels_for_multiple_files
