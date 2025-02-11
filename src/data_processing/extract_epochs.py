import mne
import numpy as np
from typing import List, Tuple, Dict

class EpochExtractor:
	"""
	Extracts epochs and associated labels from filtered EEG data.
	"""
	def __init__(self):
		""""
		Initializes the EpochExtractor with experiment configurations.

		Returns:
			None

		1-Baseline, eyes open
		2-Baseline, eyes closed
		3-Task 1 (open and close left or right fist) #run 3-7-11-T1:left real, T2:right real
		4-Task 2 (imagine opening and closing left or right fist) run 4-8-12-T1:left imagined, T2:right imagined
		5-Task 3 (open and close both fists or both feet) run 5-9-13-T1:both fists T2:both feet real
		6-Task 4 (imagine opening and closing both fists or both feet) run 6-10-14-T1:both fists imagined T2:both feet imagined
		"""
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
		"""
		Extracts epochs from the raw EEG data and returns the epochs along with the sampling frequency.

		Args:
			data (mne.io.Raw): The raw EEG data.

		Returns:
			Tuple:
				mne.epochs.Epochs: The extracted epochs.
				float: The sampling frequency.
		"""
		
		''''
		T0 corresponds to rest -> in baseline case this means nothing happens
		T1 corresponds to onset of motion (real or imagined) ->open eye for r1, close eye for r2
		T0 for rest/“no task”
		T1 for the single baseline task (either eyes open or eyes closed in that run).
		No T2 event is present if the baseline run only involves “rest” versus one condition.
		Therefore mne.events_from_annotations call 
		naturally finds only those two codes (T0, T1). 
		Since theres effectively only one “active” event (T1) and a
		rest event (T0), we end up with a single “nonrest” label in that data.
		'''
		#only t0 = 1 in the baseline eye open
		events, event_id = mne.events_from_annotations(data)
		#print(f'{event_id} are event ids')
		sfreq = data.info["sfreq"]

		#we could make this prettier with a dict eventually
		if (len(event_id) < 3): #single event until experiment 89, then 2 events, baseline open or closed eyes
			t_min = 0
			t_max = 0.793
		else:
			event_id = {"T1": 1, "T2": 2} #this is fine in general apart from run 1,2
			t_min = -2
			t_max= 5.1
		epochs = mne.Epochs(data, events, event_id=event_id, tmin=t_min, tmax=t_max,
							baseline=None, preload=True)
		
		return epochs, sfreq



	def extract_epochs_and_labels(self, filtered_eeg_data: mne.io.Raw) -> Tuple[Dict[str, mne.epochs.Epochs], Dict[str, np.ndarray]]:
		"""
		Extracts epochs and associated labels from filtered EEG data.

		Args:
			filtered_eeg_data (mne.io.Raw): Filtered EEG data for multiple runs.

		Returns:
			Tuple:
				dict[str, mne.epochs.Epochs]: Mapping from run identifiers to extracted epochs.
				dict[str, np.ndarray]: Mapping from run identifiers to associated labels.
		"""
		epochs_for_multiple_files = {}
		labels_for_multiple_files = {}
		for key, raw_data in filtered_eeg_data.items():
			if raw_data is not None:
				current_epochs, _ = self.extract_epochs(raw_data)
				epochs_for_multiple_files[key] = current_epochs 

				current_labels = current_epochs.events[:, 2] - 1
				labels_for_multiple_files[key] = current_labels

		return epochs_for_multiple_files, labels_for_multiple_files
