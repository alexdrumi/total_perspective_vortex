import mne
import numpy as np
from typing import List, Tuple


#we dont need the baseestimator and transformermixin here actually
class EpochExtractor:
	def __init__(self):
		pass



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
				label_list.append(epochs.events[:, 2] - 1) #1 or 2 essentially for T1, T2
			
			return epochs_list, np.concatenate(label_list)
