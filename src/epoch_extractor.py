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
				"event_id": {"left fist": 2, "right fist": 3},
			},
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
			 {
				"runs": [6, 10, 14],
				"mapping": {0: "rest", 1: "both fists imagined", 2: "both feet imagined"},
				"event_id": {"imagine both fists": 8, "imagine both feet": 9},
			},
		]
		self.subject_count = 109 #in dataset, its up to S109 folders with subject data respectively




	# def extract_epochs(self, data: mne.io.Raw) -> Tuple[mne.epochs.Epochs, float]:
	# 	event_id = {"T1": 1, "T2": 2}
	# 	events, _ = mne.events_from_annotations(data)

	# 	sfreq = data.info["sfreq"] #this is 160 but we could create a custom dataclass to pass this along, transform only expects an X output
	# 	epochs = mne.Epochs(data, events, event_id=event_id, tmin=-2, tmax=5.1,
	# 						baseline=None, preload=True)
		
	# 	return epochs, sfreq



	# def extract_epochs_and_labels(self, filtered_eeg_data: List[Tuple[mne.io.Raw, dict, int]]) -> Tuple[List[mne.epochs.Epochs], np.ndarray]:
	# 		'''
	# 		Input: List of tuples containing raw data, experiment info, and subject id
	# 		output: Filtered epochs (based on different timeframes and associated high/low frequencies)
	# 		'''
	# 		epochs_list = []
	# 		label_list = []
	# 		target_runs = {3, 4, 5, 6}  #define the target runs to check against

	# 		# print(filtered_eeg_data[0])
	# 		# print('INSIDE EXTRACT EPOCHS AND LABELS')
	# 		for data, experiment, subject_id in filtered_eeg_data:
				
	# 			print(f'data:{data}')
	# 			print(f'experiment:{experiment}')
	# 			print(f'subjectid:{subject_id}')

			# 	# print(f'experiment)
			# 	# print(subject_id)

			# 	events, event_id = mne.events_from_annotations(data)

			# 	print(f'current events:{events}, current eventid:{event_id}')

			# 	# Use the actual event IDs from your data
			# 	event_mapping = {desc: id for desc, id in event_id.items()}
			# 	print(f'event mapping:{event_mapping}')

			# 	# Map your desired events
			# 	desired_events = {k: event_mapping.get(k) for k in ['T1', 'T2']}
			# 	print(f'current desired events {desired_events}')

			# 	# Remove None values
			# 	desired_events = {k: v for k, v in desired_events.items() if v is not None}
			# 	print(f'current desired events without nan {desired_events}')

			# 	if not desired_events:
			# 		continue
			# 	epochs = mne.Epochs(data, events, event_id=desired_events, tmin=-2, tmax=5.1, baseline=None, preload=True)
			# 	if len(epochs.events) > 0:

			# 		print(f'{epochs.events[:, 2] - min(epochs.events[:, -1])} are the labels')
			# 		print(f'{epochs} are the epochs')

			# 		labels = epochs.events[:, -1] - min(epochs.events[:, -1])
			# 		label_list.append(labels)
			# 		epochs_list.append(epochs)
			# if epochs_list and label_list:
			# 	return epochs_list, np.concatenate(label_list)
			# else:
			# 	raise ValueError("No valid epochs or labels were extracted from the provided data.")
				
			# 	print(target_runs.intersection(experiment['runs']))
			# 	if target_runs.intersection(experiment['runs']):
			# 		#determine the mapping based on the first matched run in target_runs
			# 		event_label = 0
			# 		if 3 in experiment['runs']:
			# 			# event_mapping = {'T1': experiment['event_id']['left fist'], 'T2': experiment['event_id']['right fist']}
			# 			event_label = 1
			# 		elif 4 in experiment['runs']:
			# 			# event_mapping = {'T1': experiment['event_id']['left fist imagined'], 'T2': experiment['event_id']['right fist imagined']}
			# 			event_label = 3
			# 		elif 5 in experiment['runs']:
			# 			# event_mapping = {'T1': experiment['event_id']['both fists'], 'T2': experiment['event_id']['both feet']}
			# 			event_label = 5
			# 		elif 6 in experiment['runs']:
			# 			# event_mapping = {'T1': experiment['event_id']['imagine both fists'], 'T2': experiment['event_id']['imagine both feet']}
			# 			event_label = 7
			# 		else:
			# 			continue  # Skip if no match found (shouldn't occur with the check above)
			# 		events, _ = mne.events_from_annotations(data) #
			# 		print(events)
			# 		# print(f'{event_mapping} is event mapping')
			# 		# Ensure events are present
			# 		if len(events) > 0:
			# 			epochs = mne.Epochs(data, events, tmin=-2, tmax=5.1, baseline=None, preload=True)
			# 			# print(epochs['sfreq'])
			# 			if len(epochs.events) > 0:  #append only if epochs are created successfully
			# 				print(f'{epochs.event_id} is the epoch event id') #t0:1 (this is rest) t1:2, t2:3
			# 				print(f'{epochs.events[:, 2] - 1} is epoch events') #t1 = left fist = 0; t2 = right fist = 1
			# 				label_nr = epochs.events[:, 2] - 2 + event_label
			# 				# label_nr = [nr for nr in label_nr if nr != 0] #maybe we need to assign 0 as well
			# 				print(f'{label_nr} is going to be the label') #t1 = left fist = 0; t2 = right fist = 1

			# 				# print(f'{epochs.events[:, 2] - 2 + event_label} is going to be the label') #t1 = left fist = 0; t2 = right fist = 1
			# 				# label_list.append(epochs.events[:, 2] - 2)
			# 				# if label_nr is not 0:
			# 				# 	# print(f'assigning label {label_nr}')
			# 				# nr_numbers = [nr for nr in label_nr if nr != 0]
			# 				# for number in nr_numbers:
			# 				# 	label_list.append(number)
			# 				# print(label_list)
			# 				label_list.append(label_nr)
			# 				epochs_list.append(epochs)
			# 		else:
			# 			print(f"No valid events found for subject {subject_id} with runs {experiment['runs']}")
			# 	else:
			# 		print(f"Skipping subject {subject_id}, runs {experiment['runs']} do not match target runs")

			# if epochs_list and label_list:
			# 	return epochs_list, np.concatenate(label_list)
			# else:
			# 	raise ValueError("No valid epochs or labels were extracted from the provided data.")



			# return epochs_list, np.concatenate(label_list)






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
			epochs_for_multiple_files = {}
			labels_for_multiple_files = {}
			for key, raw_data in filtered_eeg_data.items():
				if raw_data is not None:
					current_epochs, _ = self.extract_epochs(raw_data)
					print(f'{_} is sfreq')
					epochs_for_multiple_files[key] = current_epochs #this will be epochs for now

					current_labels = current_epochs.events[:, 2] -1 #1 or 2 for t1 t2
					labels_for_multiple_files[key] = current_labels

			# for key, labels in labels_for_multiple_files:
			# 	np.concatenate(labels_for_multiple_files[key])
				# epochs_list = []
				# label_list = []
			# for data in filtered_eeg_data:
			# 	epochs, _ = self.extract_epochs(data) #could store frequency somewhere if its really needed, for this project its 160 across
			# 	epochs_list.append(epochs)
			# 	print(epochs.events)
			# 	label_list.append(epochs.events[:, 2] - 1) #1 or 2 essentially for T1, T2
			
			return epochs_for_multiple_files, labels_for_multiple_files







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