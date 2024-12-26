
from pathlib import Path
from typing import List, Tuple
import mne
import os
import yaml

'''
TODO: 
-MAP THIS FOR ALL RUNS NOT JUST 3,7,11 TO HAVE THE REST OF THE EXPERIMENTS
-ADD MORE CHANNELS TO CLASSIFY ALL OTHER MAPPING OTHER THAN PURE LEFT FIST AND RIGHT FIST


'''

class Preprocessor:
	def __init__(self):
		self.data_channels =  ["Fc1.","Fc2.", "Fc3.", "Fc4.", "C3..", "C1..", "Cz..", "C2..", "C4.."]
		# self.data_channels =  ["Fc1.","Fc2.", "Fc3.", "Fcz.", "Fc4.", "C3..", "C1..", "Cz..", "C2..", "C4.."]
		self.raw_data = []
		self.experiments_list = [
			{
				"runs": [3,7,11],
				"mapping": {0: "rest", 1: "left fist", 2: "right fist"},
				"event_id": {"left fist": 2, "right fist": 3},
			},
			{
				"runs": [4,8,12],
				"mapping": {0: "rest", 3: "left fist imagined", 4: "right fist imagined"},
				"event_id": {"left fist imagined": 2, "right fist imagined": 3},
			},
			{
				"runs": [5,9,13],
				"mapping": {0: "rest", 5: "both fists", 6: "both feet"},
				"event_id": {"both fists": 2, "both feet": 3},
			},
			 {
				"runs": [6, 10, 14],
				"mapping": {0: "rest", 7: "both fists imagined", 8: "both feet imagined"},
				"event_id": {"imagine both fists": 2, "imagine both feet": 3},
			},
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
		]



	def get_experiment_by_run(self, run_nr):
		for experiment in self.experiments_list:
			if run_nr in experiment['runs']:
				return experiment
		return None



	def load_raw_data(self, data_path: List[str]):
		loaded_raw_data = []
		concatted_raw_data_dict = {
			'1': [],
			'2': [],
			'3': [],
			'4': [],
			'5': [],
			'6': []
		}

		with open(data_path, 'r') as f:
			yaml_data_path = yaml.safe_load(f)

		for file_path in yaml_data_path['data_paths']:
			file_path = Path(file_path) 
			print(file_path)
			if not file_path.exists():
				raise FileNotFoundError(f"Data path/file '{file_path}' does not exist.")
			try:
				filename = os.path.basename(file_path)
				filename_without_ext = os.path.splitext(filename)[0]
				subject_part = filename_without_ext[:4]
				run_part = filename_without_ext[4:]

				if subject_part[0] != 'S' or run_part[0] != 'R':
					raise ValueError(f"Invalid filename format: '{filename}'")

				run_nr = int(run_part[-2:])
				subject_id = int(subject_part[1:])

				experiment = self.get_experiment_by_run(run_nr)
				if not experiment:
					print(f"Run {run_nr} does not correspond to any experiment, skipping it.")
					continue

				raw = mne.io.read_raw_edf(file_path, include=self.data_channels)

				if (raw.info['sfreq'] != 160.0): #all the raw frequencies should be 160.0 but there are some faulty ones, those we ignore
					continue 
				if run_nr == 1:
					concatted_raw_data_dict['1'].append(raw)
				if run_nr == 2:
					concatted_raw_data_dict['2'].append(raw)
				if run_nr in [3, 7, 11]:
					concatted_raw_data_dict['3'].append(raw)
				elif run_nr in [4, 8, 12]:
					concatted_raw_data_dict['4'].append(raw)
				elif run_nr in [5, 9, 13]:
					concatted_raw_data_dict['5'].append(raw)
				elif run_nr in [6, 10, 14]:
					concatted_raw_data_dict['6'].append(raw)

				available_channels = raw.ch_names
				if not all(channel in available_channels for channel in self.data_channels):
					raise ValueError(f"File '{file_path}' does not contain the expected channels: {self.data_channels}")

				loaded_raw_data.append((raw, experiment, subject_id))

			except PermissionError:
				raise PermissionError(f"Permission denied: Unable to access '{file_path}'. Check file permissions.")
			except IOError as e:
				raise IOError(f"Error reading file '{file_path}': {e}")
			except ValueError as ve:
				raise ValueError(f"Invalid EDF file: {ve}")

		for key, raw_list in concatted_raw_data_dict.items():
			if raw_list:
				print(f'{concatted_raw_data_dict[key]} is current raw data dict with key {key}')
				concatted_raw_data_dict[key] = mne.concatenate_raws(raw_list)
			else:
				concatted_raw_data_dict[key] = None

		return concatted_raw_data_dict

# 


	def filter_frequencies(self, raw: mne.io.Raw, lo_cut: float, hi_cut: float, noise_cut: float):
		filtered_lo_hi = raw.copy().filter(lo_cut, hi_cut)
		filter_noise = filtered_lo_hi.notch_filter(noise_cut)

		return filter_noise



	def filter_raw_data(self, loaded_raw_data) -> List[mne.io.Raw]:
		filtered_data_dict = {}
		for key, raw in loaded_raw_data.items():
			if raw is not None:  #ensure there is data to filter
				raw.load_data()  #load for filtering
				current_filtered = self.filter_frequencies(raw, lo_cut=0.1, hi_cut=30, noise_cut=50)
				filtered_data_dict[key] = raw
			else:
				print(f'KEY {key} IS EMPTY')
				filtered_data_dict[key] = None #handle empty keys if needed

		return filtered_data_dict
	