
from pathlib import Path
from typing import List, Tuple
import mne
import os


'''
TODO: 
-MAP THIS FOR ALL RUNS NOT JUST 3,7,11 TO HAVE THE REST OF THE EXPERIMENTS
-ADD MORE CHANNELS TO CLASSIFY ALL OTHER MAPPING OTHER THAN PURE LEFT FIST AND RIGHT FIST


'''

class Preprocessor:
	def __init__(self):
		self.data_channels = ["Fc3.", "Fcz.", "Fc4.", "C3..", "C1..", "Cz..", "C2..", "C4.."]
		self.raw_data = []
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



	def get_experiment_by_run(self, run_nr):
		for experiment in self.experiments_list:
			if run_nr in experiment['runs']:
				return experiment
		return None



	def load_raw_data(self, data_path: List[str]):
		data_with_experiments = []
		for file_path in data_path:
			file_path = Path(file_path)  #convert each individual path to a Path object
			if not file_path.exists():
				raise FileNotFoundError(f"Data path/file '{file_path}' does not exist.")
			try:
				print(f'{file_path} is the filepath')

				filename = os.path.basename(file_path)
				filename_without_ext = os.path.splitext(filename)[0]
				subject_part = filename_without_ext[:4]  #'S***'->see in data folder
				run_part = filename_without_ext[4:]

				# print(subject_part)
				# print(run_part)
				#in case someone w upload different data, check for names
				print('1')
				if subject_part[0] != 'S' or run_part[0] != 'R':
					raise ValueError(f"Invalid filename format: '{filename}'")

				print('2')
				run_nr = int(run_part[1:]) #Snrnrnr
				subject_id = int(subject_part[1:]) #already extracted above from 4:

				#get the experiment based on the run number
				experiment = self.get_experiment_by_run(run_nr)
				if not experiment:
					print(f"Run {run_nr} does not correspond to any experiment, skipping it.")
					continue
				print('3')
				raw = mne.io.read_raw_edf(file_path, include=self.data_channels)
	
				available_channels = raw.ch_names
				if not all(channel in available_channels for channel in self.data_channels):
					raise ValueError(f"File '{file_path}' does not contain the expected channels: {self.data_channels}")
				
				self.raw_data.append((raw, experiment, subject_id))
				# data_with_experiments.append((raw, experiment, subject_id))
				print('4')


			except PermissionError:
				raise PermissionError(f"Permission denied: Unable to access '{file_path}'. Check file permissions.")
			except IOError as e:
				raise IOError(f"Error reading file '{file_path}': {e}")
			except ValueError as ve:
				raise ValueError(f"Invalid EDF file: {ve}")
		print('5')
		return self.raw_data
		# return data_with_experiments



	def filter_frequencies(self, raw: mne.io.Raw, lo_cut: float, hi_cut: float, noise_cut: float):
		filtered_lo_hi = raw.copy().filter(lo_cut, hi_cut)
		filter_noise = filtered_lo_hi.notch_filter(noise_cut)

		return filter_noise



	def filter_raw_data(self) -> List[mne.io.Raw]:
		filtered_data = []
		for raw in self.raw_data:
			raw[0].load_data()
			print(f'data: {raw[0]}, experiment:{raw[1]}, subjectid: {raw[2]}')

			filtered_data.append(self.filter_frequencies(raw[0], lo_cut=0.1, hi_cut=30, noise_cut=50))
		self.raw_data = [] #empty memory, wouldnt it leak?

		return filtered_data






'''
	def load_raw_data(self, data_path: List[str]) -> List[mne.io.Raw]:
		for file_path in data_path:
			file_path = Path(file_path)  #convert each individual path to a Path object
			if not file_path.exists():
				raise FileNotFoundError(f"Data path/file '{file_path}' does not exist.")
			try:
				raw = mne.io.read_raw_edf(file_path, include=self.data_channels)
				
				available_channels = raw.ch_names
				if not all(channel in available_channels for channel in self.data_channels):
					raise ValueError(f"File '{file_path}' does not contain the expected channels: {self.data_channels}")
				
				self.raw_data.append(raw)
			except PermissionError:
				raise PermissionError(f"Permission denied: Unable to access '{file_path}'. Check file permissions.")
			except IOError as e:
				raise IOError(f"Error reading file '{file_path}': {e}")
			except ValueError as ve:
				raise ValueError(f"Invalid EDF file: {ve}")

		return self.raw_data



	def filter_frequencies(self, raw: mne.io.Raw, lo_cut: float, hi_cut: float, noise_cut: float):
		filtered_lo_hi = raw.copy().filter(lo_cut, hi_cut)
		filter_noise = filtered_lo_hi.notch_filter(noise_cut)

		return filter_noise



	def filter_raw_data(self) -> List[mne.io.Raw]:
		filtered_data = []
		for raw in self.raw_data:
			raw.load_data() #gotta close somewhere prob
			filtered_data.append(self.filter_frequencies(raw, lo_cut=0.1, hi_cut=30, noise_cut=50))
		self.raw_data = [] #empty memory, wouldnt it leak?

		return filtered_data

'''