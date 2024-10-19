
from pathlib import Path
from typing import List, Tuple
import mne



class Preprocessor:
	def __init__(self):
		self.data_channels = ["Fc3.", "Fcz.", "Fc4.", "C3..", "C1..", "Cz..", "C2..", "C4.."]
		self.raw_data = []



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
