
from pathlib import Path
import mne

class Preprocessor:
	def __init__(self):
		self.data_channels = ["Fc3.", "Fcz.", "Fc4.", "C3..", "C1..", "Cz..", "C2..", "C4.."]
		self.raw_data = []


	def load_raw_data(self, data_path):
		for file_path in data_path:
			file_path = Path(file_path)  #convert each individual path to a Path object
			if not file_path.exists():
				raise FileNotFoundError(f"Data path/file '{file_path}' does not exist.")
			try:
				raw = mne.io.read_raw_edf(file_path, include=self.data_channels)
				self.raw_data.append(raw)
				# return self.raw_data
			except IOError as e:
				raise e
		return self.raw_data


#these ones should go to another part of the pipeline called FilterTransformer
	def filter_frequencies(self, raw, lo_cut, hi_cut, noise_cut):
		filtered_lo_hi = raw.copy().filter(lo_cut, hi_cut)
		filter_noise = filtered_lo_hi.notch_filter(noise_cut)
		return filter_noise



	def filter_raw_data(self):
		filtered_data = []
		for raw in self.raw_data:
			raw.load_data() #gotta close somewhere prob
			filtered_data.append(self.filter_frequencies(raw, lo_cut=0.1, hi_cut=30, noise_cut=50))
		self.raw_data = [] #empty memory, wouldnt it leak? 
		return filtered_data
