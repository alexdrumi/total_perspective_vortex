import numpy as np
import mne
from typing import List, Tuple



class FeatureExtractor():
	def __init__(self) -> None:
		"""
		The object extracts features of the given data based on the following paper:
			- doi:10.1016/s1388-2457(01)00661-7
			- https://arxiv.org/pdf/1312.2877.pdf
		"""
		pass



	def calculate_mean_power_energy(self, activation: np.ndarray, epoch: np.ndarray, sfreq: float) -> np.ndarray:
		"""
		Calculate a feature based on mean activity, energy and power of the received signal.

		Args:
			activation (np.ndarray): centered activation array
			epoch: Epoch (action) array.
			sfreq: Sampling frequency of the signal in Hz.

		Returns:
			np.ndarray: Array (3, n_channels) of the mean, energy and power together.
		"""	
		mean_act = np.mean(activation, axis=1)
		energy = np.sum(activation ** 2, axis=1)
		power = energy / (len(epoch) * sfreq)
		current_feature_vec = np.vstack((mean_act, energy, power))

		return current_feature_vec



	def create_feature_vectors(self, epochs: mne.epochs.Epochs, sfreq: float) -> np.ndarray:
		"""
		Generate feature vectors for all epochs.

		Args:
			epochs (mne.epochs.Epochs): Input epochs.
			sfreq (float): Sampling frequency of the signal in Hz.

		Returns:
			np.ndarray: Array of shape (n_epochs, 3, n_channels) containing feature vectors.
		"""
		feature_matrix = []
		
		for epoch in epochs:
			mean = np.mean(epoch, axis=0)
			activation = epoch - mean
			current_feature_vec = self.calculate_mean_power_energy(activation, epoch, sfreq)
			feature_matrix.append(current_feature_vec) #list

		return np.array(feature_matrix)



	def extract_features(self, extracted_epochs_dict: mne.epochs.Epochs, run_type: str) -> np.ndarray:
		"""
		Extract features from filtered and cropped epochs.

		Args:
			extracted_epochs_dict (mne.epochs.Epochs): Filtered and cropped epochs.
			run_type (str): Type of the run, e.g., 'baseline', 'mrcp', 'erd', or 'ers'.

		Returns:
			np.ndarray: Concatenated feature matrix of shape (n_epochs, n_features).
		"""
		sfreq = 160.0 #this keeps some of them out from the training but otherwise we would have to alter the structure of the pipeline, for now this is ok
		analysis = {
			'mrcp': {'tmin': -2, 'tmax': 0, 'lofreq': 3, 'hifreq': 30},
			'erd': {'tmin': -2, 'tmax': 0, 'lofreq': 8, 'hifreq': 30},
			'ers': {'tmin': 4.1, 'tmax': 5.1, 'lofreq': 8, 'hifreq': 30}
		}

		if run_type == 'baseline':  #if run 1 or 2. tmin and tmax are derived from events; T0 mostly in the epoch extractor
			for key in analysis:
				analysis[key]['tmin'] = 0
				analysis[key]['tmax'] = 0.793

		feature_matrices = []
		for analysis_name, params in analysis.items():
			print(f'{params['tmin']} tmin, {params['tmax']} is tmax\n\n')
			cropped_epochs = extracted_epochs_dict.copy().crop(tmin=params['tmin'], tmax=params['tmax'])
			print(f"  - Cropped epochs for analysis '{analysis_name}': {params['tmin']} to {params['tmax']} seconds.")

			filtered_epoch = cropped_epochs.filter(
				l_freq=params['lofreq'],
				h_freq=params['hifreq'],
				method='iir'
			)

			print(f"  - Applied IIR filter: {params['lofreq']}-{params['hifreq']} Hz.")
			feature_matrix = self.create_feature_vectors(filtered_epoch, sfreq)
			print(f"  - Extracted features for analysis '{analysis_name}': {feature_matrix.shape}")
			feature_matrices.append(feature_matrix)

		if not feature_matrices:
			raise ValueError("No features extracted for the group.")

		concatenated_features = np.concatenate(feature_matrices, axis=1)
		return concatenated_features

