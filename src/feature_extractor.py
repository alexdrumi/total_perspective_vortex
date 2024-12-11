import numpy as np
import mne
from typing import List, Tuple



class FeatureExtractor():
	def __init__(self):
		pass



	def calculate_mean_power_energy(self, activation: np.ndarray, epoch: np.ndarray, sfreq: float) -> np.ndarray:
			mean_act = np.mean(activation, axis=1)
			energy = np.sum(activation ** 2, axis=1)
			power = energy / (len(epoch) * sfreq)

			current_feature_vec = np.vstack((mean_act, energy, power))

			return current_feature_vec



	def create_feature_vectors(self, epochs: mne.epochs.Epochs, sfreq: float) -> np.ndarray:
		feature_matrix = []
		
		for epoch in epochs:
			mean = np.mean(epoch, axis=0)
			activation = epoch - mean
			current_feature_vec = self.calculate_mean_power_energy(activation, epoch, sfreq)
			feature_matrix.append(current_feature_vec) #list

		return np.array(feature_matrix)





	def extract_features(self, extracted_epochs_dict):
		'''
		Input: filtered and cropped list of epochs from EpochExtractor
		Output: a (x,y,z)d np array of created features based on mean, energy, power
		NO LABELS HERE, WILL DO SEPARATE
		'''
		sfreq = 160.0 #this keeps some of them out from the training but otherwise we would have to alter the structure of the pipeline, for now this is ok
		all_features = []

		analysis = {
			'mrcp': {'tmin': -2, 'tmax': 0, 'lofreq': 3, 'hifreq': 30},
			'erd': {'tmin': -2, 'tmax': 0, 'lofreq': 8, 'hifreq': 30},
			'ers': {'tmin': 4.1, 'tmax': 5.1, 'lofreq': 8, 'hifreq': 30}
		}

		
		feature_matrices = []
		for analysis_name, params in analysis.items():
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

