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
		
		for epoch in epochs: #epoch is #np.ndarray
			mean = np.mean(epoch, axis=0)
			activation = epoch - mean #np.ndarray
			current_feature_vec = self.calculate_mean_power_energy(activation, epoch, sfreq) #np.ndarray
			feature_matrix.append(current_feature_vec) #list

		return np.array(feature_matrix)



	def extract_features(self, extracted_epochs: list) -> np.ndarray:
		'''
		Input: filtered and cropped list of epochs from EpochExtractor
		Output: a (x,y,z)d np array of created features based on mean, energy, power
		NO LABELS HERE, WILL DO SEPARATE
		'''
	
		sfreq = 160.0 #this we could get from mne but is consistent across all data here.
		all_features = []
		for filtered_epochs in extracted_epochs:
			analysis = {
				'mrcp': {'tmin': -2, 'tmax': 0, 'lofreq': 3, 'hifreq': 30},
				'erd': {'tmin': -2, 'tmax': 0, 'lofreq': 8, 'hifreq': 30},
				'ers': {'tmin': 4.1, 'tmax': 5.1, 'lofreq': 8, 'hifreq': 30}
			}

			feature_matrices = []
			for analysis_name, parameters in analysis.items():
				#you need the copy here
				cropped_epochs = filtered_epochs.copy().crop(tmin=parameters['tmin'], tmax=parameters['tmax'])
				filtered_epoch = cropped_epochs.filter(h_freq=parameters['hifreq'],
														l_freq=parameters['lofreq'],
														method='iir')
				
				feature_matrix =  self.create_feature_vectors(filtered_epoch, sfreq)
				feature_matrices.append(feature_matrix)
				
				#check samples for consistent counts
				sample_counts = [fm.shape[0] for fm in feature_matrices]
				if not all(count == sample_counts[0] for count in sample_counts):
					raise ValueError("Inconsistent number of samples across analyses. Ensure all have the same number of epochs.")
				#we could have used maybe multiprocessing with pool but can be buggy and this works just fine for now
			all_features.append(np.concatenate(feature_matrices, axis=1))

		concatenated_features = np.concatenate(all_features, axis=0) #this is now (59 epoch list, 21 epochs inside, 9*8 feature combinations) thus we need to concat them 
		return concatenated_features





