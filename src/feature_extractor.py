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





	def extract_features(self, extracted_epochs_dict):
		'''
		Input: filtered and cropped list of epochs from EpochExtractor
		Output: a (x,y,z)d np array of created features based on mean, energy, power
		NO LABELS HERE, WILL DO SEPARATE
		'''
	


		sfreq = 160.0
		all_features = []

		#define analysis parameters
		analysis = {
			'mrcp': {'tmin': -2, 'tmax': 0, 'lofreq': 3, 'hifreq': 30},
			'erd': {'tmin': -2, 'tmax': 0, 'lofreq': 8, 'hifreq': 30},
			'ers': {'tmin': 4.1, 'tmax': 5.1, 'lofreq': 8, 'hifreq': 30}
		}

		
		feature_matrices = []
		for analysis_name, params in analysis.items():
			#crop epochs based on analysis parameters
			cropped_epochs = extracted_epochs_dict.copy().crop(tmin=params['tmin'], tmax=params['tmax'])
			print(f"  - Cropped epochs for analysis '{analysis_name}': {params['tmin']} to {params['tmax']} seconds.")

			# Apply filtering
			filtered_epoch = cropped_epochs.filter(
				l_freq=params['lofreq'],
				h_freq=params['hifreq'],
				method='iir'
			)
			print(f"  - Applied IIR filter: {params['lofreq']}-{params['hifreq']} Hz.")

			#create feature vectors
			feature_matrix = self.create_feature_vectors(filtered_epoch, sfreq)
			print(f"  - Extracted features for analysis '{analysis_name}': {feature_matrix.shape}")

			feature_matrices.append(feature_matrix)

		if not feature_matrices:
			raise ValueError("No features extracted for the group.")

		# Concatenate feature matrices horizontally (features combined)
		concatenated_features = np.concatenate(feature_matrices, axis=1)  # Shape: (n_epochs, total_features)
		# print(f"  - Concatenated features shape for key '{key}': {concatenated_features.shape}")

			# all_features.append(concatenated_features)

		# if not all_features:
		# 	raise ValueError("No features were extracted from the provided epochs.")

		# # Concatenate all features vertically (all epochs across keys)
		# print(f'{all_features[1].shape} is all features shape')
		# concatenated_features = np.concatenate(all_features, axis=1)
		# n_samples, n_feature_types, n_channels = concatenated_features.shape #total epochs, 9, n channels
		# reshaped_features, labels = concatenated_features.reshape(n_samples, n_feature_types*n_channels)

		# # Shape: (total_epochs, total_features)
		# # flattened_features = concatenated_features.reshape(n_samples, n_features * n_channels)

		# print(n_samples, n_features, n_channels)
		# print('are the samples features and channels')
		# # print(f"Total flattened features shape: {flattened_features.shape}")
		# print(f'{concatenated_features.shape} is the concatted shape')
		return concatenated_features



	#extracted epochs will be adict now


	# def extract_features(self, extracted_epochs_dict):
	# 	'''
	# 	Input: filtered and cropped list of epochs from EpochExtractor
	# 	Output: a (x,y,z)d np array of created features based on mean, energy, power
	# 	NO LABELS HERE, WILL DO SEPARATE
	# 	'''
	


	# 	sfreq = 160.0
	# 	all_features = []

	# 	#define analysis parameters
	# 	analysis = {
	# 		'mrcp': {'tmin': -2, 'tmax': 0, 'lofreq': 3, 'hifreq': 30},
	# 		'erd': {'tmin': -2, 'tmax': 0, 'lofreq': 8, 'hifreq': 30},
	# 		'ers': {'tmin': 4.1, 'tmax': 5.1, 'lofreq': 8, 'hifreq': 30}
	# 	}

	# 	for key, epochs in extracted_epochs_dict.items():
	# 		print(f'INSIDE KEY EPOCHS LOOP IN FEATURE EXTRACTOR, key is: {key}')
	# 		# print(f'{epochs} is epochs here')
	# 		if epochs is None:
	# 			print(f"Key '{key}' has no epochs, skipping feature extraction.")
	# 			continue

	# 		print(f"Processing features for key: '{key}' with {len(epochs)} epochs.")

	# 		feature_matrices = []
	# 		for analysis_name, params in analysis.items():
	# 			#crop epochs based on analysis parameters
	# 			cropped_epochs = epochs.copy().crop(tmin=params['tmin'], tmax=params['tmax'])
	# 			print(f"  - Cropped epochs for analysis '{analysis_name}': {params['tmin']} to {params['tmax']} seconds.")

	# 			# Apply filtering
	# 			filtered_epoch = cropped_epochs.filter(
	# 				l_freq=params['lofreq'],
	# 				h_freq=params['hifreq'],
	# 				method='iir'
	# 			)
	# 			print(f"  - Applied IIR filter: {params['lofreq']}-{params['hifreq']} Hz.")

	# 			#create feature vectors
	# 			feature_matrix = self.create_feature_vectors(filtered_epoch, sfreq)
	# 			print(f"  - Extracted features for analysis '{analysis_name}': {feature_matrix.shape}")

	# 			feature_matrices.append(feature_matrix)

	# 		if not feature_matrices:
	# 			print(f"No features extracted for key '{key}'.")
	# 			continue

	# 		# Concatenate feature matrices horizontally (features combined)
	# 		concatenated_features = np.concatenate(feature_matrices, axis=1)  # Shape: (n_epochs, total_features)
	# 		print(f"  - Concatenated features shape for key '{key}': {concatenated_features.shape}")

	# 		all_features.append(concatenated_features)

	# 	if not all_features:
	# 		raise ValueError("No features were extracted from the provided epochs.")

	# 	# Concatenate all features vertically (all epochs across keys)
	# 	print(f'{all_features[1].shape} is all features shape')
	# 	concatenated_features = np.concatenate(all_features, axis=0)
	# 	n_samples, n_features, n_channels = concatenated_features.shape #total epochs, 9, n channels
  	# 	# Shape: (total_epochs, total_features)
	# 	# flattened_features = concatenated_features.reshape(n_samples, n_features * n_channels)

	# 	print(n_samples, n_features, n_channels)
	# 	print('are the samples features and channels')
	# 	# print(f"Total flattened features shape: {flattened_features.shape}")
	# 	print(f'{concatenated_features.shape} is the concatted shape')
	# 	return concatenated_features




