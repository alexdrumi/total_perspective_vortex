import numpy as np
import mne
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureExtractor(BaseEstimator, TransformerMixin):
	# def __init__(self, data_to_extract_from):
	def __init__(self):
		self._labels = []
		#better not creating a lifetime container here, threading might be tricky, keep it to local variables
		pass

	#feature extractor should have this for sure
	def calculate_mean_power_energy(self, activation, epoch, sfreq):
		mean_act = np.mean(activation, axis=1)
		energy = np.sum(activation ** 2, axis=1)
		power = energy / (len(epoch) * sfreq)
		
		# current_feature_vec = np.array([mean_act, energy, power])
		current_feature_vec = np.zeros((3,8))
		current_feature_vec[0] = mean_act
		current_feature_vec[1] = energy
		current_feature_vec[2] = power
		return current_feature_vec


	#this should be the transform for this class
	def create_feature_vectors(self, epochs, sfreq, compute_y=False):
		y = [] if compute_y else None #we only need this onece, if its ers, since event types are the same across epochs
		feature_matrix = []
		for idx, epoch in enumerate(epochs):
			#epoch is already filtered by now
			mean = np.mean(epoch, axis=0)
			activation = epoch - mean

			current_feature_vec = self.calculate_mean_power_energy(activation, epoch, sfreq)
			feature_matrix.append(current_feature_vec)

			if compute_y == True:
				event_type = epochs.events[idx][2] - 1  #[18368(time)     0(?)     1(event_type)]
				y.append(event_type)
			
		feature_matrix = np.array(feature_matrix)
		y = np.array(y) if compute_y else None

		return feature_matrix, y
	


	def transform(self, X):
		'''
		Input: filtered and cropped list of epochs from EpochExtractor
		Output: a (x,y,z)d np array of created features based on mean, energy, power
		NO LABELS HERE, WILL DO SEPARATE
		'''
		#this is now SEPARATE AND PROB WE DONT ASSIGN LABELS
		sfreq = 160.0
		
		all_features = []
		all_labels = []
		for idx, filtered_epochs in enumerate(X): #X are the extracted epochs
			# print(f'{idx} IS INDEX, {filtered_epochs} IS EPOCHS')
			# epochs, sfreq = self.extract_epochs(filtered_data)
			# epochs_data = epochs.get_data() #https://mne.tools/1.7/generated/mne.Epochs.html#mne.Epochs.get_data (data array of shape (n_epochs, n_channels, n_times))
			#this would return a numpy array, for later maybe not now, this will be useful when we do crop manually

			analysis = {
				'mrcp': {'tmin': -2, 'tmax': 0, 'lofreq': 3, 'hifreq': 30},
				'erd': {'tmin': -2, 'tmax': 0, 'lofreq': 8, 'hifreq': 30},
				'ers': {'tmin': 4.1, 'tmax': 5.1, 'lofreq': 8, 'hifreq': 30}
			}

			feature_matrices = []
			labels = []
			for analysis_name, parameters in analysis.items():
				# start_time = (parameters['tmin'] - epochs_data.tmin) * sfreq) #vectorizing is gonna vbe faster than copying objects
				# we gotta use numpy to make vector operations for now this is ok.
				cropped_epochs = filtered_epochs.copy().crop(tmin=parameters['tmin'], tmax=parameters['tmax'])
				filtered_epoch = cropped_epochs.filter(h_freq=parameters['hifreq'],
														l_freq=parameters['lofreq'],
														method='iir')
				compute_y = (analysis_name == 'ers')
				

				# if labels is None:
				# 	raise ValueError("Labels were not assigned. Ensure that at least one analysis computes labels.")
				
				feature_matrix, y = self.create_feature_vectors(filtered_epoch, 160.0, compute_y)
				feature_matrices.append(feature_matrix)
				if (compute_y == True):
					labels = y

				if labels is None:
					raise ValueError("Labels were not assigned. Ensure that at least one analysis computes labels.")

				#check samples for consistent counts
				sample_counts = [fm.shape[0] for fm in feature_matrices]
				if not all(count == sample_counts[0] for count in sample_counts):
					raise ValueError("Inconsistent number of samples across analyses. Ensure all have the same number of epochs.")
			# print(f'{feature_matrices} ARE THE FEATURE MATRICES INSIDE THE EXTRACTOR')
			# print(feature_matrices.shape)

			all_features.append(np.concatenate(feature_matrices, axis=1))
			all_labels.append(np.array(labels))
	
		# res = np.concatenate(all_features, axis=1)
		self._labels = np.concatenate(all_labels, axis=0) #would solve it more elegantly but transformer only returns one data, X
		ret = np.concatenate(all_features, axis=0) #this is now (59 epoch list, 21 epochs inside, 9*8 feature combinations) thus we need to concat them 
		
		print(ret.shape)
		print(self._labels.shape)

		# print(f'{ret} are all the features')
		return ret


#PROBLEM IS POSSIBLY THAT ITS ONLY A 2D ARRAY, WE NEED 3 D ARRAYS



		# for filtered_epoch in X:
		# 	feature_matrix  = self.feature_extractor.create_feature_vectors(X, sfreq)
		# 	feature_matrices.append(feature_matrix)
		
		# # if compute_y == True:
		# # 	labels = y

		# # if labels is None:
		# # 	raise ValueError("Labels were not assigned. Ensure that at least one analysis computes labels.")
		
		# sample_counts = [fm.shape[0] for fm in feature_matrices]
		# if not all(count == sample_counts[0] for count in sample_counts):
		# 	raise ValueError("Inconsistent number of samples across analyses. Ensure all have the same number of epochs.")

		# res = np.concatenate(feature_matrices, axis=1)
		# return res




def calculate_mean_power_energy(activation, epoch, sfreq):
		mean_act = np.mean(activation, axis=1)
		energy = np.sum(activation ** 2, axis=1)
		power = energy / (len(epoch) * sfreq)
		
		# current_feature_vec = np.array([mean_act, energy, power])
		current_feature_vec = np.zeros((3,8))
		current_feature_vec[0] = mean_act
		current_feature_vec[1] = energy
		current_feature_vec[2] = power
		return current_feature_vec


#this should be the transform for this class
def create_feature_vectors(epochs, sfreq, compute_y=False):
	y = [] if compute_y else None #we only need this onece, if its ers, since event types are the same across epochs
	feature_matrix = []
	for idx, epoch in enumerate(epochs):
		#epoch is already filtered by now
		mean = np.mean(epoch, axis=0)
		activation = epoch - mean

		current_feature_vec = calculate_mean_power_energy(activation, epoch, sfreq)
		feature_matrix.append(current_feature_vec)

		if compute_y == True:
			event_type = epochs.events[idx][2] - 1  #[18368(time)     0(?)     1(event_type)]
			y.append(event_type)
		
	feature_matrix = np.array(feature_matrix)
	y = np.array(y) if compute_y else None

	return feature_matrix, y



def feature_extractor(X):
	'''
	Input: filtered and cropped list of epochs from EpochExtractor
	Output: a (x,y,z)d np array of created features based on mean, energy, power
	NO LABELS HERE, WILL DO SEPARATE
	'''
	#this is now SEPARATE AND PROB WE DONT ASSIGN LABELS
	sfreq = 160.0
	
	all_features = []
	all_labels = []
	for idx, filtered_epochs in enumerate(X): #X are the extracted epochs
		# print(f'{idx} IS INDEX, {filtered_epochs} IS EPOCHS')
		# epochs, sfreq = self.extract_epochs(filtered_data)
		# epochs_data = epochs.get_data() #https://mne.tools/1.7/generated/mne.Epochs.html#mne.Epochs.get_data (data array of shape (n_epochs, n_channels, n_times))
		#this would return a numpy array, for later maybe not now, this will be useful when we do crop manually

		analysis = {
			'mrcp': {'tmin': -2, 'tmax': 0, 'lofreq': 3, 'hifreq': 30},
			'erd': {'tmin': -2, 'tmax': 0, 'lofreq': 8, 'hifreq': 30},
			'ers': {'tmin': 4.1, 'tmax': 5.1, 'lofreq': 8, 'hifreq': 30}
		}

		feature_matrices = []
		labels = []
		for analysis_name, parameters in analysis.items():
			# start_time = (parameters['tmin'] - epochs_data.tmin) * sfreq) #vectorizing is gonna vbe faster than copying objects
			# we gotta use numpy to make vector operations for now this is ok.
			cropped_epochs = filtered_epochs.copy().crop(tmin=parameters['tmin'], tmax=parameters['tmax'])
			filtered_epoch = cropped_epochs.filter(h_freq=parameters['hifreq'],
													l_freq=parameters['lofreq'],
													method='iir')
			compute_y = (analysis_name == 'ers')
			

			# if labels is None:
			# 	raise ValueError("Labels were not assigned. Ensure that at least one analysis computes labels.")
			
			feature_matrix, y = create_feature_vectors(filtered_epoch, 160.0, compute_y)
			feature_matrices.append(feature_matrix)
			if (compute_y == True):
				labels = y

			if labels is None:
				raise ValueError("Labels were not assigned. Ensure that at least one analysis computes labels.")

			#check samples for consistent counts
			sample_counts = [fm.shape[0] for fm in feature_matrices]
			if not all(count == sample_counts[0] for count in sample_counts):
				raise ValueError("Inconsistent number of samples across analyses. Ensure all have the same number of epochs.")
		# print(f'{feature_matrices} ARE THE FEATURE MATRICES INSIDE THE EXTRACTOR')
		# print(feature_matrices.shape)

		all_features.append(np.concatenate(feature_matrices, axis=1))
		all_labels.append(np.array(labels))

	# res = np.concatenate(all_features, axis=1)
	new_labels = np.concatenate(all_labels, axis=0) #would solve it more elegantly but transformer only returns one data, X
	ret = np.concatenate(all_features, axis=0) #this is now (59 epoch list, 21 epochs inside, 9*8 feature combinations) thus we need to concat them 
	
	# print(ret.shape)
	# print(self._labels.shape)

	# print(f'{ret} are all the features')
	return ret