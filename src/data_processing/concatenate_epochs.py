import numpy as np

class EpochConcatenator:
	"""
	Handles the concatenation of epochs, along with event times and optional mean calculations.
	"""
	def __init__(self, epoch_duration=7.1):
		"""
		Initializes the EpochConcatenator.

		Args:
			epoch_duration (float): Duration of each epoch in seconds. Default is 7.1.

		Returns:
			None
		"""
		self.epoch_duration = epoch_duration

	def concatenate_all_epochs(self, epochs_chunk, labels_chunk, predictions_chunk):
		"""
		Concatenates epochs by computing the mean over channels/time for each epoch and computes event times.

		Args:
			epochs_chunk (List): List of epochs (each epoch is an array).
			labels_chunk (List): List of labels corresponding to each epoch.
			predictions_chunk (List): List of predictions corresponding to each epoch.

		Returns:
			Tuple:
				np.ndarray: Concatenated array of mean values from each epoch.
				list: List of event times corresponding to each epoch.
		"""
		n_epochs = len(epochs_chunk)
		concatenated_data = []
		event_times = []
		concatenated_labels = []
		concatenated_predictions = []

		for idx, (epoch, label, pred) in enumerate(zip(epochs_chunk, labels_chunk, predictions_chunk)):
			mean_data = epoch.mean(axis=0)
			concatenated_data.append(mean_data)
			event_times.append(idx * self.epoch_duration)
			concatenated_labels.append(label)
			concatenated_predictions.append(pred)

		concatenated_data = np.concatenate(concatenated_data)
		return concatenated_data, event_times
