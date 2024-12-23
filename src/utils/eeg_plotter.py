import numpy as np
import mne
import sys
import matplotlib.pyplot as plt
from src.data_extraction.epoch_concatenator import EpochConcatenator


class EEGPlotter:
	"""
	Plots EEG epochs in a continuous form, annotating predictions vs. ground truth.
	"""
	def __init__(self, epoch_duration=7.1):
		self.epoch_duration = epoch_duration
		self.concatenator = EpochConcatenator(epoch_duration=self.epoch_duration)

	def plot_eeg_epochs_chunk(self, current_batch_idx, epochs_chunk, labels_chunk, predictions_chunk, label_names, ax, alpha=0.3, linewidth=0.7):
		"""
		Plots a chunk of epochs on a single continuous plot, with annotations for correctness.
		"""
		n_epochs = len(epochs_chunk)
		total_duration = n_epochs * self.epoch_duration

		# Concatenate and get event times
		concatenated_data, event_times = self.concatenator.concatenate_all_epochs(
			epochs_chunk, labels_chunk, predictions_chunk
		)

		times = np.linspace(0, total_duration, concatenated_data.shape[0])

		ax.clear()
		ax.plot(times, concatenated_data, label='EEG Signal', alpha=alpha, linewidth=linewidth)

		# Vertical lines for each epoch boundary
		for event_time in event_times:
			ax.axvline(x=event_time, color='gray', linestyle='--', linewidth=0.5)

		y_min, y_max = ax.get_ylim()
		annotation_y = y_max - 0.05 * (y_max - y_min)

		# Annotate each epoch boundary with predictions vs. truth
		for idx, event_time in enumerate(event_times):
			true_label = label_names[1] if labels_chunk[idx] == 0 else label_names[2]
			predicted_label = label_names[1] if predictions_chunk[idx] == 0 else label_names[2]
			is_correct = (labels_chunk[idx] == predictions_chunk[idx])
			annotation = f"{true_label}\n----------------\n{predicted_label}"
			color = 'green' if is_correct else 'red'
			ax.text(
				event_time + self.epoch_duration / 2,
				annotation_y,
				annotation,
				horizontalalignment='center',
				verticalalignment='bottom',
				fontsize=4,
				fontweight='bold',
				color=color,
				bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1)
			)

		# Slight padding for annotations
		padding = 0.1 * (y_max - y_min)
		ax.set_ylim(y_min, y_max + padding)

		ax.set_title(f'Continuous EEG Data for {n_epochs} Epochs in Batch index {current_batch_idx}')
		ax.set_xlabel('Time (s)')
		ax.set_ylabel('Amplitude (µV)')
		ax.grid(True)
		ax.legend(loc='upper right', fontsize=8)
		plt.draw()
		plt.pause(1)


