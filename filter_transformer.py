from sklearn.base import BaseEstimator, TransformerMixin


class InitialFilterTransformer(BaseEstimator, TransformerMixin):
	'''
	Input: Raw EEG data (from mne.io.Raw object)
	Output: Filtered EEG data (mne.io.Raw object)
	transform method should always return the transformed data: eg np array of sorts
	fit method: accepts (transformed) data, performs computation and returns SELF
	'''
	
	def __init__(self):
		self.lo_cut = 0.1
		self.hi_cut = 30
		self.noise_cut = 50

	#we need a fit and a transform method, implement filter_raw_data as transform
	'''
	Calling fit on the pipeline is the same as calling fit on each estimator in turn, transform the input and pass it on to the next step. The pipeline has all the methods that the last estimator in the pipeline has, i.e. if the last estimator is a classifier, the Pipeline can be used as a classifier.
	If the last estimator is a transformer, again, so is the pipeline.
	Returns:

	@transform (https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html)
	Return:
	Xt
	ndarray of shape (n_samples, n_transformed_features)
	Transformed data.
	'''
	def fit(self):
		return self

	#these ones should go to another part of the pipeline called FilterTransformer
	def filter_frequencies(self, raw, lo_cut, hi_cut, noise_cut):
		filtered_lo_hi = raw.copy().filter(lo_cut, hi_cut)
		filter_noise = filtered_lo_hi.notch_filter(noise_cut)
		return filter_noise


	def transform(self, X):
		'''
		Input: X -> raw eeg data
		Output: filtered eeg data
		'''
		filtered_data = []
		for raw in X:
			raw.load_data() #gotta close somewhere prob
			filtered_data.append(self.filter_frequencies(raw, self.lo_cut, self.hi_cut, self.noise_cut))
		# self.raw_data = [] #empty memory, wouldnt it leak? 
		return filtered_data




def filter_frequencies(raw, lo_cut, hi_cut, noise_cut):
		filtered_lo_hi = raw.copy().filter(lo_cut, hi_cut)
		filter_noise = filtered_lo_hi.notch_filter(noise_cut)
		return filter_noise


def initial_filter(X):
	'''
	Input: X -> raw eeg data
	Output: filtered eeg data
	'''
	lo_cut = 0.1
	hi_cut = 30
	noise_cut = 50
	filtered_data = []
	for raw in X:
		raw.load_data() #gotta close somewhere prob
		filtered_data.append(filter_frequencies(raw, lo_cut, hi_cut, noise_cut))
	# self.raw_data = [] #empty memory, wouldnt it leak? 
	return filtered_data