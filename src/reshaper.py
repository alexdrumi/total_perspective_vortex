from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import numpy as np

'''
If X has a shape of (100, 64, 50), 
representing 100 samples, 64 channels, and 50 time points, Reshaper will transform it into (100, 3200), flattening 64 * 50 into a single feature dimension.
Like this we can always analyze 2 features and do PCA on 2 features->(s)amples, channels * time points)
'''

class Reshaper(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass

	def fit(self, X: np.ndarray, y=None):
		return self
	
	def transform(self, X:np.ndarray) -> np.ndarray:
		#flatten the last two dimensions (channels,time points) into a single feature dimension
		return X.reshape((X.shape[0], -1))