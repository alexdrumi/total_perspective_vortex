from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import numpy as np

"""
If X has a shape of (100, 64, 50), 
representing 100 samples, 64 channels, and 50 time points, Reshaper will transform it into (100, 3200), flattening 64 * 50 into a single feature dimension.
Like this we can always analyze 2 features and do PCA on 2 features->samples, channels * time points)
"""

class Reshaper(BaseEstimator, TransformerMixin):
	def __init__(self):
		"""
		A custom transformer that reshapes input arrays to have a flattened second dimension.
		
		Returns:
			None
		"""
		pass



	def fit(self, X: np.ndarray, y=None) -> "Reshaper":
		"""
		Fits the transformer. No fitting is required for this transformer, but leaving this for compatibility.

		Args:
			X (np.ndarray): Input array of shape (n_samples, ...).
			y: Ignored, exists for compatibility.

		Returns:
			Reshaper: The fitted transformer.
		"""
		return self



	def transform(self, X:np.ndarray) -> np.ndarray:
		"""
		Transforms the input array by reshaping it to (n_samples, -1).

		Args:
			X (np.ndarray): Input array of shape (n_samples, ...).

		Returns:
			np.ndarray: Reshaped array of shape (n_samples, -1).
		"""
		if X.ndim < 2:
			raise ValueError("Input array must have at least two dimensions for reshaping.")
		return X.reshape((X.shape[0], -1))