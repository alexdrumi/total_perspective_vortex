from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys




class CustomScaler(BaseEstimator, TransformerMixin):
	def __init__(self):
		"""
		A custom scaler for 3D arrays. Scales each feature dimension separately using standardscaler.
		Initializes the custromscaler with a dictionary to store individual standardscaler instances
		for each feature dimension.
		"""
		self.scalers = {}
	


	def fit(self, X: np.ndarray, y=None):
		"""
		Fits a standardscaler for each feature dimension in the 3D input array.

		Args:
			X (np.ndarray): Input data of shape (n_samples, n_features, n_timesteps).
			y: Ignored, exists for compatibility with scikit-learn API.

		Returns:
			CustomScaler: The fitted scaler instance.

		Raises:
			TypeError: If the input X is not a 3D numpy array.
		"""
		if not isinstance(X, np.ndarray) or X.ndim !=3:
			raise TypeError('Input X must be a 3D np.ndarray: ')

		for i in range(X.shape[1]):
			scaler = StandardScaler()
			scaler.fit(X[:, i, :].reshape(-1, X.shape[2]))
			self.scalers[i] = scaler
		return self
	


	def transform(self, X: np.ndarray):
		"""
		Transforms the input data using the fitted scalers.

		Args:
			X (np.ndarray): Input data of shape (n_samples, n_features, n_timesteps).

		Returns:
			np.ndarray: Scaled data of the same shape as input.

		Raises:
			ValueError: If the scaler is not fitted before calling transform.
		"""
		if not self.scalers: #unlikely
			raise ValueError("CustomScaler has not been fitted yet.")

		x_copy = X.copy()
		for i in range(X.shape[1]):
			x_copy[:, i, :] = self.scalers[i].transform(X[:, i, :])
		return x_copy



	def fit_transform(self, X:np.ndarray, y=None):
		"""
		Fits the scaler to the data and then transforms it.

		Args:
			X (np.ndarray): Input data of shape (n_samples, n_features, n_timesteps).
			y: Just placeholder, exists for compatibility with scikit API.

		Returns:
			np.ndarray: Scaled data of the same shape as input.
		"""
		return self.fit(X, y).transform(X)