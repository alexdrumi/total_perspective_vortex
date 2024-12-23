from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import numpy as np

import sys

class CustomScaler(BaseEstimator, TransformerMixin):
	def __init__(self):
		self.scalers = {}
	


	def fit(self, X: np.ndarray, y=None):
		if not isinstance(X, np.ndarray) or X.ndim !=3:
			raise TypeError('Input X must be a 3D np.ndarray: ')
		for i in range(X.shape[1]):
			scaler = StandardScaler()
			scaler.fit(X[:, i, :].reshape(-1, X.shape[2]))
			self.scalers[i] = scaler
		return self
	


	def transform(self, X: np.ndarray):
		x_copy = X.copy()
		for i in range(X.shape[1]):
			x_copy[:, i, :] = self.scalers[i].transform(X[:, i, :])
		return x_copy



	def fit_transform(self, X:np.ndarray, y=None):
		return self.fit(X, y).transform(X)