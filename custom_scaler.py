from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import numpy as np


class CustomScaler(BaseEstimator, TransformerMixin):
	def __init__(self):
		self.scalers = {}
	

	def fit(self, X, y=None):
		for i in range(X.shape[1]):
			scaler = StandardScaler()
			scaler.fit(X[:, i, :].reshape(-1, X.shape[2])) #not entirely surte about these dimensions
			self.scalers[i] = scaler
			# x_copy[:, i, :] = self.scalers[i].fit_transform(X[:, i, :])
		# return x_copy
		return self
	

	def transform(self, X):
		x_copy = X.copy()
		for i in range(X.shape[1]):
			#use the previously saved scalers, this will be called when using predict as well
			x_copy[:, i, :] = self.scalers[i].transform(X[:, i, :])
		return x_copy



	def fit_transform(self, X, y=None):
		#during training this is called
		return self.fit(X, y).transform(X)