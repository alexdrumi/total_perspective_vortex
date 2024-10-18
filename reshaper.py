from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import numpy as np



class Reshaper(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass

	def fit(self, X, y=None):
		return self
	
	def transform(self, X):
		#flatten the last two dimensions (channels,time points) into a single feature dimension
		return X.reshape((X.shape[0], -1))