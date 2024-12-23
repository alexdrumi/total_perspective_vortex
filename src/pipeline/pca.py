from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class My_PCA(BaseEstimator, TransformerMixin):
	def __init__(self, n_comps = 2):
		self.n_comps = n_comps
		self.basis = None
		self.current_centered_feature = None



	def fit(self, x_features: np.ndarray, y=None) -> "My_PCA":
		self.mean_ = np.mean(x_features, axis=0)
		self.mean_ = np.reshape(np.asarray(self.mean_), (-1,)) #consistent for 1d operations

		n_samples=x_features.shape[0]

		#captures how much each feature varies from the mean with respect to every other feature
		C = x_features.T @ x_features #dot product of the transposed data with itself
		C -= ( #subtract outer product of the mean vector scaled by the nr of samples to center the data
			n_samples
			* np.reshape(self.mean_, (-1, 1))
			* np.reshape(self.mean_, (1, -1))
		)
		C /= n_samples - 1 #sample mean

		#center the data
		x_features = x_features.T
		zerodx = x_features

		for i in range(len(x_features)):
			zerodx[i] -= zerodx[i].mean()

	
		cov_matrix = np.cov(zerodx)
		cov_matrix = C

		#eigen decomposition -> find eigenvals, and their associated vectors
		eigvals, eigvecs = np.linalg.eigh(cov_matrix)
		eigvals = np.reshape(np.asarray(eigvals), (-1,))
		eigvecs = np.asarray(eigvecs)

		eigvals = np.flip(eigvals, axis=0)
		eigvecs = np.flip(eigvecs, axis=1)
		eigvals[eigvals < 0.0] = 0.0

		Vt = eigvecs.T
		self.basis = np.asarray(Vt[:self.n_comps, :])
		return self



	def transform(self, x_features: np.ndarray) -> np.ndarray:
		X_transformed = x_features @ self.basis.T
		X_transformed -= np.reshape(self.mean_, (1, -1)) @ self.basis.T
		return X_transformed

