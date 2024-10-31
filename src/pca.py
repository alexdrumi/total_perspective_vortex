from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class My_PCA(BaseEstimator, TransformerMixin):
	def __init__(self, n_comps = 2): #amount of PCAs to select, we have to check later for how much percentage do they cover
		self.n_comps = n_comps
		self.basis = None
		self.current_centered_feature = None

	def fit(self, x_features: np.ndarray, y=None) -> "My_PCA":
		self.mean_ = np.mean(x_features, axis=0)
		# When X is a scipy sparse matrix, self.mean_ is a numpy matrix, so we need
		# to transform it to a 1D array. Note that this is not the case when X
		# is a scipy sparse array.
		# TODO: remove the following two lines when scikit-learn only depends
		# on scipy versions that no longer support scipy.sparse matrices.
		self.mean_ = np.reshape(np.asarray(self.mean_), (-1,))

		n_samples=x_features.shape[0]
		C = x_features.T @ x_features
		C -= (
			n_samples
			* np.reshape(self.mean_, (-1, 1))
			* np.reshape(self.mean_, (1, -1))
		)
		C /= n_samples - 1

		x_features = x_features.T
		zerodx = x_features

		for i in range(len(x_features)):
			zerodx[i] -= zerodx[i].mean()

		#compute covariance matrix
		#cov_matrix2 = np.matmul(zerodx, zerodx.T) / (zerodx.shape[1] - 1)
		cov_matrix = np.cov(zerodx) #T is to compute between features instead of datapoints, (rows)
		cov_matrix = C

		#eigenval and eigenvec
		#eig is making imaginary numbers because of floating point precisions
		eigvals, eigvecs = np.linalg.eigh(cov_matrix)

		#eigvals and eigvecs pair up
		#eigvals are sorted in ascending order
		#eigvecs are column vectors
		eigvals = np.reshape(np.asarray(eigvals), (-1,))
		eigvecs = np.asarray(eigvecs)

		#sort eigvals and eigvecs in descending order
		eigvals = np.flip(eigvals, axis=0)
		eigvecs = np.flip(eigvecs, axis=1)
		eigvals[eigvals < 0.0] = 0.0

		Vt = eigvecs.T
		#Vt = flipstuff(Vt)
		self.basis = np.asarray(Vt[:self.n_comps, :]) #there war copy=True here but not sure why would we need a copy
		return self


	def transform(self, x_features: np.ndarray) -> np.ndarray:
		X_transformed = x_features @ self.basis.T
		X_transformed -= np.reshape(self.mean_, (1, -1)) @ self.basis.T
		return X_transformed

