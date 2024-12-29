from src.pipeline.custom_scaler import CustomScaler
from src.pipeline.reshaper import Reshaper
from src.pipeline.pca import My_PCA
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from typing import Optional
from sklearn.base import BaseEstimator



class PipelineBuilder():
	def __init__(self, n_components: int = 2, classifier: Optional[BaseEstimator] = None) -> None:
		"""
		Creates the PipelineBuilder object with n_components to be used in the PCA dimensionality reduction.

		Args:
			n_components (int): amount of components to be used in PCA.
			classifier (Optional[BaseEstimator]): Classifier specified at construction. Defaults to MultiLayerPerceptronClassifier.
		
		"""
		self.n_components = n_components
		self.classifier = classifier



	def build_pipeline(self) -> Pipeline:
		"""
		Builds the pipeline with n_components to be used in the PCA dimensionality reduction.

		Args:
			n_components (int): amount of components to be used in PCA.
			classifier (sklearn classifier): Classifier specified at construction. Defaults to MultiLayerPerceptronClassifier.
		
		Returns:
			Pipeline: A sckit-learn pipeline object with the specified parameters.	
		"""
		if self.classifier is None:
			classifier = MLPClassifier(hidden_layer_sizes=(20,10), max_iter=16000, random_state=42)
		
		pipeline_steps = [
			('scaler', CustomScaler()),
			('reshaper', Reshaper()),
			('pca', My_PCA(self.n_components)),
			('classifier', classifier)
		]

		pipeline = Pipeline(pipeline_steps)
		return pipeline