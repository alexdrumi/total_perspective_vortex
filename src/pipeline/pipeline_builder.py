from src.pipeline.custom_scaler import CustomScaler
from src.pipeline.reshaper import Reshaper
from src.pipeline.pca import My_PCA
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline



class PipelineBuilder():
	def __init__(self, n_components=2, classifier=None):
		self.n_components = n_components
		self.classifier = classifier


	def build_pipeline(self):
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