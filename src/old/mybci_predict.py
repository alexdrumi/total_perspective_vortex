import numpy as np
import mne
import sys
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import ShuffleSplit, cross_val_score, KFold
from sklearn.model_selection import KFold

from dataset_preprocessor import Preprocessor
from feature_extractor import FeatureExtractor
from experiment_predictor import ExperimentPredictor
from myapp import MyApp
from epoch_concatenator import EpochConcatenator


import joblib
import logging

from pca import My_PCA
from epoch_extractor import EpochExtractor

import time

from custom_scaler import CustomScaler
from reshaper import Reshaper

from command_line_parser import CommandLineParser

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

file_handler = logging.FileHandler('../logs/error_log.log', mode='w')
file_handler.setLevel(logging.ERROR)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)



channels = ["Fc3.", "Fcz.", "Fc4.", "C3..", "C1..", "Cz..", "C2..", "C4.."]
predict = [





# run 1, eyes open
	"../../data/S011/S011R01.edf",
	"../../data/S012/S012R01.edf",
	"../../data/S023/S023R01.edf",
	"../../data/S024/S024R01.edf",
	"../../data/S025/S025R01.edf",
	"../../data/S026/S026R01.edf",
	"../../data/S027/S027R01.edf",
	"../../data/S028/S028R01.edf",
	"../../data/S029/S029R01.edf",
	"../../data/S030/S030R01.edf",
	"../../data/S031/S031R01.edf",
	"../../data/S032/S032R01.edf",
	"../../data/S033/S033R01.edf",
	"../../data/S034/S034R01.edf",


#run 2, eyes closed
	"../../data/S011/S011R02.edf",
	"../../data/S012/S012R02.edf",
	"../../data/S023/S023R02.edf",
	"../../data/S024/S024R02.edf",
	"../../data/S025/S025R02.edf",
	"../../data/S026/S026R02.edf",
	"../../data/S027/S027R02.edf",
	"../../data/S028/S028R02.edf",
	"../../data/S029/S029R02.edf",
	"../../data/S030/S030R02.edf",
	"../../data/S031/S031R02.edf",
	"../../data/S032/S032R02.edf",
	"../../data/S033/S033R02.edf",
	"../../data/S034/S034R02.edf",



	#3,7,11
	#3-Task 1 (open and close left or right fist) #run 3-T1:left, T2:right real
	#4-Task 2 (imagine opening and closing left or right fist) run 3-T1:left, T2:right imagined
	#5-Task 3 (open and close both fists or both feet) run 5-9-13-T1:both fists T2:both feet real
	#6-Task 4 (imagine opening and closing both fists or both feet) run 6-10-14-T1:both fists imagined T2:both feet imagined
	# "../../data/S030/S030R03.edf", #run type 1
	# "../../data/S030/S030R07.edf", #run type 2
	# "../../data/S030/S030R11.edf", #run type 3
	# "../../data/S031/S031R03.edf",
	# "../../data/S031/S031R07.edf",
	# "../../data/S031/S031R11.edf",
	# "../../data/S032/S032R03.edf",
	# "../../data/S032/S032R07.edf",
	# "../../data/S032/S032R11.edf",

	"../../data/S022/S022R03.edf",
	"../../data/S022/S022R07.edf",
	"../../data/S022/S022R11.edf",
	"../../data/S023/S023R03.edf",
	"../../data/S023/S023R07.edf",
	"../../data/S023/S023R11.edf",
	"../../data/S024/S024R03.edf",
	"../../data/S024/S024R07.edf",
	"../../data/S024/S024R11.edf",
	"../../data/S025/S025R03.edf",
	"../../data/S025/S025R07.edf",
	"../../data/S025/S025R11.edf",
	"../../data/S026/S026R03.edf",
	"../../data/S026/S026R07.edf",
	"../../data/S026/S026R11.edf",
	"../../data/S027/S027R03.edf",
	"../../data/S027/S027R07.edf",
	"../../data/S027/S027R11.edf",
	"../../data/S028/S028R03.edf",
	"../../data/S028/S028R07.edf",
	"../../data/S028/S028R11.edf",
	"../../data/S029/S029R03.edf",
	"../../data/S029/S029R07.edf",
	"../../data/S029/S029R11.edf",
	"../../data/S030/S030R03.edf",
	"../../data/S030/S030R07.edf",
	"../../data/S030/S030R11.edf",
	"../../data/S031/S031R03.edf",
	"../../data/S031/S031R07.edf",
	"../../data/S031/S031R11.edf",
	"../../data/S032/S032R03.edf",
	"../../data/S032/S032R07.edf",
	"../../data/S032/S032R11.edf",
	"../../data/S033/S033R03.edf",
	"../../data/S033/S033R07.edf",
	"../../data/S033/S033R11.edf",
	"../../data/S034/S034R03.edf",
	"../../data/S034/S034R07.edf",
	"../../data/S034/S034R11.edf",
	"../../data/S035/S035R03.edf",
	"../../data/S035/S035R07.edf",
	"../../data/S035/S035R11.edf",
	"../../data/S036/S036R03.edf",
	"../../data/S036/S036R07.edf",
	"../../data/S036/S036R11.edf",
	"../../data/S037/S037R03.edf",
	"../../data/S037/S037R07.edf",
	"../../data/S037/S037R11.edf",
	"../../data/S038/S038R03.edf",
	"../../data/S038/S038R07.edf",
	"../../data/S038/S038R11.edf",
	"../../data/S039/S039R03.edf",
	"../../data/S039/S039R07.edf",
	"../../data/S039/S039R11.edf",
	"../../data/S040/S040R03.edf",
	"../../data/S040/S040R07.edf",
	"../../data/S040/S040R11.edf",
	"../../data/S041/S041R03.edf",
	"../../data/S041/S041R07.edf",
	"../../data/S041/S041R11.edf",
	"../../data/S042/S042R03.edf",
	"../../data/S042/S042R07.edf",
	"../../data/S042/S042R11.edf",
	"../../data/S043/S043R03.edf",
	"../../data/S043/S043R07.edf",
	"../../data/S043/S043R11.edf",
	"../../data/S044/S044R03.edf",
	"../../data/S044/S044R07.edf",
	"../../data/S044/S044R11.edf",
	"../../data/S045/S045R03.edf",
	"../../data/S045/S045R07.edf",
	"../../data/S045/S045R11.edf",
	"../../data/S046/S046R03.edf",
	"../../data/S046/S046R07.edf",
	"../../data/S046/S046R11.edf",
	"../../data/S047/S047R03.edf",
	"../../data/S047/S047R07.edf",
	"../../data/S047/S047R11.edf",
	"../../data/S048/S048R03.edf",
	"../../data/S048/S048R07.edf",
	"../../data/S048/S048R11.edf",
	"../../data/S049/S049R03.edf",
	"../../data/S049/S049R07.edf",
	"../../data/S049/S049R11.edf",
	"../../data/S050/S050R03.edf",
	"../../data/S050/S050R07.edf",
	"../../data/S050/S050R11.edf",

	#4,8,12
	"../../data/S031/S031R08.edf", #run type 4
	"../../data/S031/S031R12.edf", #run type 5
	"../../data/S032/S032R04.edf", #run type 6
	"../../data/S032/S032R08.edf",
	"../../data/S032/S032R12.edf",
	"../../data/S033/S033R04.edf",
	"../../data/S033/S033R08.edf",
	"../../data/S033/S033R12.edf",


	"../../data/S056/S056R08.edf",
	"../../data/S056/S056R12.edf",
	"../../data/S057/S057R04.edf",
	"../../data/S057/S057R08.edf",
	"../../data/S057/S057R12.edf",
	"../../data/S058/S058R04.edf",
	"../../data/S058/S058R08.edf",
	"../../data/S058/S058R12.edf",
	"../../data/S059/S059R04.edf",
	"../../data/S059/S059R08.edf",
	"../../data/S059/S059R12.edf",
	"../../data/S060/S060R04.edf",
	"../../data/S060/S060R08.edf",
	"../../data/S060/S060R12.edf",
	"../../data/S061/S061R04.edf",
	"../../data/S061/S061R08.edf",
	"../../data/S061/S061R12.edf",
	"../../data/S062/S062R04.edf",
	"../../data/S062/S062R08.edf",
	"../../data/S062/S062R12.edf",
	"../../data/S063/S063R04.edf",
	"../../data/S063/S063R08.edf",
	"../../data/S063/S063R12.edf",
	"../../data/S064/S064R04.edf",
	"../../data/S064/S064R08.edf",
	"../../data/S064/S064R12.edf",
	"../../data/S065/S065R04.edf",
	"../../data/S065/S065R08.edf",
	"../../data/S065/S065R12.edf",
	"../../data/S066/S066R04.edf",
	"../../data/S066/S066R08.edf",
	"../../data/S066/S066R12.edf",
	"../../data/S067/S067R04.edf",
	"../../data/S067/S067R08.edf",
	"../../data/S067/S067R12.edf",
	"../../data/S068/S068R04.edf",
	"../../data/S068/S068R08.edf",
	"../../data/S068/S068R12.edf",
	"../../data/S069/S069R04.edf",
	"../../data/S069/S069R08.edf",
	"../../data/S069/S069R12.edf",
	"../../data/S070/S070R04.edf",
	"../../data/S070/S070R08.edf",
	"../../data/S070/S070R12.edf",
	"../../data/S071/S071R04.edf",
	"../../data/S071/S071R08.edf",
	"../../data/S071/S071R12.edf",
	"../../data/S072/S072R04.edf",
	"../../data/S072/S072R08.edf",
	"../../data/S072/S072R12.edf",
	"../../data/S073/S073R04.edf",
	"../../data/S073/S073R08.edf",
	"../../data/S073/S073R12.edf",
	"../../data/S074/S074R04.edf",
	"../../data/S074/S074R08.edf",
	"../../data/S074/S074R12.edf",
	"../../data/S075/S075R04.edf",
	"../../data/S075/S075R08.edf",
	"../../data/S075/S075R12.edf",
	"../../data/S076/S076R04.edf",
	"../../data/S076/S076R08.edf",
	"../../data/S076/S076R12.edf",
	"../../data/S077/S077R04.edf",
	"../../data/S077/S077R08.edf",
	"../../data/S077/S077R12.edf",
	"../../data/S078/S078R04.edf",
	"../../data/S078/S078R08.edf",
	"../../data/S078/S078R12.edf",
	"../../data/S079/S079R04.edf",
	"../../data/S079/S079R08.edf",
	"../../data/S079/S079R12.edf",
	"../../data/S080/S080R04.edf",
	"../../data/S080/S080R08.edf",
	"../../data/S080/S080R12.edf",
	"../../data/S081/S081R04.edf",


	#5,9,13
	"../../data/S031/S031R09.edf",
	"../../data/S031/S031R13.edf",
	"../../data/S032/S032R05.edf",
	"../../data/S032/S032R09.edf",
	"../../data/S032/S032R13.edf",
	"../../data/S033/S033R05.edf",
	"../../data/S033/S033R09.edf",
	"../../data/S033/S033R13.edf",


	"../../data/S056/S056R09.edf",
	"../../data/S056/S056R13.edf",
	"../../data/S057/S057R05.edf",
	"../../data/S057/S057R09.edf",
	"../../data/S057/S057R13.edf",
	"../../data/S058/S058R05.edf",
	"../../data/S058/S058R09.edf",
	"../../data/S058/S058R13.edf",
	"../../data/S059/S059R05.edf",
	"../../data/S059/S059R09.edf",
	"../../data/S059/S059R13.edf",
	"../../data/S060/S060R05.edf",
	"../../data/S060/S060R09.edf",
	"../../data/S060/S060R13.edf",
	"../../data/S061/S061R05.edf",
	"../../data/S061/S061R09.edf",
	"../../data/S061/S061R13.edf",
	"../../data/S062/S062R05.edf",
	"../../data/S062/S062R09.edf",
	"../../data/S062/S062R13.edf",
	"../../data/S063/S063R05.edf",
	"../../data/S063/S063R09.edf",
	"../../data/S063/S063R13.edf",
	"../../data/S064/S064R05.edf",
	"../../data/S064/S064R09.edf",
	"../../data/S064/S064R13.edf",
	"../../data/S065/S065R05.edf",
	"../../data/S065/S065R09.edf",
	"../../data/S065/S065R13.edf",
	"../../data/S066/S066R05.edf",
	"../../data/S066/S066R09.edf",
	"../../data/S066/S066R13.edf",
	"../../data/S067/S067R05.edf",
	"../../data/S067/S067R09.edf",
	"../../data/S067/S067R13.edf",
	"../../data/S068/S068R05.edf",
	"../../data/S068/S068R09.edf",
	"../../data/S068/S068R13.edf",
	"../../data/S069/S069R05.edf",
	"../../data/S069/S069R09.edf",
	"../../data/S069/S069R13.edf",
	"../../data/S070/S070R05.edf",
	"../../data/S070/S070R09.edf",
	"../../data/S070/S070R13.edf",
	"../../data/S071/S071R05.edf",
	"../../data/S071/S071R09.edf",
	"../../data/S071/S071R13.edf",
	"../../data/S072/S072R05.edf",
	"../../data/S072/S072R09.edf",
	"../../data/S072/S072R13.edf",
	"../../data/S073/S073R05.edf",
	"../../data/S073/S073R09.edf",
	"../../data/S073/S073R13.edf",
	"../../data/S074/S074R05.edf",
	"../../data/S074/S074R09.edf",
	"../../data/S074/S074R13.edf",
	"../../data/S075/S075R05.edf",
	"../../data/S075/S075R09.edf",
	"../../data/S075/S075R13.edf",
	"../../data/S076/S076R05.edf",
	"../../data/S076/S076R09.edf",
	"../../data/S076/S076R13.edf",
	"../../data/S077/S077R05.edf",
	"../../data/S077/S077R09.edf",
	"../../data/S077/S077R13.edf",
	"../../data/S078/S078R05.edf",
	"../../data/S078/S078R09.edf",
	"../../data/S078/S078R13.edf",
	"../../data/S079/S079R05.edf",
	"../../data/S079/S079R09.edf",
	"../../data/S079/S079R13.edf",
	"../../data/S080/S080R05.edf",
	"../../data/S080/S080R09.edf",


	#6,10,14
	"../../data/S022/S022R10.edf",
	"../../data/S022/S022R14.edf",
	"../../data/S023/S023R06.edf",
	"../../data/S023/S023R10.edf",
	"../../data/S023/S023R14.edf",
	"../../data/S024/S024R06.edf",
	"../../data/S024/S024R10.edf",
	"../../data/S024/S024R14.edf",
	"../../data/S025/S025R06.edf",
	"../../data/S025/S025R10.edf",
	"../../data/S025/S025R14.edf",
	"../../data/S026/S026R06.edf",
	"../../data/S026/S026R10.edf",
	"../../data/S026/S026R14.edf",


	"../../data/S027/S027R06.edf",
	"../../data/S027/S027R10.edf",
	"../../data/S027/S027R14.edf",
	"../../data/S028/S028R06.edf",
	"../../data/S028/S028R10.edf",
	"../../data/S028/S028R14.edf",
	"../../data/S029/S029R06.edf",
	"../../data/S029/S029R10.edf",
	"../../data/S029/S029R14.edf",
	"../../data/S030/S030R06.edf",
	"../../data/S030/S030R10.edf",
	"../../data/S030/S030R14.edf",
	"../../data/S031/S031R06.edf",
	"../../data/S031/S031R10.edf",
	"../../data/S031/S031R14.edf",
	"../../data/S032/S032R06.edf",
	"../../data/S032/S032R10.edf",
	"../../data/S032/S032R14.edf",
	"../../data/S033/S033R06.edf",
	"../../data/S033/S033R10.edf",
	"../../data/S033/S033R14.edf",
	"../../data/S034/S034R06.edf",
	"../../data/S034/S034R10.edf",
	"../../data/S034/S034R14.edf",
	"../../data/S035/S035R06.edf",
	"../../data/S035/S035R10.edf",
	"../../data/S035/S035R14.edf",
	"../../data/S036/S036R06.edf",
	"../../data/S036/S036R10.edf",
	"../../data/S036/S036R14.edf",
	"../../data/S037/S037R06.edf",
	"../../data/S037/S037R10.edf",
	"../../data/S037/S037R14.edf",
	"../../data/S038/S038R06.edf",
	"../../data/S038/S038R10.edf",
	"../../data/S038/S038R14.edf",
	"../../data/S039/S039R06.edf",
	"../../data/S039/S039R10.edf",
	"../../data/S039/S039R14.edf",
	"../../data/S040/S040R06.edf",
	"../../data/S040/S040R10.edf",
	"../../data/S040/S040R14.edf",
	"../../data/S041/S041R06.edf",
	"../../data/S041/S041R10.edf",
	"../../data/S041/S041R14.edf",
	"../../data/S042/S042R06.edf",
	"../../data/S042/S042R10.edf",
	"../../data/S042/S042R14.edf",
	"../../data/S043/S043R06.edf",
	"../../data/S043/S043R10.edf",
	"../../data/S043/S043R14.edf",
	"../../data/S044/S044R06.edf",
	"../../data/S044/S044R10.edf",
	"../../data/S044/S044R14.edf",
	"../../data/S045/S045R06.edf",
	"../../data/S045/S045R10.edf",
	"../../data/S045/S045R14.edf",
	"../../data/S046/S046R06.edf",
	"../../data/S046/S046R10.edf",
	"../../data/S046/S046R14.edf",
	"../../data/S047/S047R06.edf",
	"../../data/S047/S047R10.edf",
	"../../data/S047/S047R14.edf",
	"../../data/S048/S048R06.edf",
	"../../data/S048/S048R10.edf",
	"../../data/S048/S048R14.edf",
	"../../data/S049/S049R06.edf",
	"../../data/S049/S049R10.edf",
	"../../data/S049/S049R14.edf",
	"../../data/S050/S050R06.edf",
	"../../data/S050/S050R10.edf",
	"../../data/S050/S050R14.edf",
	"../../data/S051/S051R06.edf",
	"../../data/S051/S051R10.edf",
	"../../data/S051/S051R14.edf",
	"../../data/S052/S052R06.edf",
	"../../data/S052/S052R10.edf",
	"../../data/S052/S052R14.edf",
	"../../data/S053/S053R06.edf",
	"../../data/S053/S053R10.edf",
	"../../data/S053/S053R14.edf",
	"../../data/S054/S054R06.edf",
	"../../data/S054/S054R10.edf",
	"../../data/S054/S054R14.edf",
	"../../data/S055/S055R06.edf",
	"../../data/S055/S055R10.edf",
	"../../data/S055/S055R14.edf",
	"../../data/S056/S056R06.edf",
	"../../data/S056/S056R10.edf",
	"../../data/S056/S056R14.edf",
	"../../data/S057/S057R06.edf",
	"../../data/S057/S057R10.edf",
	"../../data/S057/S057R14.edf",
	"../../data/S058/S058R06.edf",
	"../../data/S058/S058R10.edf",
	"../../data/S058/S058R14.edf",
]

#-------------------------------------------------------

def main():
	try:
		predict_data_path = predict
		app = MyApp()
		app.run(predict_data_path)
	except FileNotFoundError as e:
		logging.error(f"File not found: {e}")
	except PermissionError as e:
		logging.error(f"Permission error: {e}")
	except IOError as e:
		logging.error(f"IO error: {e}")
	except ValueError as e:
		logging.error(f"Value error: {e}")



if __name__ == "__main__":
	main()
