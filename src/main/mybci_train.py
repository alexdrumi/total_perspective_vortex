#!/usr/bin/python
import numpy as np
import mne
import sys
import matplotlib.pyplot as plt
import joblib
import logging
from pathlib import Path
import time
import yaml
import os

# Add the `src` directory to the system path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(src_path)

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from mne.decoding import CSP

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import ShuffleSplit, cross_val_score, KFold, GridSearchCV

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from src.mlflow_manager import MlflowManager
from src.dataset_preprocessor import Preprocessor
from src.feature_extractor import FeatureExtractor
from src.pca import My_PCA
from src.epoch_extractor import EpochExtractor
from src.custom_scaler import CustomScaler
from src.reshaper import Reshaper

from src.command_line_parser import CommandLineParser
import subprocess

#logger config
#logging for both file and console
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

#file handler - Logs to a file
file_handler = logging.FileHandler('../../logs/error_log.log', mode='w')
file_handler.setLevel(logging.ERROR)

#stream handler - Logs to terminal (console)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.ERROR)

#formatfor log messages
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

#handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

mne.set_log_level(verbose='WARNING')

channels = ["Fc3.", "Fcz.", "Fc4.", "C3..", "C1..", "Cz..", "C2..", "C4.."]


# channels = ["Fc1.","Fc2.", "Fc3.", "Fcz.", "Fc4.", "C3..", "C1..", "Cz..", "C2..", "C4.."]
            # "CP3",
            # "CP1",
            # "CPz",
            # "CP2",
            # "CP4",
            # "Fpz",
# predict = [
# "data/S018/S018R11.edf",
# "data/S042/S042R07.edf",
# "data/S042/S042R03.edf",
# "data/S104/S104R11.edf",
# # "data/S104/S104R07.edf",
# # "data/S090/S090R11.edf",
# # "data/S086/S086R11.edf",
# # "data/S086/S086R03.edf",
# # "data/S086/S086R07.edf",
# # "data/S017/S017R11.edf",
# # "data/S017/S017R07.edf",
# # "data/S017/S017R03.edf",
# # "data/S013/S013R07.edf",
# # "data/S013/S013R11.edf",
# # "data/S013/S013R03.edf",
# # "data/S055/S055R11.edf",
# # "data/S055/S055R07.edf",
# # "data/S055/S055R03.edf",
# # "data/S016/S016R03.edf",
# # "data/S016/S016R07.edf",
# # "data/S016/S016R11.edf",
# #"/data/S103/S103R11.edf",
# ]


train = [
	#run 3,7,11
	"../data/S001/S001R03.edf",
	"../data/S001/S001R07.edf",
	"../data/S001/S001R11.edf",
	"../data/S002/S002R03.edf",
	"../data/S002/S002R07.edf",
	"../data/S002/S002R11.edf",
	"../data/S003/S003R03.edf",
	"../data/S003/S003R07.edf",
	"../data/S003/S003R11.edf",
	"../data/S004/S004R03.edf",
	"../data/S004/S004R07.edf",
	"../data/S004/S004R11.edf",
	"../data/S005/S005R03.edf",
	"../data/S005/S005R07.edf",
	"../data/S005/S005R11.edf",
	"../data/S006/S006R03.edf",
	"../data/S006/S006R07.edf",
	"../data/S006/S006R11.edf",
	"../data/S007/S007R03.edf",
	"../data/S007/S007R07.edf",
	"../data/S007/S007R11.edf",
	"../data/S008/S008R03.edf",
	"../data/S008/S008R07.edf",
	"../data/S008/S008R11.edf",
	"../data/S009/S009R03.edf",
	"../data/S009/S009R07.edf",
	"../data/S009/S009R11.edf",
	"../data/S010/S010R03.edf",
	"../data/S010/S010R07.edf",
	"../data/S010/S010R11.edf",
	"../data/S011/S011R03.edf",
	"../data/S011/S011R07.edf",
	"../data/S011/S011R11.edf",
	"../data/S012/S012R03.edf",
	"../data/S012/S012R07.edf",
	"../data/S012/S012R11.edf",
	"../data/S013/S013R03.edf",
	"../data/S013/S013R07.edf",
	"../data/S013/S013R11.edf",
	"../data/S014/S014R03.edf",
	"../data/S014/S014R07.edf",
	"../data/S014/S014R11.edf",
	"../data/S015/S015R03.edf",
	"../data/S015/S015R07.edf",
	"../data/S015/S015R11.edf",
	"../data/S016/S016R03.edf",
	"../data/S016/S016R07.edf",
	"../data/S016/S016R11.edf",
	"../data/S017/S017R03.edf",
	"../data/S017/S017R07.edf",
	"../data/S017/S017R11.edf",
	"../data/S018/S018R03.edf",
	"../data/S018/S018R07.edf",
	"../data/S018/S018R11.edf",
	"../data/S019/S019R03.edf",
	"../data/S019/S019R07.edf",
	"../data/S019/S019R11.edf",
	"../data/S020/S020R03.edf",
	"../data/S020/S020R07.edf",
	"../data/S020/S020R11.edf",
	"../data/S021/S021R03.edf",
	"../data/S021/S021R07.edf",
	"../data/S021/S021R11.edf",
	"../data/S022/S022R03.edf",
	"../data/S022/S022R07.edf",
	"../data/S022/S022R11.edf",
	"../data/S023/S023R03.edf",
	"../data/S023/S023R07.edf",
	"../data/S023/S023R11.edf",
	"../data/S024/S024R03.edf",
	"../data/S024/S024R07.edf",
	"../data/S024/S024R11.edf",
	"../data/S025/S025R03.edf",
	"../data/S025/S025R07.edf",
	"../data/S025/S025R11.edf",

	# "../data/S026/S026R03.edf",
	# "../data/S026/S026R07.edf",
	# "../data/S026/S026R11.edf",
	# "../data/S027/S027R03.edf",
	# "../data/S027/S027R07.edf",
	# "../data/S027/S027R11.edf",
	# "../data/S028/S028R03.edf",
	# "../data/S028/S028R07.edf",
	# "../data/S028/S028R11.edf",
	# "../data/S029/S029R03.edf",
	# "../data/S029/S029R07.edf",
	# "../data/S029/S029R11.edf",
	# "../data/S030/S030R03.edf",
	# "../data/S030/S030R07.edf",
	# "../data/S030/S030R11.edf",
	# "../data/S031/S031R03.edf",
	# "../data/S031/S031R07.edf",
	# "../data/S031/S031R11.edf",
	# "../data/S032/S032R03.edf",
	# "../data/S032/S032R07.edf",
	# "../data/S032/S032R11.edf",
	# "../data/S033/S033R03.edf",
	# "../data/S033/S033R07.edf",
	# "../data/S033/S033R11.edf",
	# "../data/S034/S034R03.edf",
	# "../data/S034/S034R07.edf",
	# "../data/S034/S034R11.edf",
	# "../data/S035/S035R03.edf",
	# "../data/S035/S035R07.edf",
	# "../data/S035/S035R11.edf",
	# "../data/S036/S036R03.edf",
	# "../data/S036/S036R07.edf",
	# "../data/S036/S036R11.edf",
	# "../data/S037/S037R03.edf",
	# "../data/S037/S037R07.edf",
	# "../data/S037/S037R11.edf",
	# "../data/S038/S038R03.edf",
	# "../data/S038/S038R07.edf",
	# "../data/S038/S038R11.edf",
	# "../data/S039/S039R03.edf",
	# "../data/S039/S039R07.edf",
	# "../data/S039/S039R11.edf",
	# "../data/S040/S040R03.edf",
	# "../data/S040/S040R07.edf",
	# "../data/S040/S040R11.edf",
	# "../data/S041/S041R03.edf",
	# "../data/S041/S041R07.edf",
	# "../data/S041/S041R11.edf",
	# "../data/S042/S042R03.edf",
	# "../data/S042/S042R07.edf",
	# "../data/S042/S042R11.edf",
	# "../data/S043/S043R03.edf",
	# "../data/S043/S043R07.edf",
	# "../data/S043/S043R11.edf",
	# "../data/S044/S044R03.edf",
	# "../data/S044/S044R07.edf",
	# "../data/S044/S044R11.edf",
	# "../data/S045/S045R03.edf",
	# "../data/S045/S045R07.edf",
	# "../data/S045/S045R11.edf",
	# "../data/S046/S046R03.edf",
	# "../data/S046/S046R07.edf",
	# "../data/S046/S046R11.edf",
	# "../data/S047/S047R03.edf",
	# "../data/S047/S047R07.edf",
	# "../data/S047/S047R11.edf",
	# "../data/S048/S048R03.edf",
	# "../data/S048/S048R07.edf",
	# "../data/S048/S048R11.edf",
	# "../data/S049/S049R03.edf",
	# "../data/S049/S049R07.edf",
	# "../data/S049/S049R11.edf",
	# "../data/S050/S050R03.edf",
	# "../data/S050/S050R07.edf",
	# "../data/S050/S050R11.edf",


# run 1, eyes open
	"../data/S001/S001R01.edf",
	"../data/S002/S002R01.edf",
	"../data/S003/S003R01.edf",
	"../data/S004/S004R01.edf",
	"../data/S005/S005R01.edf",
	"../data/S006/S006R01.edf",
	"../data/S007/S007R01.edf",
	"../data/S008/S008R01.edf",
	"../data/S009/S009R01.edf",
	"../data/S010/S010R01.edf",
	"../data/S011/S011R01.edf",
	"../data/S012/S012R01.edf",
	"../data/S013/S013R01.edf",
	"../data/S014/S014R01.edf",


#run 2, eyes closed
	"../data/S001/S001R02.edf",
	"../data/S002/S002R02.edf",
	"../data/S003/S003R02.edf",
	"../data/S004/S004R02.edf",
	"../data/S005/S005R02.edf",
	"../data/S006/S006R02.edf",
	"../data/S007/S007R02.edf",
	"../data/S008/S008R02.edf",
	"../data/S009/S009R02.edf",
	"../data/S010/S010R02.edf",
	"../data/S011/S011R02.edf",
	"../data/S012/S012R02.edf",
	"../data/S013/S013R02.edf",
	"../data/S014/S014R02.edf",



# #run 4,8,12
	"../data/S001/S001R04.edf",
	"../data/S001/S001R08.edf",
	"../data/S001/S001R12.edf",
	"../data/S002/S002R04.edf",
	"../data/S002/S002R08.edf",
	"../data/S002/S002R12.edf",
	"../data/S003/S003R04.edf",
	"../data/S003/S003R08.edf",
	"../data/S003/S003R12.edf",
	"../data/S004/S004R04.edf",
	"../data/S004/S004R08.edf",
	"../data/S004/S004R12.edf",
	"../data/S005/S005R04.edf",
	"../data/S005/S005R08.edf",
	"../data/S005/S005R12.edf",
	"../data/S006/S006R04.edf",
	"../data/S006/S006R08.edf",
	"../data/S006/S006R12.edf",
	"../data/S007/S007R04.edf",
	"../data/S007/S007R08.edf",
	"../data/S007/S007R12.edf",
	"../data/S008/S008R04.edf",
	"../data/S008/S008R08.edf",
	"../data/S008/S008R12.edf",
	"../data/S009/S009R04.edf",
	"../data/S009/S009R08.edf",
	"../data/S009/S009R12.edf",
	"../data/S010/S010R04.edf",
	"../data/S010/S010R08.edf",
	"../data/S010/S010R12.edf",
	"../data/S011/S011R04.edf",
	"../data/S011/S011R08.edf",
	"../data/S011/S011R12.edf",
	"../data/S012/S012R04.edf",
	"../data/S012/S012R08.edf",
	"../data/S012/S012R12.edf",
	"../data/S013/S013R04.edf",
	"../data/S013/S013R08.edf",
	"../data/S013/S013R12.edf",
	"../data/S014/S014R04.edf",
	"../data/S014/S014R08.edf",
	"../data/S014/S014R12.edf",
	"../data/S015/S015R04.edf",
	"../data/S015/S015R08.edf",
	"../data/S015/S015R12.edf",
	"../data/S016/S016R04.edf",
	"../data/S016/S016R08.edf",
	"../data/S016/S016R12.edf",
	"../data/S017/S017R04.edf",
	"../data/S017/S017R08.edf",
	"../data/S017/S017R12.edf",
	"../data/S018/S018R04.edf",
	"../data/S018/S018R08.edf",
	"../data/S018/S018R12.edf",
	"../data/S019/S019R04.edf",
	"../data/S019/S019R08.edf",
	"../data/S019/S019R12.edf",
	"../data/S020/S020R04.edf",
	"../data/S020/S020R08.edf",
	"../data/S020/S020R12.edf",
	"../data/S021/S021R04.edf",
	"../data/S021/S021R08.edf",
	"../data/S021/S021R12.edf",
	"../data/S022/S022R04.edf",
	"../data/S022/S022R08.edf",
	"../data/S022/S022R12.edf",
	"../data/S023/S023R04.edf",
	"../data/S023/S023R08.edf",
	"../data/S023/S023R12.edf",
	"../data/S024/S024R04.edf",
	"../data/S024/S024R08.edf",
	"../data/S024/S024R12.edf",
	"../data/S025/S025R04.edf",
	"../data/S025/S025R08.edf",
	"../data/S025/S025R12.edf",

# 	"../data/S026/S026R04.edf",
# 	"../data/S026/S026R08.edf",
# 	"../data/S026/S026R12.edf",
# 	"../data/S027/S027R04.edf",
# 	"../data/S027/S027R08.edf",
# 	"../data/S027/S027R12.edf",
# 	"../data/S028/S028R04.edf",
# 	"../data/S028/S028R08.edf",
# 	"../data/S028/S028R12.edf",
# 	"../data/S029/S029R04.edf",
# 	"../data/S029/S029R08.edf",
# 	"../data/S029/S029R12.edf",
# 	"../data/S030/S030R04.edf",
# 	"../data/S030/S030R08.edf",
# 	"../data/S030/S030R12.edf",
# 	"../data/S031/S031R04.edf",
# 	"../data/S031/S031R08.edf",
# 	"../data/S031/S031R12.edf",
# 	"../data/S032/S032R04.edf",
# 	"../data/S032/S032R08.edf",
# 	"../data/S032/S032R12.edf",
# 	"../data/S033/S033R04.edf",
# 	"../data/S033/S033R08.edf",
# 	"../data/S033/S033R12.edf",
# 	"../data/S034/S034R04.edf",
# 	"../data/S034/S034R08.edf",
# 	"../data/S034/S034R12.edf",
# 	"../data/S035/S035R04.edf",
# 	"../data/S035/S035R08.edf",
# 	"../data/S035/S035R12.edf",
# 	"../data/S036/S036R04.edf",
# 	"../data/S036/S036R08.edf",
# 	"../data/S036/S036R12.edf",
# 	"../data/S037/S037R04.edf",
# 	"../data/S037/S037R08.edf",
# 	"../data/S037/S037R12.edf",
# 	"../data/S038/S038R04.edf",
# 	"../data/S038/S038R08.edf",
# 	"../data/S038/S038R12.edf",
# 	"../data/S039/S039R04.edf",
# 	"../data/S039/S039R08.edf",
# 	"../data/S039/S039R12.edf",
# 	"../data/S040/S040R04.edf",
# 	"../data/S040/S040R08.edf",
# 	"../data/S040/S040R12.edf",
# 	"../data/S041/S041R04.edf",
# 	"../data/S041/S041R08.edf",
# 	"../data/S041/S041R12.edf",
# 	"../data/S042/S042R04.edf",
# 	"../data/S042/S042R08.edf",
# 	"../data/S042/S042R12.edf",
# 	"../data/S043/S043R04.edf",
# 	"../data/S043/S043R08.edf",
# 	"../data/S043/S043R12.edf",
# 	"../data/S044/S044R04.edf",
# 	"../data/S044/S044R08.edf",
# 	"../data/S044/S044R12.edf",
# 	"../data/S045/S045R04.edf",
# 	"../data/S045/S045R08.edf",
# 	"../data/S045/S045R12.edf",
# 	"../data/S046/S046R04.edf",
# 	"../data/S046/S046R08.edf",
# 	"../data/S046/S046R12.edf",
# 	"../data/S047/S047R04.edf",
# 	"../data/S047/S047R08.edf",
# 	"../data/S047/S047R12.edf",
# 	"../data/S048/S048R04.edf",
# 	"../data/S048/S048R08.edf",
# 	"../data/S048/S048R12.edf",
# 	"../data/S049/S049R04.edf",
# 	"../data/S049/S049R08.edf",
# 	"../data/S049/S049R12.edf",
# 	"../data/S050/S050R04.edf",
# 	"../data/S050/S050R08.edf",
# 	"../data/S050/S050R12.edf",
# 	"../data/S051/S051R04.edf",
# 	"../data/S051/S051R08.edf",
# 	"../data/S051/S051R12.edf",
# 	"../data/S052/S052R04.edf",
# 	"../data/S052/S052R08.edf",
# 	"../data/S052/S052R12.edf",
# 	"../data/S053/S053R04.edf",
# 	"../data/S053/S053R08.edf",
# 	"../data/S053/S053R12.edf",
# 	"../data/S054/S054R04.edf",
# 	"../data/S054/S054R08.edf",
# 	"../data/S054/S054R12.edf",
# 	"../data/S055/S055R04.edf",
# 	"../data/S055/S055R08.edf",
# 	"../data/S055/S055R12.edf",


	# "../data/S056/S056R04.edf",
	# "../data/S056/S056R08.edf",
	# "../data/S056/S056R12.edf",
	# "../data/S057/S057R04.edf",
	# "../data/S057/S057R08.edf",
	# "../data/S057/S057R12.edf",
	# "../data/S058/S058R04.edf",
	# "../data/S058/S058R08.edf",
	# "../data/S058/S058R12.edf",
	# "../data/S059/S059R04.edf",
	# "../data/S059/S059R08.edf",
	# "../data/S059/S059R12.edf",
	# "../data/S060/S060R04.edf",
	# "../data/S060/S060R08.edf",
	# "../data/S060/S060R12.edf",
	# "../data/S061/S061R04.edf",
	# "../data/S061/S061R08.edf",
	# "../data/S061/S061R12.edf",
	# "../data/S062/S062R04.edf",
	# "../data/S062/S062R08.edf",
	# "../data/S062/S062R12.edf",
	# "../data/S063/S063R04.edf",
	# "../data/S063/S063R08.edf",
	# "../data/S063/S063R12.edf",
	# "../data/S064/S064R04.edf",
	# "../data/S064/S064R08.edf",
	# "../data/S064/S064R12.edf",
	# "../data/S065/S065R04.edf",
	# "../data/S065/S065R08.edf",
	# "../data/S065/S065R12.edf",
	# "../data/S066/S066R04.edf",
	# "../data/S066/S066R08.edf",
	# "../data/S066/S066R12.edf",
	# "../data/S067/S067R04.edf",
	# "../data/S067/S067R08.edf",
	# "../data/S067/S067R12.edf",
	# "../data/S068/S068R04.edf",
	# "../data/S068/S068R08.edf",
	# "../data/S068/S068R12.edf",
	# "../data/S069/S069R04.edf",
	# "../data/S069/S069R08.edf",
	# "../data/S069/S069R12.edf",
	# "../data/S070/S070R04.edf",
	# "../data/S070/S070R08.edf",
	# "../data/S070/S070R12.edf",
	# "../data/S071/S071R04.edf",
	# "../data/S071/S071R08.edf",
	# "../data/S071/S071R12.edf",
	# "../data/S072/S072R04.edf",
	# "../data/S072/S072R08.edf",
	# "../data/S072/S072R12.edf",
	# "../data/S073/S073R04.edf",
	# "../data/S073/S073R08.edf",
	# "../data/S073/S073R12.edf",
	# "../data/S074/S074R04.edf",
	# "../data/S074/S074R08.edf",
	# "../data/S074/S074R12.edf",
	# "../data/S075/S075R04.edf",
	# "../data/S075/S075R08.edf",
	# "../data/S075/S075R12.edf",
	# "../data/S076/S076R04.edf",
	# "../data/S076/S076R08.edf",
	# "../data/S076/S076R12.edf",
	# "../data/S077/S077R04.edf",
	# "../data/S077/S077R08.edf",
	# "../data/S077/S077R12.edf",
	# "../data/S078/S078R04.edf",
	# "../data/S078/S078R08.edf",
	# "../data/S078/S078R12.edf",
	# "../data/S079/S079R04.edf",
	# "../data/S079/S079R08.edf",
	# "../data/S079/S079R12.edf",
	# "../data/S080/S080R04.edf",
	# "../data/S080/S080R08.edf",
	# "../data/S080/S080R12.edf",
	# "../data/S081/S081R04.edf",
	# "../data/S081/S081R08.edf",
	# "../data/S081/S081R12.edf",
	# "../data/S082/S082R04.edf",
	# "../data/S082/S082R08.edf",
	# "../data/S082/S082R12.edf",
	# "../data/S083/S083R04.edf",
	# "../data/S083/S083R08.edf",
	# "../data/S083/S083R12.edf",
	# "../data/S084/S084R04.edf",
	# "../data/S084/S084R08.edf",
	# "../data/S084/S084R12.edf",
	# "../data/S085/S085R04.edf",
	# "../data/S085/S085R08.edf",
	# "../data/S085/S085R12.edf",
	# "../data/S086/S086R04.edf",
	# "../data/S086/S086R08.edf",
	# "../data/S086/S086R12.edf",
	# "../data/S087/S087R04.edf",
	# "../data/S087/S087R08.edf",
	# "../data/S087/S087R12.edf",

	#this from 88 have different frequency
	# "../data/S088/S088R04.edf",
	# "../data/S088/S088R08.edf",
	# "../data/S088/S088R12.edf",
	# "../data/S089/S089R04.edf",
	# "../data/S089/S089R08.edf",
	# "../data/S089/S089R12.edf",
	# "../data/S090/S090R04.edf",
	# "../data/S090/S090R08.edf",
	# "../data/S090/S090R12.edf",
	# "../data/S091/S091R04.edf",
	# "../data/S091/S091R08.edf",
	# "../data/S091/S091R12.edf",
	# "../data/S092/S092R04.edf",
	# "../data/S092/S092R08.edf",
	# "../data/S092/S092R12.edf",
	# "../data/S093/S093R04.edf",
	# "../data/S093/S093R08.edf",
	# "../data/S093/S093R12.edf",
	# "../data/S094/S094R04.edf",
	# "../data/S094/S094R08.edf",
	# "../data/S094/S094R12.edf",
	# "../data/S095/S095R04.edf",
	# "../data/S095/S095R08.edf",
	# "../data/S095/S095R12.edf",
	# "../data/S096/S096R04.edf",
	# "../data/S096/S096R08.edf",
	# "../data/S096/S096R12.edf",
	# "../data/S097/S097R04.edf",
	# "../data/S097/S097R08.edf",
	# "../data/S097/S097R12.edf",
	# "../data/S098/S098R04.edf",
	# "../data/S098/S098R08.edf",
	# "../data/S098/S098R12.edf",
	# "../data/S099/S099R04.edf",
	# "../data/S099/S099R08.edf",
	# "../data/S099/S099R12.edf",
	# "../data/S100/S100R04.edf",
	# "../data/S100/S100R08.edf",
	# "../data/S100/S100R12.edf",
	# "../data/S101/S101R04.edf",
	# "../data/S101/S101R08.edf",
	# "../data/S101/S101R12.edf",
	# "../data/S102/S102R04.edf",
	# "../data/S102/S102R08.edf",
	# "../data/S102/S102R12.edf",
	# "../data/S103/S103R04.edf",
	# "../data/S103/S103R08.edf",
	# "../data/S103/S103R12.edf",
	# "../data/S104/S104R04.edf",
	# "../data/S104/S104R08.edf",
	# "../data/S104/S104R12.edf",
	# "../data/S105/S105R04.edf",
	# "../data/S105/S105R08.edf",
	# "../data/S105/S105R12.edf",
	# "../data/S106/S106R04.edf",
	# "../data/S106/S106R08.edf",
	# "../data/S106/S106R12.edf",
	# "../data/S107/S107R04.edf",
	# "../data/S107/S107R08.edf",
	# "../data/S107/S107R12.edf",
	# "../data/S108/S108R04.edf",
	# "../data/S108/S108R08.edf",
	# "../data/S108/S108R12.edf",
	# "../data/S109/S109R04.edf",
	# "../data/S109/S109R08.edf",
	# "../data/S109/S109R12.edf",



	#5,9,13 all subject, 3 runs
	"../data/S001/S001R05.edf",
	"../data/S001/S001R09.edf",
	"../data/S001/S001R13.edf",
	"../data/S002/S002R05.edf",
	"../data/S002/S002R09.edf",
	"../data/S002/S002R13.edf",
	"../data/S003/S003R05.edf",
	"../data/S003/S003R09.edf",
	"../data/S003/S003R13.edf",
	"../data/S004/S004R05.edf",
	"../data/S004/S004R09.edf",
	"../data/S004/S004R13.edf",
	"../data/S005/S005R05.edf",
	"../data/S005/S005R09.edf",
	"../data/S005/S005R13.edf",
	"../data/S006/S006R05.edf",
	"../data/S006/S006R09.edf",
	"../data/S006/S006R13.edf",
	"../data/S007/S007R05.edf",
	"../data/S007/S007R09.edf",
	"../data/S007/S007R13.edf",
	"../data/S008/S008R05.edf",
	"../data/S008/S008R09.edf",
	"../data/S008/S008R13.edf",
	"../data/S009/S009R05.edf",
	"../data/S009/S009R09.edf",
	"../data/S009/S009R13.edf",
	"../data/S010/S010R05.edf",
	"../data/S010/S010R09.edf",
	"../data/S010/S010R13.edf",
	"../data/S011/S011R05.edf",
	"../data/S011/S011R09.edf",
	"../data/S011/S011R13.edf",
	"../data/S012/S012R05.edf",
	"../data/S012/S012R09.edf",
	"../data/S012/S012R13.edf",
	"../data/S013/S013R05.edf",
	"../data/S013/S013R09.edf",
	"../data/S013/S013R13.edf",
	"../data/S014/S014R05.edf",
	"../data/S014/S014R09.edf",
	"../data/S014/S014R13.edf",
	"../data/S015/S015R05.edf",
	"../data/S015/S015R09.edf",
	"../data/S015/S015R13.edf",
	"../data/S016/S016R05.edf",
	"../data/S016/S016R09.edf",
	"../data/S016/S016R13.edf",
	"../data/S017/S017R05.edf",
	"../data/S017/S017R09.edf",
	"../data/S017/S017R13.edf",
	"../data/S018/S018R05.edf",
	"../data/S018/S018R09.edf",
	"../data/S018/S018R13.edf",
	"../data/S019/S019R05.edf",
	"../data/S019/S019R09.edf",
	"../data/S019/S019R13.edf",
	"../data/S020/S020R05.edf",
	"../data/S020/S020R09.edf",
	"../data/S020/S020R13.edf",
	"../data/S021/S021R05.edf",
	"../data/S021/S021R09.edf",
	"../data/S021/S021R13.edf",
	"../data/S022/S022R05.edf",
	"../data/S022/S022R09.edf",
	"../data/S022/S022R13.edf",
	"../data/S023/S023R05.edf",
	"../data/S023/S023R09.edf",
	"../data/S023/S023R13.edf",
	"../data/S024/S024R05.edf",
	"../data/S024/S024R09.edf",
	"../data/S024/S024R13.edf",
	"../data/S025/S025R05.edf",
	"../data/S025/S025R09.edf",

	# "../data/S025/S025R13.edf",
	# "../data/S026/S026R05.edf",
	# "../data/S026/S026R09.edf",
	# "../data/S026/S026R13.edf",
	# "../data/S027/S027R05.edf",
	# "../data/S027/S027R09.edf",
	# "../data/S027/S027R13.edf",
	# "../data/S028/S028R05.edf",
	# "../data/S028/S028R09.edf",
	# "../data/S028/S028R13.edf",
	# "../data/S029/S029R05.edf",
	# "../data/S029/S029R09.edf",
	# "../data/S029/S029R13.edf",
	# "../data/S030/S030R05.edf",
	# "../data/S030/S030R09.edf",
	# "../data/S030/S030R13.edf",
	# "../data/S031/S031R05.edf",
	# "../data/S031/S031R09.edf",
	# "../data/S031/S031R13.edf",
	# "../data/S032/S032R05.edf",
	# "../data/S032/S032R09.edf",
	# "../data/S032/S032R13.edf",
	# "../data/S033/S033R05.edf",
	# "../data/S033/S033R09.edf",
	# "../data/S033/S033R13.edf",
	# "../data/S034/S034R05.edf",
	# "../data/S034/S034R09.edf",
	# "../data/S034/S034R13.edf",
	# "../data/S035/S035R05.edf",
	# "../data/S035/S035R09.edf",
	# "../data/S035/S035R13.edf",
	# "../data/S036/S036R05.edf",
	# "../data/S036/S036R09.edf",
	# "../data/S036/S036R13.edf",
	# "../data/S037/S037R05.edf",
	# "../data/S037/S037R09.edf",
	# "../data/S037/S037R13.edf",
	# "../data/S038/S038R05.edf",
	# "../data/S038/S038R09.edf",
	# "../data/S038/S038R13.edf",
	# "../data/S039/S039R05.edf",
	# "../data/S039/S039R09.edf",
	# "../data/S039/S039R13.edf",
	# "../data/S040/S040R05.edf",
	# "../data/S040/S040R09.edf",
	# "../data/S040/S040R13.edf",
	# "../data/S041/S041R05.edf",
	# "../data/S041/S041R09.edf",
	# "../data/S041/S041R13.edf",
	# "../data/S042/S042R05.edf",
	# "../data/S042/S042R09.edf",
	# "../data/S042/S042R13.edf",
	# "../data/S043/S043R05.edf",
	# "../data/S043/S043R09.edf",
	# "../data/S043/S043R13.edf",
	# "../data/S044/S044R05.edf",
	# "../data/S044/S044R09.edf",
	# "../data/S044/S044R13.edf",
	# "../data/S045/S045R05.edf",
	# "../data/S045/S045R09.edf",
	# "../data/S045/S045R13.edf",
	# "../data/S046/S046R05.edf",
	# "../data/S046/S046R09.edf",
	# "../data/S046/S046R13.edf",
	# "../data/S047/S047R05.edf",
	# "../data/S047/S047R09.edf",
	# "../data/S047/S047R13.edf",
	# "../data/S048/S048R05.edf",
	# "../data/S048/S048R09.edf",
	# "../data/S048/S048R13.edf",
	# "../data/S049/S049R05.edf",
	# "../data/S049/S049R09.edf",
	# "../data/S049/S049R13.edf",
	# "../data/S050/S050R05.edf",
	# "../data/S050/S050R09.edf",
	# "../data/S050/S050R13.edf",
	# "../data/S051/S051R05.edf",
	# "../data/S051/S051R09.edf",
	# "../data/S051/S051R13.edf",
	# "../data/S052/S052R05.edf",
	# "../data/S052/S052R09.edf",
	# "../data/S052/S052R13.edf",
	# "../data/S053/S053R05.edf",
	# "../data/S053/S053R09.edf",
	# "../data/S053/S053R13.edf",
	# "../data/S054/S054R05.edf",
	# "../data/S054/S054R09.edf",
	# "../data/S054/S054R13.edf",
	# "../data/S055/S055R05.edf",
	# "../data/S055/S055R09.edf",
	# "../data/S055/S055R13.edf",
	# "../data/S056/S056R05.edf",
	# "../data/S056/S056R09.edf",
	# "../data/S056/S056R13.edf",
	# "../data/S057/S057R05.edf",
	# "../data/S057/S057R09.edf",
	# "../data/S057/S057R13.edf",
	# "../data/S058/S058R05.edf",
	# "../data/S058/S058R09.edf",
	# "../data/S058/S058R13.edf",
	# "../data/S059/S059R05.edf",
	# "../data/S059/S059R09.edf",
	# "../data/S059/S059R13.edf",

	# "../data/S060/S060R05.edf",
	# "../data/S060/S060R09.edf",
	# "../data/S060/S060R13.edf",
	# "../data/S061/S061R05.edf",
	# "../data/S061/S061R09.edf",
	# "../data/S061/S061R13.edf",
	# "../data/S062/S062R05.edf",
	# "../data/S062/S062R09.edf",
	# "../data/S062/S062R13.edf",
	# "../data/S063/S063R05.edf",
	# "../data/S063/S063R09.edf",
	# "../data/S063/S063R13.edf",
	# "../data/S064/S064R05.edf",
	# "../data/S064/S064R09.edf",
	# "../data/S064/S064R13.edf",
	# "../data/S065/S065R05.edf",
	# "../data/S065/S065R09.edf",
	# "../data/S065/S065R13.edf",
	# "../data/S066/S066R05.edf",
	# "../data/S066/S066R09.edf",
	# "../data/S066/S066R13.edf",
	# "../data/S067/S067R05.edf",
	# "../data/S067/S067R09.edf",
	# "../data/S067/S067R13.edf",
	# "../data/S068/S068R05.edf",
	# "../data/S068/S068R09.edf",
	# "../data/S068/S068R13.edf",
	# "../data/S069/S069R05.edf",
	# "../data/S069/S069R09.edf",
	# "../data/S069/S069R13.edf",
	# "../data/S070/S070R05.edf",
	# "../data/S070/S070R09.edf",
	# "../data/S070/S070R13.edf",
	# "../data/S071/S071R05.edf",
	# "../data/S071/S071R09.edf",
	# "../data/S071/S071R13.edf",
	# "../data/S072/S072R05.edf",
	# "../data/S072/S072R09.edf",
	# "../data/S072/S072R13.edf",
	# "../data/S073/S073R05.edf",
	# "../data/S073/S073R09.edf",
	# "../data/S073/S073R13.edf",
	# "../data/S074/S074R05.edf",
	# "../data/S074/S074R09.edf",
	# "../data/S074/S074R13.edf",
	# "../data/S075/S075R05.edf",
	# "../data/S075/S075R09.edf",
	# "../data/S075/S075R13.edf",
	# "../data/S076/S076R05.edf",
	# "../data/S076/S076R09.edf",
	# "../data/S076/S076R13.edf",
	# "../data/S077/S077R05.edf",
	# "../data/S077/S077R09.edf",
	# "../data/S077/S077R13.edf",
	# "../data/S078/S078R05.edf",
	# "../data/S078/S078R09.edf",
	# "../data/S078/S078R13.edf",
	# "../data/S079/S079R05.edf",
	# "../data/S079/S079R09.edf",
	# "../data/S079/S079R13.edf",
	# "../data/S080/S080R05.edf",
	# "../data/S080/S080R09.edf",
	# "../data/S080/S080R13.edf",
	# "../data/S081/S081R05.edf",
	# "../data/S081/S081R09.edf",
	# "../data/S081/S081R13.edf",
	# "../data/S082/S082R05.edf",
	# "../data/S082/S082R09.edf",
	# "../data/S082/S082R13.edf",
	# "../data/S083/S083R05.edf",
	# "../data/S083/S083R09.edf",
	# "../data/S083/S083R13.edf",
	# "../data/S084/S084R05.edf",
	# "../data/S084/S084R09.edf",
	# "../data/S084/S084R13.edf",
	# "../data/S085/S085R05.edf",
	# "../data/S085/S085R09.edf",
	# "../data/S085/S085R13.edf",
	# "../data/S086/S086R05.edf",
	# "../data/S086/S086R09.edf",
	# "../data/S086/S086R13.edf",
	# "../data/S087/S087R05.edf",
	# "../data/S087/S087R09.edf",
	# "../data/S087/S087R13.edf",

	#wrong freq
	# "../data/S088/S088R05.edf",
	# "../data/S088/S088R09.edf",
	# "../data/S088/S088R13.edf",
	# "../data/S089/S089R05.edf",
	# "../data/S089/S089R09.edf",
	# "../data/S089/S089R13.edf",
	# "../data/S090/S090R05.edf",
	# "../data/S090/S090R09.edf",
	# "../data/S090/S090R13.edf",
	# "../data/S091/S091R05.edf",
	# "../data/S091/S091R09.edf",
	# "../data/S091/S091R13.edf",
	# "../data/S092/S092R05.edf",
	# "../data/S092/S092R09.edf",
	# "../data/S092/S092R13.edf",
	# "../data/S093/S093R05.edf",
	# "../data/S093/S093R09.edf",
	# "../data/S093/S093R13.edf",
	# "../data/S094/S094R05.edf",
	# "../data/S094/S094R09.edf",
	# "../data/S094/S094R13.edf",
	# "../data/S095/S095R05.edf",
	# "../data/S095/S095R09.edf",
	# "../data/S095/S095R13.edf",
	# "../data/S096/S096R05.edf",
	# "../data/S096/S096R09.edf",
	# "../data/S096/S096R13.edf",
	# "../data/S097/S097R05.edf",
	# "../data/S097/S097R09.edf",
	# "../data/S097/S097R13.edf",
	# "../data/S098/S098R05.edf",
	# "../data/S098/S098R09.edf",
	# "../data/S098/S098R13.edf",
	# "../data/S099/S099R05.edf",
	# "../data/S099/S099R09.edf",
	# "../data/S099/S099R13.edf",
	# "../data/S100/S100R05.edf",
	# "../data/S100/S100R09.edf",
	# "../data/S100/S100R13.edf",
	# "../data/S101/S101R05.edf",
	# "../data/S101/S101R09.edf",
	# "../data/S101/S101R13.edf",
	# "../data/S102/S102R05.edf",
	# "../data/S102/S102R09.edf",
	# "../data/S102/S102R13.edf",
	# "../data/S103/S103R05.edf",
	# "../data/S103/S103R09.edf",
	# "../data/S103/S103R13.edf",
	# "../data/S104/S104R05.edf",
	# "../data/S104/S104R09.edf",
	# "../data/S104/S104R13.edf",
	# "../data/S105/S105R05.edf",
	# "../data/S105/S105R09.edf",
	# "../data/S105/S105R13.edf",
	# "../data/S106/S106R05.edf",
	# "../data/S106/S106R09.edf",
	# "../data/S106/S106R13.edf",
	# "../data/S107/S107R05.edf",
	# "../data/S107/S107R09.edf",
	# "../data/S107/S107R13.edf",
	# "../data/S108/S108R05.edf",
	# "../data/S108/S108R09.edf",
	# "../data/S108/S108R13.edf",
	# "../data/S109/S109R05.edf",
	# "../data/S109/S109R09.edf",
	# "../data/S109/S109R13.edf"


	# "../data/S022/S022R05.edf",
	# "../data/S022/S022R09.edf",
	# "../data/S022/S022R13.edf",
	# "../data/S023/S023R05.edf",
	# "../data/S023/S023R09.edf",
	# "../data/S023/S023R13.edf",
	# "../data/S024/S024R05.edf",
	# "../data/S024/S024R09.edf",
	# "../data/S024/S024R13.edf",
	# "../data/S025/S025R05.edf",
	# "../data/S025/S025R09.edf",
	# "../data/S025/S025R13.edf",
	# "../data/S026/S026R05.edf",
	# "../data/S026/S026R09.edf",
	# "../data/S026/S026R13.edf",
	# "../data/S027/S027R05.edf",
	# "../data/S027/S027R09.edf",
	# "../data/S027/S027R13.edf",
	# "../data/S028/S028R05.edf",
	# "../data/S028/S028R09.edf",
	# "../data/S028/S028R13.edf",
	# "../data/S029/S029R05.edf",
	# "../data/S029/S029R09.edf",
	# "../data/S029/S029R13.edf",
	# "../data/S030/S030R05.edf",
	# "../data/S030/S030R09.edf",
	# "../data/S030/S030R13.edf",
	# "../data/S031/S031R05.edf",
	# "../data/S031/S031R09.edf",
	# "../data/S031/S031R13.edf",
	# "../data/S032/S032R05.edf",
	# "../data/S032/S032R09.edf",
	# "../data/S032/S032R13.edf",
	# "../data/S033/S033R05.edf",
	# "../data/S033/S033R09.edf",
	# "../data/S033/S033R13.edf",
	# "../data/S034/S034R05.edf",
	# "../data/S034/S034R09.edf",
	# "../data/S034/S034R13.edf",
	# "../data/S035/S035R05.edf",
	# "../data/S035/S035R09.edf",
	# "../data/S035/S035R13.edf",
	# "../data/S036/S036R05.edf",
	# "../data/S036/S036R09.edf",
	# "../data/S036/S036R13.edf",
	# "../data/S037/S037R05.edf",
	# "../data/S037/S037R09.edf",
	# "../data/S037/S037R13.edf",
	# "../data/S038/S038R05.edf",
	# "../data/S038/S038R09.edf",
	# "../data/S038/S038R13.edf",
	# "../data/S039/S039R05.edf",
	# "../data/S039/S039R09.edf",
	# "../data/S039/S039R13.edf",
	# "../data/S040/S040R05.edf",
	# "../data/S040/S040R09.edf",
	# "../data/S040/S040R13.edf",
	# "../data/S041/S041R05.edf",
	# "../data/S041/S041R09.edf",
	# "../data/S041/S041R13.edf",
	# "../data/S042/S042R05.edf",
	# "../data/S042/S042R09.edf",
	# "../data/S042/S042R13.edf",
	# "../data/S043/S043R05.edf",
	# "../data/S043/S043R09.edf",
	# "../data/S043/S043R13.edf",
	# "../data/S044/S044R05.edf",
	# "../data/S044/S044R09.edf",
	# "../data/S044/S044R13.edf",
	# "../data/S045/S045R05.edf",
	# "../data/S045/S045R09.edf",
	# "../data/S045/S045R13.edf",
	# "../data/S046/S046R05.edf",
	# "../data/S046/S046R09.edf",
	# "../data/S046/S046R13.edf",
	# "../data/S047/S047R05.edf",
	# "../data/S047/S047R09.edf",
	# "../data/S047/S047R13.edf",
	# "../data/S048/S048R05.edf",
	# "../data/S048/S048R09.edf",
	# "../data/S048/S048R13.edf",
	# "../data/S049/S049R05.edf",
	# "../data/S049/S049R09.edf",
	# "../data/S049/S049R13.edf",
	# "../data/S050/S050R05.edf",
	# "../data/S050/S050R09.edf",
	# "../data/S050/S050R13.edf",
	# "../data/S051/S051R05.edf",
	# "../data/S051/S051R09.edf",
	# "../data/S051/S051R13.edf",
	# "../data/S052/S052R05.edf",
	# "../data/S052/S052R09.edf",
	# "../data/S052/S052R13.edf",
	# "../data/S053/S053R05.edf",
	# "../data/S053/S053R09.edf",
	# "../data/S053/S053R13.edf",
	# "../data/S054/S054R05.edf",
	# "../data/S054/S054R09.edf",
	# "../data/S054/S054R13.edf",
	# "../data/S055/S055R05.edf",
	# "../data/S055/S055R09.edf",
	# "../data/S055/S055R13.edf",
	# "../data/S056/S056R05.edf",
	# "../data/S056/S056R09.edf",
	# "../data/S056/S056R13.edf",
	# "../data/S057/S057R05.edf",
	# "../data/S057/S057R09.edf",
	# "../data/S057/S057R13.edf",
	# "../data/S058/S058R05.edf",
	# "../data/S058/S058R09.edf",
	# "../data/S058/S058R13.edf",

	#6,10,14
	"../data/S001/S001R06.edf",
	"../data/S001/S001R10.edf",
	"../data/S001/S001R14.edf",
	"../data/S002/S002R06.edf",
	"../data/S002/S002R10.edf",
	"../data/S002/S002R14.edf",
	"../data/S003/S003R06.edf",
	"../data/S003/S003R10.edf",
	"../data/S003/S003R14.edf",
	"../data/S004/S004R06.edf",
	"../data/S004/S004R10.edf",
	"../data/S004/S004R14.edf",
	"../data/S005/S005R06.edf",
	"../data/S005/S005R10.edf",
	"../data/S005/S005R14.edf",
	"../data/S006/S006R06.edf",
	"../data/S006/S006R10.edf",
	"../data/S006/S006R14.edf",
	"../data/S007/S007R06.edf",
	"../data/S007/S007R10.edf",
	"../data/S007/S007R14.edf",
	"../data/S008/S008R06.edf",
	"../data/S008/S008R10.edf",
	"../data/S008/S008R14.edf",
	"../data/S009/S009R06.edf",
	"../data/S009/S009R10.edf",
	"../data/S009/S009R14.edf",
	"../data/S010/S010R06.edf",
	"../data/S010/S010R10.edf",
	"../data/S010/S010R14.edf",
	"../data/S011/S011R06.edf",
	"../data/S011/S011R10.edf",
	"../data/S011/S011R14.edf",
	"../data/S012/S012R06.edf",
	"../data/S012/S012R10.edf",
	"../data/S012/S012R14.edf",
	"../data/S013/S013R06.edf",
	"../data/S013/S013R10.edf",
	"../data/S013/S013R14.edf",
	"../data/S014/S014R06.edf",
	"../data/S014/S014R10.edf",
	"../data/S014/S014R14.edf",
	"../data/S015/S015R06.edf",
	"../data/S015/S015R10.edf",
	"../data/S015/S015R14.edf",
	"../data/S016/S016R06.edf",
	"../data/S016/S016R10.edf",
	"../data/S016/S016R14.edf",
	"../data/S017/S017R06.edf",
	"../data/S017/S017R10.edf",
	"../data/S017/S017R14.edf",
	"../data/S018/S018R06.edf",
	"../data/S018/S018R10.edf",
	"../data/S018/S018R14.edf",
	"../data/S019/S019R06.edf",
	"../data/S019/S019R10.edf",
	"../data/S019/S019R14.edf",
	"../data/S020/S020R06.edf",
	"../data/S020/S020R10.edf",
	"../data/S020/S020R14.edf",
	"../data/S021/S021R06.edf",
	"../data/S021/S021R10.edf",
	"../data/S021/S021R14.edf",

	# "../data/S022/S022R06.edf",
	# "../data/S022/S022R10.edf",
	# "../data/S022/S022R14.edf",
	# "../data/S023/S023R06.edf",
	# "../data/S023/S023R10.edf",
	# "../data/S023/S023R14.edf",
	# "../data/S024/S024R06.edf",
	# "../data/S024/S024R10.edf",
	# "../data/S024/S024R14.edf",
	# "../data/S025/S025R06.edf",
	# "../data/S025/S025R10.edf",
	# "../data/S025/S025R14.edf",
	# "../data/S026/S026R06.edf",
	# "../data/S026/S026R10.edf",
	# "../data/S026/S026R14.edf",




	# "../data/S027/S027R06.edf",
	# "../data/S027/S027R10.edf",
	# "../data/S027/S027R14.edf",
	# "../data/S028/S028R06.edf",
	# "../data/S028/S028R10.edf",
	# "../data/S028/S028R14.edf",
	# "../data/S029/S029R06.edf",
	# "../data/S029/S029R10.edf",
	# "../data/S029/S029R14.edf",
	# "../data/S030/S030R06.edf",
	# "../data/S030/S030R10.edf",
	# "../data/S030/S030R14.edf",
	# "../data/S031/S031R06.edf",
	# "../data/S031/S031R10.edf",
	# "../data/S031/S031R14.edf",
	# "../data/S032/S032R06.edf",
	# "../data/S032/S032R10.edf",
	# "../data/S032/S032R14.edf",
	# "../data/S033/S033R06.edf",
	# "../data/S033/S033R10.edf",
	# "../data/S033/S033R14.edf",
	# "../data/S034/S034R06.edf",
	# "../data/S034/S034R10.edf",
	# "../data/S034/S034R14.edf",
	# "../data/S035/S035R06.edf",
	# "../data/S035/S035R10.edf",
	# "../data/S035/S035R14.edf",
	# "../data/S036/S036R06.edf",
	# "../data/S036/S036R10.edf",
	# "../data/S036/S036R14.edf",
	# "../data/S037/S037R06.edf",
	# "../data/S037/S037R10.edf",
	# "../data/S037/S037R14.edf",
	# "../data/S038/S038R06.edf",
	# "../data/S038/S038R10.edf",
	# "../data/S038/S038R14.edf",
	# "../data/S039/S039R06.edf",
	# "../data/S039/S039R10.edf",
	# "../data/S039/S039R14.edf",
	# "../data/S040/S040R06.edf",
	# "../data/S040/S040R10.edf",
	# "../data/S040/S040R14.edf",
	# "../data/S041/S041R06.edf",
	# "../data/S041/S041R10.edf",
	# "../data/S041/S041R14.edf",
	# "../data/S042/S042R06.edf",
	# "../data/S042/S042R10.edf",
	# "../data/S042/S042R14.edf",
	# "../data/S043/S043R06.edf",
	# "../data/S043/S043R10.edf",
	# "../data/S043/S043R14.edf",
	# "../data/S044/S044R06.edf",
	# "../data/S044/S044R10.edf",
	# "../data/S044/S044R14.edf",
	# "../data/S045/S045R06.edf",
	# "../data/S045/S045R10.edf",
	# "../data/S045/S045R14.edf",
	# "../data/S046/S046R06.edf",
	# "../data/S046/S046R10.edf",
	# "../data/S046/S046R14.edf",
	# "../data/S047/S047R06.edf",
	# "../data/S047/S047R10.edf",
	# "../data/S047/S047R14.edf",
	# "../data/S048/S048R06.edf",
	# "../data/S048/S048R10.edf",
	# "../data/S048/S048R14.edf",
	# "../data/S049/S049R06.edf",
	# "../data/S049/S049R10.edf",
	# "../data/S049/S049R14.edf",
	# "../data/S050/S050R06.edf",
	# "../data/S050/S050R10.edf",
	# "../data/S050/S050R14.edf",
	# "../data/S051/S051R06.edf",
	# "../data/S051/S051R10.edf",
	# "../data/S051/S051R14.edf",
	# "../data/S052/S052R06.edf",
	# "../data/S052/S052R10.edf",
	# "../data/S052/S052R14.edf",
	# "../data/S053/S053R06.edf",
	# "../data/S053/S053R10.edf",
	# "../data/S053/S053R14.edf",
	# "../data/S054/S054R06.edf",
	# "../data/S054/S054R10.edf",
	# "../data/S054/S054R14.edf",
	# "../data/S055/S055R06.edf",
	# "../data/S055/S055R10.edf",
	# "../data/S055/S055R14.edf",
	# "../data/S056/S056R06.edf",
	# "../data/S056/S056R10.edf",
	# "../data/S056/S056R14.edf",
	# "../data/S057/S057R06.edf",
	# "../data/S057/S057R10.edf",
	# "../data/S057/S057R14.edf",
	# "../data/S058/S058R06.edf",
	# "../data/S058/S058R10.edf",
	# "../data/S058/S058R14.edf",

	
#------------------------
		
		# "../data/S078/S078R11.edf",
#------------------------



		#different freq
		# "../data/S088/S088R03.edf",
		# "../data/S088/S088R07.edf",
		# "../data/S088/S088R11.edf",


		# "../data/S098/S098R03.edf",
		# "../data/S098/S098R07.edf",
		# "../data/S098/S098R11.edf",


		# "../data/S018/S018R03.edf",
		# "../data/S018/S018R07.edf",
		# "../data/S018/S018R11.edf",


		# "../data/S018/S018R03.edf",
		# "../data/S018/S018R07.edf",
		# "../data/S018/S018R11.edf",



		# "../data/S018/S018R04.edf",
		# "../data/S018/S018R08.edf",
		# "../data/S018/S018R12.edf",

		# "../data/S018/S018R05.edf",
		# "../data/S018/S018R09.edf",
		# "../data/S018/S018R13.edf",

		# "../data/S018/S018R06.edf",
		# "../data/S018/S018R10.edf",
		# "../data/S018/S018R14.edf",

		# "../data/S028/S028R03.edf",
		# "../data/S028/S028R07.edf",
		# "../data/S028/S028R11.edf",

		# "../data/S028/S028R04.edf",
		# "../data/S028/S028R08.edf",
		# "../data/S028/S028R12.edf",

		# "../data/S028/S028R05.edf",
		# "../data/S028/S028R09.edf",
		# "../data/S028/S028R13.edf",

		# "../data/S028/S028R06.edf",
		# "../data/S028/S028R10.edf",
		# "../data/S028/S028R14.edf",



		# "../data/S038/S038R03.edf",
		# "../data/S038/S038R07.edf",
		# "../data/S038/S038R11.edf",
		# "../data/S038/S038R04.edf",
		# "../data/S038/S038R08.edf",
		# "../data/S038/S038R12.edf",
		# "../data/S038/S038R05.edf",
		# "../data/S038/S038R09.edf",
		# "../data/S038/S038R13.edf",
		# "../data/S038/S038R06.edf",
		# "../data/S038/S038R10.edf",
		# "../data/S038/S038R14.edf",

		# "../data/S048/S048R03.edf",
		# "../data/S048/S048R07.edf",
		# "../data/S048/S048R11.edf",
		# "../data/S048/S048R04.edf",
		# "../data/S048/S048R08.edf",
		# "../data/S048/S048R12.edf",
		# "../data/S048/S048R05.edf",
		# "../data/S048/S048R09.edf",
		# "../data/S048/S048R13.edf",
		# "../data/S048/S048R06.edf",
		# "../data/S048/S048R10.edf",
		# "../data/S048/S048R14.edf",

		# "../data/S104/S104R03.edf",
		# "../data/S091/S091R11.edf",
		# "../data/S091/S091R03.edf",
		# "../data/S091/S091R07.edf",
		# "../data/S082/S082R11.edf",
		# "../data/S082/S082R03.edf",
		# "../data/S082/S082R07.edf",
		# "../data/S048/S048R03.edf",
		# "../data/S048/S048R11.edf",
		# "../data/S048/S048R07.edf",
		# "../data/S038/S038R11.edf",
		# "../data/S038/S038R07.edf",
		# "../data/S038/S038R03.edf",
		# "../data/S040/S040R03.edf",
		# "../data/S040/S040R07.edf",
		# "../data/S040/S040R11.edf",
		# "../data/S093/S093R07.edf",
		# "../data/S093/S093R11.edf",
		# "../data/S093/S093R03.edf",
		# "../data/S047/S047R11.edf",
		# "../data/S047/S047R07.edf",
		# "../data/S047/S047R03.edf",
		# "../data/S102/S102R07.edf",
		# "../data/S102/S102R03.edf",
		# "../data/S102/S102R11.edf",
		# "../data/S083/S083R11.edf",
		# "../data/S083/S083R03.edf",
		# "../data/S083/S083R07.edf",
		# "../data/S034/S034R07.edf",
		# "../data/S034/S034R03.edf",
		# "../data/S034/S034R11.edf",
		# "../data/S041/S041R07.edf",
		# "../data/S041/S041R03.edf",
		# "../data/S041/S041R11.edf",
		# "../data/S035/S035R07.edf",
		# "../data/S035/S035R11.edf",
		# "../data/S035/S035R03.edf",
		# "../data/S060/S060R07.edf",
		# "../data/S060/S060R11.edf",
		# "../data/S060/S060R03.edf",
		# "../data/S009/S009R11.edf",
		# "../data/S009/S009R07.edf",
		# "../data/S009/S009R03.edf",
		# "../data/S045/S045R11.edf",
		# "../data/S045/S045R07.edf",
		# "../data/S045/S045R03.edf",
		# "../data/S044/S044R03.edf",
		# "../data/S044/S044R11.edf",
		# "../data/S044/S044R07.edf",
		# "../data/S029/S029R11.edf",
		# "../data/S029/S029R03.edf",
		# "../data/S029/S029R07.edf",
		# "../data/S056/S056R03.edf",
		# "../data/S056/S056R11.edf",
		# "../data/S056/S056R07.edf",
		# "../data/S076/S076R07.edf",
		# "../data/S076/S076R03.edf",
		# "../data/S076/S076R11.edf",
		# "../data/S105/S105R07.edf",
		# "../data/S105/S105R11.edf",
		# "../data/S105/S105R03.edf",
		# "../data/S106/S106R07.edf",
		# "../data/S106/S106R03.edf",
		# "../data/S106/S106R11.edf",
		# "../data/S050/S050R07.edf",
		# "../data/S050/S050R03.edf",
		# "../data/S050/S050R11.edf",
		# "../data/S099/S099R07.edf",
		# "../data/S099/S099R03.edf",
		# "../data/S099/S099R11.edf",
		# "../data/S031/S031R03.edf",
		# "../data/S031/S031R11.edf",
		# "../data/S031/S031R07.edf",
		# "../data/S061/S061R03.edf",
		# "../data/S061/S061R07.edf",
		# "../data/S061/S061R11.edf",
		# "../data/S059/S059R07.edf",
		# "../data/S059/S059R11.edf",
		# "../data/S059/S059R03.edf",
		# "../data/S072/S072R07.edf",
		# "../data/S072/S072R03.edf",
		# "../data/S072/S072R11.edf",
		# "../data/S023/S023R03.edf",
		# "../data/S023/S023R11.edf",
		# "../data/S023/S023R07.edf",
		# "../data/S043/S043R11.edf",
		# "../data/S043/S043R07.edf",
		# "../data/S043/S043R03.edf",
		# "../data/S073/S073R07.edf",
		# "../data/S073/S073R11.edf",
		# "../data/S073/S073R03.edf",
		# "../data/S046/S046R11.edf",
		# "../data/S046/S046R07.edf",
		# "../data/S046/S046R03.edf",
		# "../data/S075/S075R07.edf",
		# "../data/S075/S075R11.edf",
		# "../data/S075/S075R03.edf",
		# "../data/S011/S011R03.edf",
		# "../data/S011/S011R07.edf",
		# "../data/S011/S011R11.edf",
		# "../data/S066/S066R03.edf",
		# "../data/S066/S066R07.edf",
		# "../data/S066/S066R11.edf",
		# "../data/S006/S006R11.edf",
		# "../data/S006/S006R03.edf",
		# "../data/S006/S006R07.edf",
		# "../data/S021/S021R11.edf",
		# "../data/S021/S021R03.edf",
		# "../data/S021/S021R07.edf",
		# "../data/S010/S010R03.edf",
		# "../data/S010/S010R07.edf",
		# "../data/S010/S010R11.edf",
		# "../data/S008/S008R07.edf",
		# "../data/S008/S008R03.edf",
		# "../data/S008/S008R11.edf",
		# "../data/S089/S089R03.edf",
		# "../data/S089/S089R07.edf",
		# "../data/S089/S089R11.edf",
		# "../data/S058/S058R07.edf",
		# "../data/S058/S058R11.edf",
		# "../data/S058/S058R03.edf",
		# "../data/S090/S090R03.edf",
		# "../data/S090/S090R07.edf",
]

# ica = mne.preprocessing.ICA(method="infomax")
#--------------------------------------------------------------------------------------------------------------------------


# def initate_mlflow_environment():
# 	subprocess.Popen(["mlflow", "ui"])
# 	mlflow.set_tracking_uri("http://localhost:5000")  #uri
# 	print('mlfow is running on http://localhost:5000", here you can follow the model metrics.')
# 	time.sleep(2)



# def create_grid_search_parameters():
# 	with open('../configs/grid_search_parameters.yaml', 'r') as f:
# 		config = yaml.safe_load(f)

# 	classifier_mapping = {
# 			'MLPClassifier': MLPClassifier(max_iter=10000,early_stopping=True,n_iter_no_change=50,verbose=False),
# 			'SVC': SVC(),
# 			'RandomForestClassifier': RandomForestClassifier(),
# 			'LogisticRegression': LogisticRegression(),
# 			'DecisionTreeClassifier': DecisionTreeClassifier()
# 		}

# 	grid_search_params = []
# 	for param_set in config['grid_search_params']:
# 		classifier_name = param_set['classifier']
# 		if classifier_name in classifier_mapping:
# 			param_set['classifier'] = [classifier_mapping[classifier_name]] 
# 			print(f'{param_set} is gonna be the paramset now')
# 			grid_search_params.append(param_set)

# 	return grid_search_params


from src.experiment_trainer import ExperimentTrainerFacade

def main():
	# try:
	experiment_trainer = ExperimentTrainerFacade()
	experiment_trainer.run_experiment()
	
	# try:
	# 	argument_config = [
	# 		{
	# 			'name': '--mlflow',
	# 			'type': str,
	# 			'default': 'false',
	# 			'choices': ['true', 'false'],
	# 			'help':'Enable (True) or disable (False) the mlflow server for tracking model analysis. Default is False.\n'
	# 		}
	# 	]

	# 	#cli parser
	# 	arg_parser = CommandLineParser(argument_config)
	# 	mlflow_enabled = arg_parser.parse_arguments()
	# 	print(mlflow_enabled)
	# 	if (mlflow_enabled == True):
	# 		print(f'MLFLOW enabled: go to localhost:5000 to see model metrics.') #green color?
	# 		initate_mlflow_environment()

	# 	#data processing
	# 	dataset_preprocessor_instance = Preprocessor()
	# 	loaded_raw_data = dataset_preprocessor_instance.load_raw_data(data_path=train) #RETURN DOESNT WORK, IT RETURNS AFTER 1 FILE
	# 	filtered_data = dataset_preprocessor_instance.filter_raw_data(loaded_raw_data) #this returns a triplet now
	
	# 	#epoch (events) and label extraction
	# 	epoch_extractor_instance = EpochExtractor()
	# 	epochs_dict, labels_dict = epoch_extractor_instance.extract_epochs_and_labels(filtered_data)
		
	# 	#process run groups (14 test runs separated to different run groups based on motion onse)
	# 	run_groups = epoch_extractor_instance.experiments_list
	# 	for groups in run_groups:
	# 		groups_runs = groups['runs']
	# 		group_key = f'runs_{"_".join(map(str, groups_runs))}'
	# 		print(f"\nProcessing group: {group_key} with runs {groups_runs[0]}")

	# 		run_keys = [run_key for run_key in epochs_dict.keys() if int(run_key[-2:]) in groups_runs]
	# 		available_runs = [run_key for run_key in run_keys if run_key in epochs_dict]

	# 		if not available_runs:
	# 			print(f"No available runs for group '{group_key}', skipping.")
	# 			continue
			
	# 		feature_extraction_method = 'baseline' if groups_runs[0] in [1,2] else 'events'
			
	# 		#feature extraction
	# 		feature_extractor_instance = FeatureExtractor()
	# 		X_train = feature_extractor_instance.extract_features(epochs_dict[run_keys[0]], feature_extraction_method) #trained_extracted_features, for now groups runs[0] is ok but at 13 etc it wont be
	# 		y_train = labels_dict[run_keys[0]] #trained_extracted_labels
			
			
	# 		#https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.FunctionTransformer.html
	# 		# custom_scaler = CustomScaler()
	# 		# reshaper = Reshaper()
	# 		# my_pca = My_PCA(n_comps=100)
	# 		# mlp_classifier = MLPClassifier(hidden_layer_sizes=(20,10),
	# 		# 							max_iter=16000,
	# 		# 							random_state=42
	# 		# )

	# 		# pipeline = Pipeline([
	# 		# 	('scaler', custom_scaler),
	# 		# 	('reshaper', reshaper),
	# 		# 	('pca', my_pca),
	# 		# 	('classifier', mlp_classifier) #mlp will be replaced in grid search
	# 		# ])

	# 		from pipeline_builder import PipelineBuilder
	# 		pipeline_builder = PipelineBuilder(n_components = 40)
	# 		pipeline = pipeline_builder.build_pipeline()

	# 		from grid_search_manager import GridSearchManager
			
	# 		grid_search_manager = GridSearchManager()
	# 		grid_search = grid_search_manager.create_grid_search(pipeline)

	# 		# grid_search_params = create_grid_search_parameters()
	# 		# grid_search = GridSearchCV(
	# 		# 	estimator=pipeline,
	# 		# 	param_grid=grid_search_params,
	# 		# 	cv=9,  #9fold cross-val
	# 		# 	scoring='accuracy',  #evalmetric
	# 		# 	n_jobs=-1,  #util all all available cpu cores
	# 		# 	verbose=1,  #2 would be for detailed output
	# 		# 	refit=True #this fits it automatically to the best estimator, just to emphasize here, its True by default
	# 		# )

	# 		mlflow_manager = MlflowManager()
	# 		grid_search.fit(X_train, y_train)
	# 		best_params = {k: (float(v) if isinstance(v, (np.float64, np.float32)) else v) for k, v in grid_search.best_params_.items()}
	# 		best_score = float(grid_search.best_score_)  # Ensure it's a Python float
	# 		best_pipeline = grid_search.best_estimator_


	# 		if mlflow_enabled == True:
	# 			# print(f'Waiting for you to launch in a console: mlflow ui, then you can go to: {mlflow.get_tracking_uri()})')  #should have a uri output, do i have to start mlflow before?
	# 			with mlflow.start_run(run_name=group_key):
	# 			# 	grid_search.fit(X_train, y_train)


	# 			# 	best_params = {k: (float(v) if isinstance(v, (np.float64, np.float32)) else v) for k, v in grid_search.best_params_.items()}
	# 			# 	best_score = float(grid_search.best_score_)  # Ensure it's a Python float
					
	# 			# 	mlflow.set_experiment(f"{group_key}")
	# 			# 	mlflow.log_param('group_key', group_key)
	# 			# 	mlflow.log_params(best_params)
	# 			# 	mlflow.log_metric('best_cross_val_accuracy', best_score)


	# 			# 	print("Best Parameters:")
	# 			# 	print(best_params)
	# 			# 	print(f"Best Cross-Validation Accuracy: {best_score:.2f}")


	# 			# 	signature = infer_signature(X_train, y_train)
	# 			# 	best_pipeline = grid_search.best_estimator_
	# 			# 	model_filename = f"../models/pipe_{group_key}.joblib"
	# 			# 	joblib.dump(best_pipeline, model_filename)

	# 			# 	mlflow.sklearn.log_model(
	# 			# 		sk_model=best_pipeline, 
	# 			# 		artifact_path='models', 
	# 			# 		signature=signature, 
	# 			# 		registered_model_name=f"model_{group_key}"
	# 			# 	)
	# 				mlflow_manager.log_mlflow_experiment(group_key, best_params, best_score, best_pipeline, X_train, y_train)
	# 		else:
	# 			grid_search.fit(X_train, y_train)

	# 			best_params = {k: (float(v) if isinstance(v, (np.float64, np.float32)) else v) for k, v in grid_search.best_params_.items()}
	# 			best_score = float(grid_search.best_score_)  # Ensure it's a Python float
	# 			best_pipeline = grid_search.best_estimator_

	# 			#console log maybe?
	# 			model_filename = f"../models/pipe_{group_key}.joblib"
	# 			joblib.dump(best_pipeline, model_filename)
	# 			print(f'best score is: {best_score}, with the best pipeline estimator of: {best_pipeline}.\nModel saved to: ../models/pipe_{group_key}.joblib')

	# 			kfold = KFold(n_splits=5, shuffle=True, random_state=0)
	# 			scores = cross_val_score(
	# 				pipeline, X_train, 
	# 				y_train, 
	# 				scoring='accuracy', 
	# 				cv=kfold
	# 			)

	# 			print(scores)
	# 			print(f"\033[92mAverage accuracy with cross-validation for group: {groups_runs}: {scores.mean():.2f}\033[0m")



	# except FileNotFoundError as e:
	# 	logging.error(f"File not found: {e}")
	# except PermissionError as e:
	# 	logging.error(f"Permission on the file denied: {e}")
	# except IOError as e:
	# 	logging.error(f"Error reading the data file: {e}")
	# except ValueError as e:
	# 	logging.error(f"Invalid EDF data: {e}")
	# except TypeError as e:
	# 		logging.error(f"{e}")

if __name__ == '__main__':
	main()




# from pipeline_builder import PipelineBuilder
# from grid_search_manager import GridSearchManager
# from pipeline_executor import PipelineExecutor

# #create a facade which has all the subsystems
# class ExperimentTrainerFacade(self, config_path='../configs/grid_search_parameters.yaml', mlflow_enabled=False):
# 	self.command_line_parser = CommandLineParser({
# 				'name': '--mlfow',
# 				'type': str,
# 				'default': 'false',
# 				'choices': ['true', 'false'],
# 				'help':'Enable (True) or disable (False) the mlflow server for tracking model analysis. Default is False.\n'
# 			})
# 	self.mlflow_manager = MlflowManager()
# 	self.data_preprocessor = Preprocessor()
# 	self.epoch_extractor = EpochExtractor()
# 	self.pipeline_executor = PipelineExecutor()
# 	self.pipeline_builder = PipelineBuilder(n_components=40)
# 	self.grid_search_manager = GridSearchManager()


# 	def run_experiment(self):
# 		try:
# 			mlflow_enabled = self.command_line_parser.arg_parser.parse_arguments()
# 			if (mlflow_enabled == True):
# 				self.mlflow_manager.start_mlflow_server()

# 		#load data
# 		raw_data = self.data_preprocessor.load_raw_data(data_path=train)
# 		filtered_data = dataset_preprocessor_instance.filter_raw_data(raw_data) #this returns a triplet now

# 		#extract epochs and associated labels
# 		epochs_dict, labels_dict = epoch_extractor_instance.extract_epochs_and_labels(filtered_data)
# 		run_groups = epoch_extractor_instance.experiments_list
		
# 		#run the experiments on different groups
# 		self.process_run_groups(run_groups, mlflow_enabled)



# 	def process_run_groups(self, run_groups, mlflow_enabled):
# 		for groups in run_groups:
# 			groups_runs = groups['runs']
# 			group_key = f'runs_{"_".join(map(str, groups_runs))}'
# 			print(f"\nProcessing group: {group_key} with runs {groups_runs[0]}")

# 			run_keys = [run_key for run_key in epochs_dict.keys() if int(run_key[-2:]) in groups_runs]
# 			available_runs = [run_key for run_key in run_keys if run_key in epochs_dict]

# 			if not available_runs:
# 				print(f"No available runs for group '{group_key}', skipping.")
# 				continue
			
# 			feature_extraction_method = 'baseline' if groups_runs[0] in [1,2] else 'events'
			
# 			#feature extraction
# 			feature_extractor_instance = FeatureExtractor()
# 			X_train = feature_extractor_instance.extract_features(epochs_dict[run_keys[0]], feature_extraction_method) #trained_extracted_features, for now groups runs[0] is ok but at 13 etc it wont be
# 			y_train = labels_dict[run_keys[0]] #trained_extracted_labels
			
# 			#build a pipeline
# 			pipeline = pipeline_builder.build_pipeline()

# 			#run the grid search
# 			best_params, best_score, best_pipeline = self.grid_search_manager.run_grid_search(pipeline, X_train, y_train)
			
# 			if mlflow_enabled == True:
# 				# log metrics to mlflow
# 				with mlflow.start_run(run_name=group_key):
# 					#we could use the pipeline executor as an external function to save pipeline metrics?
# 					self.mlflow_manager.log_mlflow_experiment(group_key, best_params, best_score, best_pipeline, X_train, y_train) #this also dumps model
# 			else
# 				#save model
# 				self.pipeline_executor.save_model(best_pipeline, group_key)
# 				#print cross val scores
# 				self.pipeline_executor.evaluate_pipeline(group_key, best_pipeline, best_score)






'''
def main():
	try:
		dataset_preprocessor_instance = Preprocessor()
		loaded_raw_data = dataset_preprocessor_instance.load_raw_data(data_path=train) #RETURN DOESNT WORK, IT RETURNS AFTER 1 FILE
		# print(dataset_preprocessor_instance.raw_data)
		print(type(loaded_raw_data))
		filtered_data = dataset_preprocessor_instance.filter_raw_data(loaded_raw_data) #this returns a triplet now
		print(filtered_data) #this is a dict now
		# print(loaded_raw_data)
		# sys.exit(1)
		# for data in filtered_data:
		# 	print(data[0], data[1], data[2])
		# sys.exit(1)
		# print(experiment)
		epoch_extractor_instance = EpochExtractor()
		epochs, labels = epoch_extractor_instance.extract_epochs_and_labels(filtered_data)

		# print(labels)
		# print(epochs)
		# sys.exit(1)

		# for idx, epoch in enumerate(epochs):
		# 	print(epoch)
		# 	print(labels[idx])

		feature_extractor_instance = FeatureExtractor()
		trained_extracted_features = feature_extractor_instance.extract_features(epochs) #callable
		# trained_extracted_features = trained_extracted_features['3']

		

		# print(f'{trained_extracted_features}')
		print(f'{type(trained_extracted_features)} is the feature type, {type(labels)} is the typeoflabels')
		# print(type(trained_extracted_features))

		# sys.exit(1)

		#https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.FunctionTransformer.html
		custom_scaler = CustomScaler()
		reshaper = Reshaper()
		my_pca = My_PCA(n_comps=100)
		mlp_classifier = MLPClassifier(hidden_layer_sizes=(20,10),
									max_iter=16000,
									random_state=42
		)
		# sys.exit(1)

	#for customscaler 3d shape check
	

	# for key_nr in range(1,4):


		pipeline = Pipeline([
			('scaler', custom_scaler),
			('reshaper', reshaper),
			('pca', my_pca),
			('classifier', mlp_classifier) #mlp will be replaced in grid search
		])
		print(f"{labels['4'].shape}")
		pipeline.fit(trained_extracted_features, labels['4'])

		sys.exit(1)

	# #------------------------------------------------------------------------------------------------------------

	# predict_raw = dataset_preprocessor_instance.load_raw_data(data_path=predict)
	# predict_filtered = dataset_preprocessor_instance.filter_raw_data()
	# epochs_predict, labels_predict = epoch_extractor_instance.extract_epochs_and_labels(predict_filtered)

	# test_extracted_features = feature_extractor_instance.extract_features(epochs_predict) #callable


		shuffle_split_validation = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

		# # scoring = ['accuracy', 'precision', 'f1_micro'] this only works for: scores = cross_validate(pipeline_custom, x_train, y_train, scoring=scoring, cv=k_fold_cross_val)
		# # scores = cross_val_score(pipeline_custom, x_train, y_train, scoring='accuracy', cv=shuffle_split_validation)
		scores = cross_val_score(
			pipeline, trained_extracted_features, 
			labels['4'],  
			scoring='accuracy', 
			cv=shuffle_split_validation
		)
		
		print(scores)
		# print(f'Average accuracy: {scores.mean()}')


		# sys.exit(1)

		grid_search_params = [
			#MLP
			{
				'classifier': [MLPClassifier(
					max_iter=16000,
					early_stopping=True,
					n_iter_no_change=100, #if it doesnt improve for 10 epochs
					verbose=True)],
				'pca__n_comps': [20,30,42,50],
				#hidden layers of multilayer perceptron class
				'classifier__hidden_layer_sizes': [(20, 10), (50, 20), (100, 50)],
				#relu->helps mitigate vanishing gradients, faster convergence
				#tanh->hyperbolic tangent, outputs centered around zero
				'classifier__activation': ['relu', 'tanh'],
				#adam, efficient for large datasets, adapts learning rates
				#stochastic gradient, generalize better, slower convergence
				'classifier__solver': ['adam', 'sgd'],
				'classifier__learning_rate_init': [0.001, 0.01, 0.1]

			},
			#SVC
			{
				'classifier': [SVC()],
				'pca__n_comps': [20, 30, 42, 50],
				'classifier__C': [0.1, 1, 10],
				'classifier__kernel': ['linear', 'rbf']
			},
			
			# RANDOM FOREST
			{
				'classifier': [RandomForestClassifier()],
				'pca__n_comps': [20,30,42,50],
				'classifier__n_estimators': [50, 100, 200],
				'classifier__max_depth': [None, 10, 20]
			},
			#DECISION TREE
			{
				'pca__n_comps': [20, 30, 42, 50],
				'classifier': [DecisionTreeClassifier()],
				'classifier__max_depth': [None, 10, 20],
				'classifier__min_samples_split': [2, 5, 10]
			},
			# Logistic Regression
			{
				'classifier': [LogisticRegression()],
				'pca__n_comps': [20, 30, 42, 50],
				'classifier__C': [0.1, 1, 10],
				'classifier__penalty': ['l1', 'l2'],
				'classifier__solver': ['liblinear'],  # 'liblinear' supports 'l1' penalty
				'classifier__multi_class': ['auto'],
				'classifier__max_iter': [1000, 5000]
			}
		]

		from sklearn.model_selection import GridSearchCV

		grid_search = GridSearchCV(
			estimator=pipeline,
			param_grid=grid_search_params,
			cv=9,  #9fold cross-val
			scoring='accuracy',  #evalmetric
			n_jobs=-1,  #util all all available cpu cores
			verbose=2,  # For detailed output
			refit=True #this fits it automatically to the best estimator, just to emphasize here, its True by default
		)

		#just to use standard variables
		X_train = trained_extracted_features
		y_train = labels['3']
		grid_search.fit(X_train, y_train)

		print("Best Parameters:")
		print(grid_search.best_params_)
		print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.2f}")


		best_pipeline = grid_search.best_estimator_
		joblib.dump(best_pipeline, '../models/pipe.joblib')


	except FileNotFoundError as e:
		logging.error(f"File not found: {e}")
	except PermissionError as e:
		logging.error(f"Permission on the file denied: {e}")
	except IOError as e:
		logging.error(f"Error reading the data file: {e}")
	except ValueError as e:
		logging.error(f"Invalid EDF data: {e}")
	except TypeError as e:
			logging.error(f"{e}")

if __name__ == '__main__':
	main()
'''




#2024.12.20
# grid_search_params = [
# 	#MLP
# 	{
# 		'classifier': [MLPClassifier(
# 			max_iter=10000,
# 			early_stopping=True,
# 			n_iter_no_change=50, #if it doesnt improve for 10 epochs
# 			verbose=False)],
# 		'pca__n_comps': [20,42,50],
# 		#hidden layers of multilayer perceptron class
# 		'classifier__hidden_layer_sizes': [(20, 10), (50, 20), (100, 30)],
# 		#relu->helps mitigate vanishing gradients, faster convergence
# 		#tanh->hyperbolic tangent, outputs centered around zero
# 		'classifier__activation': ['relu', 'tanh'],
# 		#adam, efficient for large datasets, adapts learning rates
# 		#stochastic gradient, generalize better, slower convergence
# 		'classifier__solver': ['adam', 'sgd'],
# 		'classifier__learning_rate_init': [0.001, 0.01, 0.1]

# 	},

# 	# SVC
# 	{
# 		'classifier': [SVC()],
# 		'pca__n_comps': [20, 42, 50],
# 		'classifier__C': [0.1, 1, 8],
# 		'classifier__kernel': ['linear', 'rbf']
# 	},
	
# 	# RANDOM FOREST
# 	{
# 		'classifier': [RandomForestClassifier()],
# 		'pca__n_comps': [20, 42],
# 		'classifier__n_estimators': [50, 100, 200],
# 		'classifier__max_depth': [None, 10, 20]
# 	},
# 	#DECISION TREE
# 	{
# 		'pca__n_comps': [20, 42],
# 		'classifier': [DecisionTreeClassifier()],
# 		'classifier__max_depth': [None, 10, 20],
# 		'classifier__min_samples_split': [2, 5, 10]
# 	},
# 	# Logistic Regression
# 	{
# 		'classifier': [LogisticRegression()],
# 		'pca__n_comps': [20, 42, 50],
# 		'classifier__C': [0.1, 1, 10],
# 		'classifier__penalty': ['l1', 'l2'],
# 		'classifier__solver': ['liblinear'],  # 'liblinear' supports 'l1' penalty
# 		'classifier__max_iter': [1000, 5000]
# 	}
# ]
