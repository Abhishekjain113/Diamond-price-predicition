import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model
from dataclasses import dataclass
import os
import sys

@dataclass
class ModelTrainingConfig:
    trained_model_config=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainingConfig()
    def inital_model_training(self,train_array,test_array):
        try:
            logging.info('spiltting dependent and independent varibles from train and test case ')
            X_train,Y_train,X_test,Y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models={
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'ElasticNet':ElasticNet()
            }

            models_report:dict=evaluate_model(X_train,Y_train,X_test,Y_test,models)
            logging.info('Model:resport:{model_report}')
            # to get the best model score from dictionary
            best_model_score=max(sorted(models_report.values()))
            best_model_name=list(models_report.keys())[list(models_report.values()).index(best_model_score)]

            best_model=models[best_model_name]
            logging.info("Got best model")
            save_object(
                file_path=self.model_trainer_config.trained_model_config,
                obj=best_model
            )
        except Exception as e:
            logging.info("modeltraining failed")