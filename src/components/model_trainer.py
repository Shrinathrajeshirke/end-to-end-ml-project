import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("spllitting train test data")

            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "Linear Regression":LinearRegression(),
                "Ridge": Ridge(),
                #"Lasso":Lasso(),
                #"KNR": KNeighborsRegressor(),
                #"Random Forest": RandomForestRegressor(),
                #"Decision Tree": DecisionTreeRegressor(),
                #"Gradient Boost Regressor": GradientBoostingRegressor(),
                #"AdaBoost Regressor":AdaBoostRegressor(),
                #"XGBClassifier":XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
            }

            params = {
                #"Decision Tree":{
                #    'crterion':['squared_error','friedman_mse','absolute_error','poisson'],
                #    'splitter':['best','random'],
                #    'max_features':['sqrt','log2']
                #},
                #"Random Forest":{
                #    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                #    'max_features':['sqrt','log2',None],
                #   'n_estimators':[8,16,32,64,128,256]
                #},
                #'Gradient boosting':{
                #    'loss':['squared_error','huber','absolute_error','squantile'],
                #    'learning_rate':[0.1,0.01,0.05,0.001],
                #    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                #    'criterion':['squared_error','friedman_mse'],
                #    'max_features':['auto','sqrt','log2'],
                #    'n_estimators':[8,16,32,64,128,256]
                #},
                'Linear Regression':{},
                'Ridge':{},
                #'KNR':{
                #    'n_neighbors':[5,7,9,11],
                #    'weights':['uniform','distance'],
                #    'algorithm':['ball_tree','kd_tree','brute']
                #},
                #'XGB':{
                #    'learning_rate':[0.1,0.01,0.05,0.001],
                #    'n_estimators':[8,16,32,64,128,256]
                #},
                'CatBoostRegressor':{
                    'depth':[6,8,10],
                    'learning_rate':[0.1,0.01,0.05,0.001]
                },
                #'Adaboost':{
                #    'learning_rate':[0.1,0.01,0.05,0.001],
                #    'loss':['linear','square','exponential'],
                #    'n_estimators':[8,16,32,64,128,256]
                #}

            }

            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train,X_test=X_test,y_test=y_test, models=models, params=params)

            ## to get the best model score
            best_model_score = max(sorted(model_report.values()))

            ## to get the best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info("Best model found")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2 = r2_score(y_test, predicted)
            return best_model, r2

        except Exception as e:
            raise CustomException(e,sys)
