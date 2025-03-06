import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from src.utils import evaluate_models,save_object
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerCOnfig:
    model_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerCOnfig()

    logging.info("Model Training Initiated :- ")
    def initiate_model_train(self,train_arr,test_arr):
        try:
            X_train,y_train,X_test,y_test = train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]

            models = {
                "Linear Regression" : LinearRegression(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGB Regressor": XGBRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False)
            }

            params={
                    "Decision Tree Regressor": {
                        'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                        # 'splitter':['best','random'],
                        # 'max_features':['sqrt','log2'],
                    },
                    "Random Forest Regressor":{
                        # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],

                        # 'max_features':['sqrt','log2',None],
                        'n_estimators': [8,16,32,64,128,256]
                    },
                    "Gradient Boosting Regressor":{
                        # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                        'learning_rate':[.1,.01,.05,.001],
                        'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                        # 'criterion':['squared_error', 'friedman_mse'],
                        # 'max_features':['auto','sqrt','log2'],
                        'n_estimators': [8,16,32,64,128,256]
                    },
                    "Linear Regression":{},
                    "XGB Regressor":{
                        'learning_rate':[.1,.01,.05,.001],
                        'n_estimators': [8,16,32,64,128,256]
                    },
                    "CatBoost Regressor":{
                        'depth': [6,8,10],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'iterations': [30, 50, 100]
                    },
                    "AdaBoost Regressor":{
                        'learning_rate':[.1,.01,0.5,.001],
                        # 'loss':['linear','square','exponential'],
                        'n_estimators': [8,16,32,64,128,256]
                    }

            }
            logging.info("Evaluating Models :- ")
            report = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)

            best_model_score = max(sorted(report.values()))
            best_model_name = list(report.keys())[
                list(report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            logging.info("Best Model Obtained :- ")
            if best_model_score <=0.6:
                raise CustomException("No best model Found")
            logging.info("Best Model Found based on training and testing data")


            save_object(
                file_path=self.model_trainer_config.model_path,
                obj= best_model
            )
            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square,best_model_name

            return 
        except Exception as e:
            raise CustomException(e,sys)
        
