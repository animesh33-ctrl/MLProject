from src.exception import CustomException
import os
import sys
import dill
import pickle
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import GridSearchCV 

def save_object(file_path,obj):
    try:
        dir = os.path.dirname(file_path)
        os.makedirs(dir,exist_ok=True)
        with open(file_path,'wb') as fj:
            pickle.dump(obj,fj)

    except Exception as e:
        raise CustomException(e,sys)


def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]

            para=params[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #prediction
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Scores
            mse_train = mean_squared_error(y_train_pred,y_train)
            mse_test = mean_squared_error(y_test_pred,y_test)
            r2_train = r2_score(y_train_pred,y_train)
            r2_test = r2_score(y_test_pred,y_test)
            mae_train = mean_absolute_error(y_train_pred,y_train)
            mae_test = mean_absolute_error(y_test_pred,y_test)

            report[list(models.keys())[i]] = r2_test

        return report
            
    except Exception as e:
        raise CustomException(e,sys)
    

def load_object(filepath):
    try:
        with open(filepath,"rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e,sys)