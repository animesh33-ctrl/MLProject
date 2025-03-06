import os
import sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.utils import save_object
from sklearn.preprocessing import OneHotEncoder,StandardScaler 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 
from dataclasses import dataclass
from src.exception import CustomException

@dataclass
class DataTransformerConfig:
    preprocessor_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformer:
    def __init__(self):
        self.data_transformer_config = DataTransformerConfig()

    def get_preprocessor_object(self):
        logging.info("Pipelining initiated :-")
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore')),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformer(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = "math_score"

            logging.info("X_train,y_train,X_test,y_test initiated :- ")
            X_train = train_df.drop(columns=[target_column],axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column],axis=1)
            y_test = test_df[target_column]

            logging.info("Obtaining Preprocessor :- ")
            preprocessor = self.get_preprocessor_object()

            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            train_arr = np.c_[
                X_train, np.array(y_train)
            ]
            test_arr = np.c_[X_test, np.array(y_test)]

            logging.info(f"Saved preprocessing object.")


            save_object(
                file_path = self.data_transformer_config.preprocessor_path,
                obj = preprocessor
            )

            return (
                train_arr,
                test_arr,
                self.data_transformer_config.preprocessor_path
            )

        except Exception as e:
            raise CustomException(e,sys)