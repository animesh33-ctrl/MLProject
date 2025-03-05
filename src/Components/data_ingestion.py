import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from ..logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    logging.info("Data Ingestion Config Class Invoked")
    train_path = os.path.join("artifacts","train.csv")
    test_path = os.path.join("artifacts",'test.csv')
    raw_path = os.path.join("artifacts","raw.csv")
    logging.info("Data Ingestion Config Class IS COMPLETED")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    
    logging.info("Eneterd the data ingestion method")

    def initiate_data_ingestion(self):
        try:
            df = pd.read_csv(r'notebook\data\stud.csv')
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_path,index=False,header=True)
            logging.info("Train test Split Initialized :- ")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.data_ingestion_config.train_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_path,index=False,header=True)
            logging.info("Train test split is completed")
            return(
                self.data_ingestion_config.train_path,
                self.data_ingestion_config.test_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    train_path,test_path = obj.initiate_data_ingestion()