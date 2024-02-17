import os,sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.utils import MainUtils
from src.exception import CustomException
import json
from pymongo import MongoClient
from src.constant import *

@dataclass
class Data_Ingestion_Config:
    raw_path = os.path.join('artifacts/raw')
    

class Data_Ingestion:
    def __init__(self):
        self.data_ingestion_config = Data_Ingestion_Config()
        self.utils = MainUtils()
        
    def read_and_import_data(self,path):
        try:
            client = MongoClient(uri)
            jsoned_data = client[DATABASE_NAME][DATABASE_COLLECTION_NAME]
            df = pd.DataFrame(list(jsoned_data.find()))
            df.replace("nan",np.nan)
            return df    
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_ingestion(self,path):
        try:
            dataset = self.read_and_import_data(path)
            dataset_path = os.path.join(self.data_ingestion_config,'dataset.csv')
            dataset.to_csv(dataset_path,index=False)
            return self.data_ingestion_config.raw_path
        
        except Exception as e:
            raise CustomException(e,sys)
    
    