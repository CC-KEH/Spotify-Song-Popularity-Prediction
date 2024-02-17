import os,sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.utils import MainUtils
from src.exception import CustomException
import json
from pymongo import MongoClient
from src.constant import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

@dataclass
class Data_Transformation_Config:
    train_folder = os.path.join('artifacts/train')
    test_folder = os.path.join('artifacts/test')
    raw_path = os.path.join('artifacts/raw','dataset.csv')
    preprocessor_path = os.path.join('artifacts/preprocessor','preprocessor.pkl')
    
class Data_Transformation:
    def __init__(self):
        self.data_transformation_config = Data_Transformation_Config()
        self.utils = MainUtils()
        
    def preprocess_dataset(self,dataset):
        dataset['explicit'] = dataset['explicit'].map({False:0,True:1})
        dataset = dataset.drop(labels=['artists','album_name','track_id','Unnamed: 0','track_name'],axis=1,inplace=True)
        
        encoder = OneHotEncoder(drop='first',sparse_output=True)
        genres_encoded = encoder.fit_transform(dataset[['track_genre']])
        genres_name = encoder.get_feature_names_out(['track_genre'])
        genre_encoded_dense = genres_encoded.toarray()
        genres_data = pd.DataFrame(genre_encoded_dense, columns=genres_name)
        dataset = pd.concat([dataset,genres_data], axis=1)
        dataset = dataset.dropna()
        return dataset
    
    def read_and_transform_data(self,dataset):
        try:
            dataset = self.preprocess_dataset(dataset)
            # define the steps for the preprocessor pipeline
            imputer_step = ('imputer', SimpleImputer(strategy='constant', fill_value=0))
            scaler_step = ('scaler', StandardScaler())

            preprocessor = Pipeline(
                steps=[
                imputer_step,
                scaler_step
                ]
            )
            X = dataset[:,1:]
            y = dataset[:,0]
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,shuffle=True)
            X_train_scaled =  preprocessor.fit_transform(X_train)
            X_test_scaled  =  preprocessor.transform(X_test)
            
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_path), exist_ok= True)
            self.utils.save_object(file_path=self.preprocessor_path,obj=preprocessor)
            train_arr = np.c_[X_train_scaled,np.array(y_train)]
            test_arr = np.c_[X_test_scaled,np.array(y_test)]
            
            return (train_arr,test_arr,self.data_transformation_config.preprocessor_path)
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,path):
        try:
            dataset = pd.read_csv(self.data_transformation_config.raw_path)
            dataset = self.read_and_transform_data(dataset)
        except Exception as e:
            raise CustomException(e,sys)
    
    