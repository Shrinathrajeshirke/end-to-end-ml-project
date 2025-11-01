import sys 
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

### define config class
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

### define class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    ## to create data tranformation object
    def get_transformer_object(self):
        try:
            numerical_features = ['reading_score','writing_score'] 
            categorical_features = ['gender','race_ethnicity',
                                    'parental_level_of_education',
                                    'lunch','test_preparation_course']

            ## tranformation pipeline for numerical features
            num_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info("Numerical features transformation completed")
            
            ## transformation pipeline for categorical features
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("Onehotencoder",OneHotEncoder()),
                    #("scaler",StandardScaler())
                ]
            )
            logging.info("categorical feature tranformation completed")

            ## create preprocessor object
            preprocesssor = ColumnTransformer(
                [
                    ("numerical pipeline",num_pipeline,numerical_features),
                    ("categorical pipeline",cat_pipeline,categorical_features)
                ]
            )
            return preprocesssor
            
        except Exception as e:
            raise CustomException(e,sys) 
    
    ## function to initiate data transformation
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path) ## read train data
            test_df = pd.read_csv(test_path) ## read test data

            logging.info('train and test data imported')

            logging.info("Obtaining preprocessor object")

            preprocessor_obj = self.get_transformer_object() ## create preprocessor object

            target_column_name = "math_score"

            input_features_train_df = train_df.drop(columns=[target_column_name],axis=1) ## independent features train dataframe
            target_feature_train_df = train_df[target_column_name]  ## dependent features train dataframe
            input_features_test_df = test_df.drop(columns=[target_column_name],axis=1) ## independent features test dataframe
            target_feature_test_df = test_df[target_column_name]  ## dependent features test dataframe

            logging.info("applying preprocessing object on training dataframe and testing dataframe")

            input_features_train_array = preprocessor_obj.fit_transform(input_features_train_df) ## apply transformation on train df
            input_features_test_array=preprocessor_obj.transform(input_features_test_df) ## apply transformation on test df

            ## column stack dependent and independent features
            train_arr = np.c_[
                input_features_train_array, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_features_test_array, np.array(target_feature_test_df)
            ]

            logging.info('saved preprocessing object.')

            ## save preprocessor object 
            ## save_object defined in utils.py
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

            
        except Exception as e:
            raise CustomException(e,sys)

