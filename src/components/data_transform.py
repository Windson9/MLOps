import sys
import numpy as np
import pandas as pd
import os


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataclasses import dataclass
from src.utils import save_object

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformConfig:
    preprocessor_obj_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransform:
    def __init__(self) -> None:
        self.data_transform_config = DataTransformConfig()

    def get_data_transform(self):
        """
        This function will convert numerical and categorical values
        """
        try:
            numerical_col = ['writing_score', 'reading_score']
            categorical_col = [
                'gender', 
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info('Numberical and Categorical scaling and encoding completed')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_col),
                    ('cat_pipeline', cat_pipeline, categorical_col)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transform(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('Initiating data transform')

            logging.info('Obtaining pre-processing object')
            
            preprocessor_obj = self.get_data_transform()
            target_column = "math_score"
            numerical_col = ['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info('Applying preprocessing object on Data frame')

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info('Saving preprocessing object')

            save_object(self.data_transform_config.preprocessor_obj_path, obj=preprocessor_obj)

            return(
                train_arr,
                test_arr,
                self.data_transform_config.preprocessor_obj_path
            )

        except Exception as e:
            raise CustomException(e, sys)
            