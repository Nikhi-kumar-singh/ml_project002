import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self,num_features,cat_features):
        try:
            num_pipeline=Pipeline([
                ("imputer",SimpleImputer(strategy="mean")),
                ("scaler",StandardScaler())
            ])

            cat_pipeline=Pipeline([
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("encoder",OneHotEncoder(drop="first",handle_unknown="ignore"))
            ])

            logging.info("num and cat pipeline scaling is done.")

            preprocessor=ColumnTransformer([
                ("num_feature_transformer",num_pipeline,num_features),
                ("cat_feature_transformer",cat_pipeline,cat_features)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(sys,e)
        
    def initiate_data_transformation(self,train_file_path,test_file_path):
        try: 
            train_df=pd.read_csv(train_file_path)
            test_df=pd.read_csv(test_file_path)
            target_column="math score"
            num_features=list(train_df.select_dtypes(exclude="O").columns)
            cat_features=list(train_df.select_dtypes(include="O").columns)

            if target_column in num_features:
                num_features.remove(target_column)
            if target_column in cat_features:
                cat_features.remove(target_column)


            preprocessor_obj=self.get_data_transformer_object(num_features,cat_features)

            x_train=train_df.drop([target_column],axis=1)
            x_test=test_df.drop([target_column],axis=1)
            y_train=train_df[target_column]
            y_test=test_df[target_column]

            x_train=preprocessor_obj.fit_transform(x_train)
            x_test=preprocessor_obj.transform(x_test)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                x_train,
                x_test,
                y_train,
                y_test,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)
        




if __name__ == "__main__":
    obj=DataTransformation()
    train_file_path="artifacts/train.csv"
    test_file_path="artifacts/test.csv"
    x_train,x_test,y_train,y_test,preprocessor_obj_file_path=obj.initiate_data_transformation(train_file_path,test_file_path)

    print(f"{x_train.shape},{x_test.shape}")

