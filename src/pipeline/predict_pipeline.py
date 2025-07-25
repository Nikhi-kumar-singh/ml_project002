import sys
import os
import pickle
import dill
import pandas as pd

from src.exception import CustomException
from src.logger import logging

from src.utils import load_object




class PredictPipeline:
    def __init__(self):
        pass


    def predict(self,features):
        try:
            preprocessor_path="artifacts/preprocessor.pkl"
            model_path="artifacts/model.pkl"

            preprocessor=load_object(file_path=preprocessor_path)
            model=load_object(file_path=model_path)
            data_scaled=preprocessor.transform(features)
            y_pred=model.predict(data_scaled)
            
            # print(f"head of the file: \n{data_scaled}")
            return y_pred
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(
            self,
            gender,
            race_ethnicity,
            parental_level_of_education,
            lunch,
            test_preparation_course,
            reading_score,
            writing_score
    ):
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score


    def get_data_as_frame(self):
        try:
            custom_data_input_dict={
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_course":[self.test_preparation_course],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score]
            }

            df=pd.DataFrame(custom_data_input_dict)
            # print(f"shape of data : {df.shape}")
            # print(f"head of the file: \n{df.head()}")
            return df

        except Exception as e:
            raise CustomException(e,sys)