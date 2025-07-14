import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

from src.components.data_transformation import DataTransformation


@dataclass
class ModelTrainerConfig:
    trained_model_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    
    def GetModels(self):
        models={
            "linear regression":LinearRegression(),
            "logistic regression":LogisticRegression(),
            "random forest":RandomForestRegressor(),
            "decision tree":DecisionTreeRegressor(),
            "gradient boosting":GradientBoostingRegressor(),
            "xg boosting":XGBRegressor(),
            "cat boosting":CatBoostRegressor(),
            "ada boositng":AdaBoostRegressor(),
            "knn":KNeighborsRegressor()
        }

        return models
    


    def initiate_model_trainer(self):
        try:
            train_data_file_path = "artifacts/train.csv"
            test_data_file_path = "artifacts/test.csv"

            data_transform_obj=DataTransformation()

            x_train, y_train, x_test, y_test, preprocessor_path = data_transform_obj.initiate_data_transformation(
                train_data_file_path,
                test_data_file_path
            )

            models = self.GetModels()

            # Remove Logistic Regression as it is a classifier
            if "logistic regression" in models:
                del models["logistic regression"]

            model_report = evaluate_models(
                x_train,
                y_train,
                x_test,
                y_test,
                models
            )

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            # Fit best model before saving
            best_model.fit(x_train, y_train)

            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            y_pred = best_model.predict(x_test)
            r2 = r2_score(y_test, y_pred)
            return best_model,r2

        except Exception as e:
            raise CustomException(e, sys)

        


if __name__=="__main__":
    obj=ModelTrainer()
    best_model,r2=obj.initiate_model_trainer()
    print(f"best output by {best_model} : {r2}")     