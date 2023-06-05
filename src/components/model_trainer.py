import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model


@dataclass
class ModelTrainingConfig:
    training_model_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self) -> None:
        self.model_traininer_config = ModelTrainingConfig()

    def initiate_model_trainer(self, train, test):
        try:
            logging.info("Spliting training and test array")
            X_train, y_train, X_test, y_test = (
                train[:,:-1],
                train[:,-1],
                test[:,:-1],
                test[:,-1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(), 
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No models passes the threshold for accuracy")
            
            logging.info(f"Best model found! {best_model_name}")
            print(f"Best performing model is: {best_model_name}")

            save_object(
                file_path = self.model_traininer_config.training_model_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            score = r2_score(y_test, predicted)

            return score

        except Exception as e:
            raise CustomException(e, sys)