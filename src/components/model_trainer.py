import os
import sys
from dataclasses import dataclass

# Importing various regression models
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score  # For model evaluation
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Custom exception handling and logging
from src.exception import CustomException
from src.logger import logging

# Utility functions for saving model and evaluation
from src.utils import save_object, evaluate_models

# Configuration class for model training
@dataclass
class ModelTrainerConfig:
    # Path where the trained model will be saved
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

# Main class for handling model training
class ModelTrainer:
    def __init__(self):
        # Initialize with configuration
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            # Separating features and target variables from the datasets
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],    # All columns except the last one (features)
                train_array[:, -1],     # Only the last column (target variable)
                test_array[:, :-1],     # All columns except the last one (features)
                test_array[:, -1]       # Only the last column (target variable)
            )
            
            # Dictionary of regression models to evaluate
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            # Hyperparameter configurations for each model
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]  # Number of trees in the forest
                },
                "Gradient Boosting": {
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [.1, .01, .05, .001],      # Step size shrinkage
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],  # Fraction of samples for trees
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]   # Number of boosting stages
                },
                "Linear Regression": {},  # No hyperparameters to tune
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],                  # Depth of trees
                    'learning_rate': [0.01, 0.05, 0.1],   # Learning rate
                    'iterations': [30, 50, 100]           # Number of boosting iterations
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]  # Number of estimators
                }
            }

            # Evaluate all models with their hyperparameters using grid search
            model_report: dict = evaluate_models(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test,
                models=models, 
                param=params
            )
            
            # Find the best model based on score
            best_model_score = max(sorted(model_report.values()))

            # Get the name of the best model
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # Validate if the best model meets the minimum threshold
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            # Save the best model to disk
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Make predictions and calculate final R2 score
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            # Handle exceptions using custom exception class
            raise CustomException(e, sys)