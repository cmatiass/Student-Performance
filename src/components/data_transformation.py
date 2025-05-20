import sys  # System-specific parameters and functions
from dataclasses import dataclass  # For creating data classes with minimal boilerplate code

# Data manipulation and analysis libraries
import numpy as np 
import pandas as pd

# Scikit-learn preprocessing and modeling components
from sklearn.compose import ColumnTransformer  # For applying different transformers to different columns
from sklearn.impute import SimpleImputer  # For handling missing values
from sklearn.pipeline import Pipeline  # For sequencing preprocessing steps
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # For encoding and scaling features

# Custom modules for error handling and logging
from src.exception import CustomException
from src.logger import logging
import os  # Operating system interface for file path operations

from src.utils import save_object  # Custom utility function for saving objects

@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation settings.
    Defines file paths for storing preprocessing objects.
    """
    preprocessor_obj_file_path = os.path.join('artifacts', "proprocessor.pkl")

class DataTransformation:
    """
    Class responsible for transforming raw data into features suitable for model training.
    Handles preprocessing tasks like imputation, encoding, and scaling.
    """
    def __init__(self):
        """
        Initialize the data transformation component with configuration.
        """
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation.
        Creates preprocessing pipelines for both numerical and categorical features.
        
        Returns:
            preprocessor: A ColumnTransformer object that applies appropriate transformations
                         to numerical and categorical columns
        '''
        try:
            # Define which columns should be treated as numerical vs categorical
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Pipeline for numerical features:
            # 1. Impute missing values with median
            # 2. Apply standard scaling (mean=0, variance=1)
            num_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
                ]
            )

            # Pipeline for categorical features:
            # 1. Impute missing values with most frequent value
            # 2. Apply one-hot encoding to convert categories to binary features
            # 3. Scale the binary features (with_mean=False because data is sparse)
            cat_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine the numerical and categorical pipelines using ColumnTransformer
            # This applies each pipeline to the appropriate subset of columns
            preprocessor = ColumnTransformer(
                [
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            # Proper exception handling with custom exception class
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        Applies data transformations to training and testing datasets.
        
        Args:
            train_path: Path to the training CSV file
            test_path: Path to the testing CSV file
            
        Returns:
            tuple: Transformed training array, test array, and path to the saved preprocessor object
        """
        try:
            # Load training and testing data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            # Get the preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Define target column and features
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Separate features and target for training data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate features and target for testing data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # Apply preprocessing transformations
            # fit_transform for training data (learns parameters and transforms)
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            # transform only for test data (uses parameters learned from training data)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine preprocessed features with target variable
            # np.c_ concatenates along the second axis
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            # Save the preprocessing object for later use in prediction pipeline
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return the transformed arrays and path to the saved preprocessor
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
