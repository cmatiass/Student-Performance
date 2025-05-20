import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os  # Adding missing import


class PredictPipeline:
    """
    Class for handling the prediction pipeline.
    Loads the trained model and preprocessor, then makes predictions on new data.
    """
    def __init__(self):
        pass

    def predict(self, features):
        """
        Make predictions using the trained model on new features
        
        Args:
            features: DataFrame containing the input features for prediction
            
        Returns:
            model predictions
        """
        try:
            # Define paths to the saved model and preprocessor artifacts
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            print("Before Loading")
            # Load the model and preprocessor from disk
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            
            # Transform the input features using the preprocessor
            data_scaled = preprocessor.transform(features)
            # Make predictions using the trained model
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            # Handle exceptions and provide detailed error information
            raise CustomException(e, sys)


class CustomData:
    """
    Class for handling user input data.
    Converts input parameters to a pandas DataFrame for prediction.
    """
    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):
        """
        Initialize CustomData object with student performance parameters
        
        Args:
            gender: Gender of the student
            race_ethnicity: Race/ethnicity group of the student
            parental_level_of_education: Education level of the student's parents
            lunch: Type of lunch the student receives
            test_preparation_course: Whether the student completed a test prep course
            reading_score: Student's reading score
            writing_score: Student's writing score
        """
        # Store all the input parameters as instance variables
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """
        Convert the input parameters to a pandas DataFrame for prediction
        
        Returns:
            pandas DataFrame containing the input parameters
        """
        try:
            # Create a dictionary with the input parameters
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            
            # Convert the dictionary to a pandas DataFrame and return it
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            # Handle exceptions and provide detailed error information
            raise CustomException(e, sys)

