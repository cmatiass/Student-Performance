import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    # Define paths for storing train, test and raw data files in artifacts directory
    train_data_path: str=os.path.join('artifacts',"train.csv")  # Path to save training data
    test_data_path: str=os.path.join('artifacts',"test.csv")    # Path to save testing data
    raw_data_path: str=os.path.join('artifacts',"data.csv")     # Path to save original data

class DataIngestion:
    """
    Class responsible for data ingestion process:
    - Reading the raw data
    - Splitting into train and test sets
    - Saving the datasets to disk
    """
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()  # Initialize configuration

    def initiate_data_ingestion(self):
        """
        Execute the data ingestion process:
        1. Read the raw dataset
        2. Create directory for storing processed data if not exists
        3. Save raw data to CSV
        4. Split data into train and test sets
        5. Save train and test sets to CSV
        
        Returns:
            tuple: Paths to the train and test data files
        """
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the dataset from source
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')

            # Create the artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            # Save the raw data to CSV
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            # Split the data into training and testing sets
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)  # 80% training, 20% testing

            # Save training and testing data to CSV files
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")  # Fixed typo in the log message

            # Return the paths to the training and testing data
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)  # Raise custom exception if any error occurs
        
if __name__=="__main__":
    # This block runs when the script is executed directly
    
    # Step 1: Perform data ingestion
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    # Step 2: Perform data transformation on the ingested data
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    # Step 3: Train the model using transformed data
    modeltrainer=ModelTrainer()



