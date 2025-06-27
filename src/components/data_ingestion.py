import os 
import sys

from src.exception import CustomException
from src.logger import logging
import pandas as pd

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('DataIngestion will be started')

        try:
            credits = pd.read_csv('notebook/data/tmdb_5000_credits.csv')
            movies = pd.read_csv('notebook/data/tmdb_5000_movies.csv')

            logging.info('Merging will be stated')

            df = movies.merge(credits, on='title', copy=False)

            logging.info('Spliting the required columns')

            df = df[['movie_id', 'title', 'overview','genres','keywords','cast','crew']]

            os.makedirs(os.path.dirname(self.ingestion_config.data_path), exist_ok=True)

            logging.info('DataIngestion will be started')

            df.to_csv(self.ingestion_config.data_path, index=False, header=True)

            logging.info('DataIngestion completed')

            return self.ingestion_config.data_path

        except Exception as e:
            CustomException(e)

if __name__ == '__main__':
    obj = DataIngestion()

    data = obj.initiate_data_ingestion()

