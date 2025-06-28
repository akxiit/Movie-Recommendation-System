import os
import sys

import ast

import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from nltk.stem.porter import PorterStemmer


class dropNAN:
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.dropna()
        
        else:
            ### If it is the array so first we convert into the Dataframe and then we will drop null values
            return pd.DataFrame(X).dropna().values



class convert_genres:
    def __init__(self ,columns):
        self.columns = columns

    def fit(self, X):
        return self
    
    def transform(self, X, y=None):

        X=X.copy()

        logging.info('Genres and keyword convert started')
        def convert_genres(obj):
            try:
                L = []
                for i in ast.literal_eval(obj):
                    L.append(i['name'])

                return L
            except Exception as e:
                raise CustomException(e)
            
        for col in self.columns:
            if col in X.columns:
                X[col] = X[col].apply(convert_genres)

        return X
    
class convert_cast:

    def __init__(self, column):
        self.column = column

    def fit(self, X):
        return self
    
    def transform(self, X,y=None):
        X = X.copy()

        def convert(obj):
            try:
                L = []
                count = 0
                for i in ast.literal_eval(obj):
                    if count != 3:
                        L.append(i['name'])
                        count += 1

                    else:
                        break
                return L
            
            except Exception as e:
                raise CustomException(e)
            
        X[self.column] = X[self.column].apply(convert)

        return X
    

class convert_crew:
    def __init__(self, column):
        self.column = column

    def fit(self, X):
        return self
    
    def transform(self, X,y=None):
        X = X.copy()

        def fatch_director(obj):
            L = []

            for i in ast.literal_eval(obj):
                if i['job'] == 'Director':
                    L.append(i['name'])
                    break
            
            return L
            
        X[self.column] = X[self.column].apply(fatch_director)

        return X
    

class Text_splitter:

    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X[self.column] = X[self.column].apply(lambda x: x.split())
        return X
    

class RemoveSpaces:

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transformer(self, X):
        X = X.copy()

        for col in self.columns:
            X[col] = X[col].apply(lambda x: [i.replace(" ", "") for i in x])


        return X
    

class TagProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.ps = PorterStemmer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # Combine columns into a single list
        X['tags'] = X[self.columns].apply(lambda row: sum(row, []), axis=1)
        
        # Lowercase + join + stemming
        X['tags'] = X['tags'].apply(lambda tokens: ' '.join([self.ps.stem(token.lower()) for token in tokens]))
        
        return X[['movie_id', 'title', 'tags']]
    

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformationconfig = DataTransformationConfig()

    def convert(self, obj):
        L = []

        for i in ast.literal_eval(obj):
            L.append(i['name'])

        return L

    def get_data_transformation_obj(self):
        try:
            cols = ['movie_id', 'title', 'overview','genres','keywords','cast','crew']
            
            pipeline = Pipeline(steps=[
                ('dropna', dropNAN()),
                ('genres_transform', convert_genres(columns=['genres','keywords']), cols)
                ('cast_fatch', convert_cast(column='cast'), ['cast']),
                ('director_fatch', convert_crew(column='crew'), ['crew']),
                ('splitter', Text_splitter(column='overview')),
                ('remove_spaces', RemoveSpaces(columns=['genres', 'keywords', 'cast', 'crew'])),
                ('tag_processor', TagProcessor(columns=['genres', 'keywords', 'cast', 'crew']), ['movie_id', 'title', 'overview'])
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('pipeline', pipeline, cols)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, data_path):
        try:
            logging.info('Data Transformation started')

            df = pd.read_csv(data_path)

            logging.info('Obtaining preprocessor object')

            preprocessor_obj = self.get_data_transformation_obj()

            logging.info('Applying preprocessor object on data')

            transformed_data = preprocessor_obj.fit_transform(df)

            logging.info('Saving preprocessor object')

            

            import joblib
            joblib.dump(preprocessor_obj, self.data_transformationconfig.preprocessor_obj_file_path)    
