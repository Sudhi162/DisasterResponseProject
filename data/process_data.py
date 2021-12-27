# import libraries
import sys
import os
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

#python data/process_data.py 'data/disaster_messages.csv' 'data/disaster_categories.csv' 'data/DisasterResponse.db'

def load_data(messages_filepath, categories_filepath):
    """ Function to extract from the raw .csv files and merge messages and categories datasets
    
    Args:
        messages_filepath(str): path to the raw .csv message file
        categories_filepath(str): path to the raw .csv categories file 
        
   returns:
        dataframe : resulting dataframe after merging the two .csv files
        
    """
    messages =  pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on = ('id'),sort=True,copy=False)
    return df            


def clean_data(df):    
    """ Function to clean the dataframe that was previously constructed in the load_data function 
    cleaning process involves extracting the binary values from the categories columns and type conversion of these binary values, 
    as well as removing duplicates in the dataframe
    
    Args:
        df: dataframe
        
    returns:
        dataframe resulting from the cleaning process.
    """
    categories = df['categories'].str.split(pat=';',n=-1,expand=True)
    row = categories.iloc[1]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:].astype('int')
    
    # convert column from string to numeric
    categories[column] = pd.to_numeric(categories[column], errors='coerce')
    df.drop(['categories'],inplace=True,axis=1)
    df = pd.concat([df, categories.reindex(df.index)], axis=1)    
    df.drop_duplicates(inplace=True)
    df = df[df['related']!=2]
    return df


def save_data(df, database_filename):
    """ function to save the dataframe into a sql table on sqlite database
    
    Args:
        df: Dataframe to be saved into table
        database_filename: Name of the database file.
        
    Returns:
        None
    """
    
    
    engine = create_engine('sqlite:///'+database_filename)
    table_name = os.path.basename(database_filename).split('.')[0]
    df.to_sql(table_name, con=engine, if_exists='replace', index=False)
   


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()