import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import os
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
nltk.download(['punkt', 'wordnet', 'stopwords'])


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier


import pickle
import warnings
warnings.filterwarnings('ignore')

#python models/train_classifier.py 'data/DisasterResponse.db' 'models/classifier.pkl'

def load_data(database_filepath):
    """ Function to load the data for ML processing from the previously cleaned database table into a dataframe and prepare for ML train test split.   
    
    Args: 
        database_filepath : Path to the database file
        
    Retuns:
        X series: messages to be input into the ML model
        y series: dataframe that has the category types for each message in X
        category names: names of the category types
        
     """
        
    engine = create_engine('sqlite:///'+database_filepath)    
    table_name = os.path.basename(database_filepath).split('.')[0]
    df = pd.read_sql_table(table_name,engine)
    X = df['message']
    y = df.drop(['id','message', 'original','genre'], axis =1)
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """ This function messages into tokens that are lemmatized and cleaned. example remove url like links and lower the case, remove non alphanumeric charecters and stopwords.
    
    Args:
        text: input text that needs to be tokenised
        
    returns: 
        clean tokens - a list type
    """
    
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()    
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    clean_tokens_sw_removed = [w for w in clean_tokens if w not in stopwords.words("english")]        
    
    return clean_tokens_sw_removed


def build_model():
    
    """ Builds a ML model that can count vectorize , apply tfidf function and also classify the messages using a Random forest classifier
    
    Args: none
    
    returns: 
         ML pipeline model that can be used to make classification predictions on the message text.
         
    """
    
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    
    """ This function evaluates the ML model performance 
    
    Args: 
        model : model to be evaluated.
        X_test: test data for prediction 
        y_test: test classification category results
        category_names : Names of the category columns
        
    Returns: None
        prints out the precision, accuracy and recall scores 
        
    """
    
    y_testpred = model.predict(X_test)
    
    for i,col in enumerate(category_names):
        print('Category: {}'.format(col))
        print(classification_report(y_testpred[:,i], Y_test[col]))
        test_accuracy = (y_testpred[:,i] == Y_test[col]).mean()
        print('Test accuracy ={}'.format(test_accuracy))


def save_model(model, model_filepath):
    """ This function saves the input model into a pickle file.
    
    Args: 
        model : model to be saved.
        
    returns: None
    """
    
    
    #Exporting the ML model, i,e the Random forest classifier
    pickle.dump(model, open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()