import re
import os
import json
import plotly
import pandas as pd
import numpy as np
import sys
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract genre counts for first graph
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # extract category counts for second graph    
    dfcategory_counts = df[df.columns[4:]].astype('int')
    category_counts = dfcategory_counts[:].sum().sort_values()
    category_name = list(category_counts.index)
    
    
    # top 5 category and their message counts
    top5category_counts = df[df.columns[4:]].astype('int')
    topcategory_counts = top5category_counts[:].sum().sort_values(ascending=False)
    category_counts_top5 = topcategory_counts.head()
    category_name_top5 = list(category_counts_top5.index)
    
    """ create 3 visuals   
    1. Distribution of messages by genres.
    2. Distribution of message by all Categories
    3. Distribution of messages by top 5 categories.
    """
    
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker=dict(
                            color='rgba(55, 110, 20, 0.8)',
                            line=dict(color='rgba(55, 110, 20, 0.8)', width=2))
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'titlefont': {
                            'size': 22,
                            'color': 'rgb(107, 107, 107)'
                            },
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_name,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                 'titlefont': {
                            'size': 22,
                            'color': 'rgb(107, 107, 107)'
                            },
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_name_top5,
                    y=category_counts_top5,
                     marker=dict(
                            color='rgba(40, 10, 50, 0.7)',
                            line=dict(color='rgba(40, 10, 50, 0.7)', width=2))
                )
            ],

            'layout': {
                'title': 'Distribution of messages by top 5 categories',
                'titlefont': {
                            'size': 22,
                            'color': 'rgb(107, 107, 107)'
                            },
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()