# DisasterResponseProject
This project work was performed as part of the Udacity Data Scientist Nanodegree to satisfy the requirements of the curriculum of this program.

# Summary
Disaster Response Project aims to help the communities under adverse Disaster situations by providing the Software to mitigate issues arising out of a disaster.
This is done by creating a Text processing webapp which can intake messages, tweets etc and categorize them to various Disaster categories to enable different teams or agencies to take action based on the incoming message. The message is then fed into a machine learning model which classifies the message into appropriate categories. 
The background work before creating the Machine leanring model is performed by taking in the pre-labelled data from Figure eight. This data is processing using ETL , cleaned and standardised before being fed into the Machine leaning model for training the model. Then the Model itself is evaluated for accuracy and other parameters before being deployed into a web app for interfacing with the real world.

# Project files Structure

|-- app
|    |- run.py  # Flask file that runs app
|    |-- template
|          |-- master.html  # main page of web app
|          |-- go.html      # page for text input into classification
|    
|-- data
|    |-- DisasterResponse.db     # database to save clean data to
|    |- disaster_categories.csv  # data to process 
|    |- disaster_messages.csv    # data to process
|    |- process_data.py          # ETL pipeline script
|-- models
|    |-- classifier.pkl          # saved model. Not available in repo
|    |-- train_classifier.py     # ML pipeline script

-- README.md
-- .gitignore

# Set ups/installations/softwares required to run the project

 
# instructions to run the web app
1. Download the files or clone this repository
   git clone https://github.com/Sudhi162/DisasterResponseProject

2. Execute the scripts
   navigate to the projects root directory.
   
   Run the following commands:
- To run ETL pipeline that cleans data and stores in database
  python data/process_data.py 'data/disaster_messages.csv' 'data/disaster_categories.csv' 'data/DisasterResponse.db'
  
- To run ML pipeline that trains classifier and saves
  python models/train_classifier.py 'data/DisasterResponse.db' 'models/classifier.pkl'  

  Navigate to the app directory and run the command
  python run.py
  
  The web app instantiates. Type http://0.0.0.0:3001/ or http://localhost:3001/ to launch the webpage on the web browser.
  
  Input any message in the input box and click on the Classify Message button to see the categories that the message may best be classified into.
  
  
# Observations and discussions
  
# ETL Pipeline 
  Two datasets were provided by Figure eight namely messages and categories with an id column common to both messages. These two datasets were merged in the ETL Data pipeline.
  The categories were listed as "categoryname-number" format which needed to be parsed and cleaned to converted to numeric binary values for classification models.
  There were some duplicates in the dataset after merging which needed to be cleaned (~170 duplicate records).
  
# ML Pipeline
  
  
  ## Licensing, Authors, Acknowledgements
  Data set prelabelled provided by [Figure Eight](https://www.figure-eight.com/) for providing the datasets and directions
 [Udacity](https://www.udacity.com/) for project guidelines and setup.
 Author: Sudeendra Mangalwadekar
