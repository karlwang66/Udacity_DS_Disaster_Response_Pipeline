import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import nltk
from nltk.tokenize import *



def load_data(database_filepath):
    '''Load Datasets From Sqlite Database'''
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_query('select * from response_message',engine)
    # Extract column names
    cat_col = df.columns[4:].tolist()
    # Drop null values
    df = df.dropna(subset = cat_col)
    # Define features
    X = df['message']
    # Define Targets
    Y = df.iloc[:,4:]
    
    return X,Y,cat_col

def tokenize(text):
    '''Text Processing'''
    # Normalization
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenization
    new_text = word_tokenize(text)
    return new_text

def build_model():
    '''Model Pipline'''
    # Build Pipline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    # Define Parameters for gridsearchcv
    parameters = {'clf__estimator__min_samples_split':[2,10],
                  'clf__estimator__n_estimators':[10,128]}
    
    rf_grid = GridSearchCV(pipeline, param_grid=parameters)
    
    return rf_grid

def evaluate_model(model, X_test, Y_test, category_names):
    '''Model Evaluation'''
    # Make prediction
    y_pred = model.predict(X_test)
    # Print classification report and accuracy score
    for i in range(len(category_names)):
        print(classification_report(Y_test.iloc[:,i],y_pred[:,i]))
        print(accuracy_score(Y_test.iloc[:,i],y_pred[:,i]))
    
def save_model(model, model_filepath):
    '''Save Model'''
    pickle.dump(model, open(model_filepath, "wb"))

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