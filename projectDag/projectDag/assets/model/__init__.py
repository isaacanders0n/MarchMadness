import pandas as pd
from dagster import asset, multi_asset, AssetIn, AssetOut, MaterializeResult
from ncaa_mod import cleaning as c
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
import numpy as np



@asset(ins = {'test_data': AssetIn('test_data')})
def cleaned_validation_data(test_data):
    '''Clean data that was scraped from internet'''
    df = c.createW_L(test_data)
    df = c.arrange_validation_set(df)
    return df


@multi_asset(ins ={'parameter_tuning': AssetIn('parameter_tuning')},
            outs={
                    'X_train': AssetOut(),
                    'X_test': AssetOut(),
                    'y_train': AssetOut(),
                    'y_test': AssetOut()
                })
def split_data(parameter_tuning):
    '''Split the data into training and testing sets'''
    X = parameter_tuning.iloc[:, 4:-5]
    X = csr_matrix(X)
    y = parameter_tuning.iloc[:, -1]
    y = np.array(y) 
    print(f'THIS IS WHAT Y LOOKS LIKE: {y}')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    return X_train, X_test, y_train, y_test

@asset(ins = {'X_train': AssetIn('X_train'),
              'X_test': AssetIn('X_test'),
              'y_train': AssetIn('y_train'),
              'y_test': AssetIn('y_test')})
def bubbleClassifier(X_train, X_test, y_train, y_test):
    '''This is a placeholder until Klein model working'''

    print(type(X_train))
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = lr.predict(X_test)

    # Evaluate the model's accuracy
    accuracy = accuracy_score(y_test, y_pred)
    MaterializeResult(metadata={"accuracy": accuracy})
    return lr

@asset(ins = {'bubbleClassifier': AssetIn('bubbleClassifier'),
              'cleaned_validation_data': AssetIn('cleaned_validation_data')})
def bubbleClassification(bubbleClassifier: LogisticRegression, cleaned_validation_data):
    '''Placeholder for Klein's model'''
    X = cleaned_validation_data.iloc[:, 4:-1]

    output = bubbleClassifier.predict(X)

    return MaterializeResult(metadata={"output": output})