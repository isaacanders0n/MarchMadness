import pandas as pd
from dagster import asset

@asset
def grab_dataset():
    df = pd.read_csv('data/2019.csv')
    return df