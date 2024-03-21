import pandas as pd
import os
import numpy as np


def read_to_one_frame(path) -> pd.DataFrame:
    """Reads all csv files in a directory and concatenates them into a single DataFrame"""
    df = pd.DataFrame()
    for file in os.listdir(path):
        year = '20' + file[-2:]
        file = os.path.join(path, file)
        if file.endswith('.csv'):
            currYear = pd.read_csv(file)
            currYear['YEAR'] = year
            df = pd.concat([df, currYear])
    return df


def encode_postseason(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes postseason games as 1 and regular season games as 0"""
    rankings = {
        'Champions': 1,
        '2ND': 2,
        'F4': 3, 
        'E8': 4,
        'S16': 5,
        'R32': 6,
        'R64': 7,
    }

    df['POSTSEASON'] = df['POSTSEASON'].map(rankings)
    return df


def arrange_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rearranges columns"""
    return df.iloc[:, :-2].join(df.iloc[:, -1]).join(df.iloc[:, -2])


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans data"""
    df = encode_postseason(df)
    df = arrange_cols(df)
    return df


def create_made_postseason(df):
    """Creates a new column that indicates whether a team made the postseason"""
    df['MADE_POSTSEASON'] = np.where(df['POSTSEASON'] == 'NA', 0, 1)
    return df


