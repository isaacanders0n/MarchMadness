import pandas as pd
import os
import numpy as np


def read_to_one_frame(path) -> pd.DataFrame:
    """Reads all csv files in a directory and concatenates them into a single DataFrame"""
    df = pd.DataFrame()
    for file in os.listdir(path):
        if not file.endswith('20.csv'):
            year = '20' + str(file[-6:-4])
            file = os.path.join(path, file)
            if file.endswith('.csv'):
                currYear = pd.read_csv(file)
                currYear['YEAR'] = year
                df = pd.concat([df, currYear])
    df.drop('EFGD_D', axis=1, inplace=True)
    return df

def clean_NA(df) -> pd.DataFrame:
    return df.replace(['NA', 'N/A'], None)

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
        'R68': 8,
    }

    df['POSTSEASON'] = df['POSTSEASON'].map(rankings)
    return df


def arrange_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rearranges columns"""
    col_order = list(df.columns[:-3]) + ['SEED', 'YEAR', 'POSTSEASON']
    return df[col_order]


def create_made_postseason(df):
    """Creates a new column that indicates whether a team made the postseason"""
    df['MADE_POSTSEASON'] = np.where(pd.isna(df['POSTSEASON']), 0, 1)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans data"""
    df = clean_NA(df)
    df = encode_postseason(df)
    df = arrange_cols(df)
    df = create_made_postseason(df)
    return df


def read_and_clean(path: str) -> pd.DataFrame:
    """Reads and cleans data"""
    df = read_to_one_frame(path)
    return clean_data(df)

def createW_L(df):
    df[['W', 'L']] = df['RECORD'].str.split('-', expand=True)
    df.drop('RECORD', axis=1, inplace=True)
    return df

def arrange_validation_set(df):
    cols = ["TEAM","CONF","W","L","ADJOE","ADJDE","BARTHAG","EFG_O",
            "TOR","TORD","ORB","DRB","FTR","FTRD","2P_O","2P_D","3P_O","3P_D",
            "ADJ_T","WAB"]
    df.rename(columns = {'ADJT': 'ADJ_T'}, inplace = True)

    return df[cols]