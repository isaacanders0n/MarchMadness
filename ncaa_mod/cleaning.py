import pandas as pd
import os


def read_to_one_frame(path) -> pd.DataFrame:
    """Reads all csv files in a directory and concatenates them into a single DataFrame"""
    df = pd.DataFrame()
    for file in os.listdir(path):
        file = os.path.join(path, file)
        if file.endswith('.csv'):
            df = pd.concat([df, pd.read_csv(file)])
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

    df['postseason'] = df['postseason'].map(rankings)
    return df

def arrange_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rearranges columns"""
    return df.iloc[:, :-2].join(df.iloc[:, -1]).join(df.iloc[:, -2])


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans data"""
    df = encode_postseason(df)
    df = arrange_cols(df)
    return df