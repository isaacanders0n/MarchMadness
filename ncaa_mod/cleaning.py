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



