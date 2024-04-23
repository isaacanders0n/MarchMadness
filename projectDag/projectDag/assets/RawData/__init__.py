import pandas as pd
import sys
import os
from dagster import asset, job
from ncaa_mod import cleaning as c
from dagster import AssetExecutionContext, MetadataValue, asset, MaterializeResult
import plotly.express as px
import plotly.io as pio
from ncaa_mod import scraping as s

DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data'))

@asset 
def ncaa_rankings() -> pd.DataFrame:
    '''Table containing team statistics for all NCAA mbb teams for years 2013-2023'''
    df = c.read_to_one_frame(DATA_FOLDER)
    print(f'ingested dataframe with {len(df)} rows')
    df.to_csv(f'{DATA_FOLDER}/processed/concatenated_data.csv', index = False)

    return df


@asset
def test_data():
    df = s.join()
    return df