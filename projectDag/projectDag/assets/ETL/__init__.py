import pandas as pd
import sys
import os
from dagster import asset, job
from ncaa_mod import cleaning as c
from dagster import AssetExecutionContext, MetadataValue, asset, MaterializeResult

DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data'))

@asset 
def ncaa_rankings() -> None:
    '''Table containing team statistics for all NCAA mbb teams for years 2013-2023'''
    df = c.read_to_one_frame(DATA_FOLDER)
    print(f'ingested dataframe with {len(df)} rows')
    df.to_csv('../data/concatenated_data.csv')


@asset(deps = [ncaa_rankings])
def ncaa_cleaned():
    '''Cleaned data'''
    df = pd.read_csv('../data/concatenated_data.csv')
    df = c.clean_data(df)
    print(f'cleaned dataframe with {len(df)} rows')
    return MaterializeResult(
        metadata={
            "num_records": len(df),  # Metadata can be any key-value pair
            "preview": MetadataValue.md(df.head().to_markdown()),
            # The `MetadataValue` class has useful static methods to build Metadata
        }
    )
