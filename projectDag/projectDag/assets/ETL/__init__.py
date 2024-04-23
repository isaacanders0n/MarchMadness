import pandas as pd
import sys
import os
from dagster import asset, job
from ncaa_mod import cleaning as c
from dagster import AssetExecutionContext, MetadataValue, asset, MaterializeResult, AssetIn
import plotly.express as px
import plotly.io as pio
from io import BytesIO
import base64

DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data'))

# @asset 
# def ncaa_rankings() -> None:
#     '''Table containing team statistics for all NCAA mbb teams for years 2013-2023'''
#     df = c.read_to_one_frame(DATA_FOLDER)
#     print(f'ingested dataframe with {len(df)} rows')
#     df.to_csv(f'{DATA_FOLDER}/processed/concatenated_data.csv', index = False)

#     return MaterializeResult(
#         metadata={
#             "num_records": len(df),  # Metsadata can be any key-value pair
#             "preview": MetadataValue.md(df.head().to_markdown()),
#             # The `MetadataValue` class has useful static methods to build Metadata
#         }
#     )


@asset(ins = {'ncaa_rankings': AssetIn('ncaa_rankings')})
def ncaa_cleaned(ncaa_rankings):
    '''Cleaned data'''
    df = pd.read_csv(f'{DATA_FOLDER}/processed/concatenated_data.csv')
    print(type(df))
    df = c.clean_data(df)
    df.to_csv(f'{DATA_FOLDER}/processed/cleaned_data.csv')

    return df


@asset(deps=[ncaa_cleaned])
def parameter_tuning():
    '''Klein this is where you define a function which trims the cleaned dataset to only contain the parameters you care about'''
    return pd.DataFrame()
