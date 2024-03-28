import pandas as pd
import sys
import os
from dagster import asset

@asset 
def hello():
    return pd.DataFrame({'hello': ['world']})


@asset
def goodbye():

