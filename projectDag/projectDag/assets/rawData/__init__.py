import pandas as pd
import sys
import os
from dagster import asset
from ncaa_mod import cleaning as c

@asset 
def ncaa_rankings():
    '''Table containing team statistics for all NCAA mbb teams for years 2013-2023'''
    df = c.read_and_clean('/Users/isaacanderson/Desktop/CSE314/MarchMadness/data')
    return df 