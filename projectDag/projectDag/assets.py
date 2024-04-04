import pandas as pd
import sys
import os
from dagster import asset
from ncaa_mod import cleaning as c

@asset 
def grab_dataset():
    df = c.read_and_clean('/Users/isaacanderson/Desktop/CSE314/MarchMadness/data')
    print(len(df))
    return df 

