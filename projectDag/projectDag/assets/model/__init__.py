import pandas as pd
from dagster import asset
from ncaa_mod import scraping as s


@asset
def cleaned_test_data():
    df = s.join()
    return df