import pytest
import unittest
from projectDag.assets import EDA, RawData, model

def test_data_concat():
    '''Test that year 2020 was excluded '''
    assert len(RawData.ncaa_rankings() == 3524)

def test_new_data_cleaned():
    '''Test that the new data was scraped accurately'''
    columns = ["TEAM","CONF","W","L","ADJOE","ADJDE","BARTHAG","EFG_O",
            "TOR","TORD","ORB","DRB","FTR","FTRD","2P_O","2P_D","3P_O","3P_D",
            "ADJ_T","WAB"]
    assert (model.cleaned_validation_data().columns == columns)