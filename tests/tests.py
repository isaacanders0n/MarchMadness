import pytest
import pandas as pd
import numpy as np
from ncaa_mod import cleaning

def test_one_frame():
    actual = cleaning.read_to_one_frame("../MarchMadness/data")
    assert len(actual) >= 3000

def test_encode_postseason():
    test_df = pd.DataFrame(
        {
            'POSTSEASON': ['Champions', '2ND', 'F4', 'E8', 'S16', 'R32', 'R64', 'R68']
        }
    )
    actual = cleaning.encode_postseason(test_df)
    pd.testing.assert_frame_equal(
        actual,
        pd.DataFrame(
        {
            'POSTSEASON': [1, 2, 3, 4, 5, 6, 7, 8]
        }
    )
    )

def test_clean_NA():
    test_df = pd.DataFrame(
        {
            'POSTSEASON': ['Champions', '2ND', 'F4', 'NA', 'N/A'],
            'SEED': ['1', '2', '4', 'NA', 'N/A']
        }
    )
    actual = cleaning.clean_NA(test_df)
    pd.testing.assert_frame_equal(
        actual,
        pd.DataFrame(
        {
            'POSTSEASON': ['Champions', '2ND', 'F4', None, None],
            'SEED': ['1', '2', '4', None, None]
        }
    )
    )

def test_cols_order():
    test_df = pd.DataFrame(
        {
            'TEAM': ['Kansas'],
            'CONF': ['B12'],
            'G': ['36'],
            'POSTSEASON': ['R32'],
            'SEED': ['1'],
            'YEAR': ['2023']
        }
    )
    actual = cleaning.arrange_cols(test_df)
    pd.testing.assert_frame_equal(
        actual,
        pd.DataFrame(
        {
            'TEAM': ['Kansas'],
            'CONF': ['B12'],
            'G': ['36'],
            'SEED': ['1'],
            'YEAR': ['2023'],
            'POSTSEASON': ['R32']
        }
    )
    )

def test_made_postseason():
    test_df = pd.DataFrame(
        {
            'POSTSEASON': ['Champions', '2ND', 'F4', 'E8', 'S16', 'R32', 'R64', 'R68', None, None]
        }
    )
    actual = cleaning.create_made_postseason(test_df)
    pd.testing.assert_frame_equal(
        actual,
        pd.DataFrame(
        {
            'POSTSEASON': ['Champions', '2ND', 'F4', 'E8', 'S16', 'R32', 'R64', 'R68', None, None],
            'MADE_POSTSEASON': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
        }
    )
    )
