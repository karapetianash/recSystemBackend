import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import pytest
from preprocess import normalize_ratings, check_data_quality


def test_normalize_ratings():
    ratings_df = pd.DataFrame({
        'userId': [1, 2],
        'movieId': [1, 2],
        'rating': [3.0, 5.0]
    })
    normalized_df = normalize_ratings(ratings_df.copy())
    assert normalized_df['rating'].min() == 0.0
    assert normalized_df['rating'].max() == 1.0


def test_check_data_quality():
    ratings_df = pd.DataFrame({
        'userId': [1, 2],
        'movieId': [1, 2],
        'rating': [3.0, 5.0]
    })
    check_data_quality(ratings_df)

    ratings_with_nulls = ratings_df.copy()
    ratings_with_nulls.loc[0, 'rating'] = None
    with pytest.raises(AssertionError):
        check_data_quality(ratings_with_nulls)

    ratings_out_of_bounds = ratings_df.copy()
    ratings_out_of_bounds.loc[0, 'rating'] = 6.0
    with pytest.raises(AssertionError):
        check_data_quality(ratings_out_of_bounds)
