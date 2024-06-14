import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from similarity import calculate_similarity


def test_calculate_similarity():
    ratings_df = pd.DataFrame({
        'userId': [1, 1, 2, 2],
        'movieId': [1, 2, 1, 2],
        'rating': [4.0, 5.0, 4.0, 5.0]
    })
    similarity_df = calculate_similarity(ratings_df)
    assert similarity_df.loc[1, 2] == 1.0  # Users have identical ratings
