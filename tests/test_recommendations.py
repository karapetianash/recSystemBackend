import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import pytest
import sqlite3
from recommendations import get_recommendations, save_recommendations, fetch_recommendations


@pytest.fixture
def db_connection():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE recommendations (
            userId INTEGER,
            movieId INTEGER,
            predicted_rating REAL,
            PRIMARY KEY (userId, movieId)
        )
    ''')
    conn.commit()
    yield conn
    conn.close()


def test_get_recommendations():
    ratings_df = pd.DataFrame({
        'userId': [1, 1, 2, 2, 2],
        'movieId': [1, 2, 1, 2, 3],
        'rating': [4.0, 5.0, 3.0, 2.0, 5.0]
    })
    similarity_df = pd.DataFrame({
        1: [1.0, 0.8],
        2: [0.8, 1.0]
    }, index=[1, 2])
    recommendations = get_recommendations(1, ratings_df, similarity_df)

    assert len(recommendations) > 0
    for movie_id, predicted_rating in recommendations:
        assert predicted_rating > 0


def test_save_recommendations(db_connection):
    recommendations = {
        1: [(3, 4.5), (4, 3.2)],
        2: [(1, 4.7)]
    }
    save_recommendations(db_connection, recommendations)
    fetched_recommendations = fetch_recommendations(db_connection)

    assert len(fetched_recommendations) == 3
    assert fetched_recommendations[0] == (1, 3, 4.5)
    assert fetched_recommendations[1] == (1, 4, 3.2)
    assert fetched_recommendations[2] == (2, 1, 4.7)
