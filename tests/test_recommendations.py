import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import sqlite3
import pytest
from recommendations import get_recommendations, save_recommendations


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
    print("Ratings DataFrame:")
    print(ratings_df)
    print("\nSimilarity DataFrame:")
    print(similarity_df)

    recommendations = get_recommendations(1, ratings_df, similarity_df, top_n=1)
    print("\nRecommendations:")
    print(recommendations)

    assert len(recommendations) > 0
    assert recommendations[0][0] == 3
    assert recommendations[0][1] > 0


def test_save_recommendations(db_connection):
    recommendations = {1: [(1, 4.5), (2, 4.7)]}
    save_recommendations(db_connection, recommendations)
    cursor = db_connection.cursor()
    cursor.execute('SELECT * FROM recommendations')
    assert cursor.fetchall() == [(1, 1, 4.5), (1, 2, 4.7)]
