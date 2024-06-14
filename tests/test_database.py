import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sqlite3
import pandas as pd
import pytest
from database import create_tables, insert_movies, insert_ratings


@pytest.fixture
def db_connection():
    conn = sqlite3.connect(':memory:')
    create_tables(conn)
    yield conn
    conn.close()


def test_insert_movies(db_connection):
    movies_df = pd.DataFrame({
        'movieId': [1, 2],
        'title': ['Movie 1', 'Movie 2']
    })
    insert_movies(db_connection, movies_df)
    cursor = db_connection.cursor()
    cursor.execute('SELECT * FROM movies')
    assert cursor.fetchall() == [(1, 'Movie 1'), (2, 'Movie 2')]


def test_insert_ratings(db_connection):
    ratings_df = pd.DataFrame({
        'userId': [1, 2],
        'movieId': [1, 2],
        'rating': [4.0, 5.0]
    })
    insert_ratings(db_connection, ratings_df)
    cursor = db_connection.cursor()
    cursor.execute('SELECT * FROM ratings')
    assert cursor.fetchall() == [(1, 1, 4.0), (2, 2, 5.0)]
