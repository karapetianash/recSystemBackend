import sys
import os
import sqlite3
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database import create_tables, insert_movies, insert_ratings
from preprocess import normalize_ratings, check_data_quality
from similarity import calculate_similarity
from recommendations import get_recommendations, save_recommendations, fetch_recommendations


@pytest.fixture
def db_connection():
    conn = sqlite3.connect(':memory:')
    create_tables(conn)
    yield conn
    conn.close()


def test_integration(db_connection):
    # Step 1: Load data
    movies_df = pd.DataFrame({
        'movieId': [1, 2, 3, 4],
        'title': ['Movie 1', 'Movie 2', 'Movie 3', 'Movie 4']
    })
    ratings_df = pd.DataFrame({
        'userId': [1, 1, 2, 2, 2],
        'movieId': [1, 2, 1, 2, 3],
        'rating': [4.0, 5.0, 3.0, 2.0, 5.0]
    })

    insert_movies(db_connection, movies_df)
    insert_ratings(db_connection, ratings_df)

    # Step 2: Preprocess data
    check_data_quality(ratings_df)
    normalized_ratings_df = normalize_ratings(ratings_df)

    # Step 3: Calculate similarity
    similarity_df = calculate_similarity(normalized_ratings_df)

    # Step 4: Generate recommendations for all users
    all_recommendations = {}
    for user_id in normalized_ratings_df['userId'].unique():
        all_recommendations[user_id] = get_recommendations(user_id, normalized_ratings_df, similarity_df, movies_df)

    save_recommendations(db_connection, all_recommendations)

    # Verify the recommendations are saved correctly
    recommendations = fetch_recommendations(db_connection)

    print("\nRecommendations from database:")
    print(recommendations)

    assert len(recommendations) > 0
    for user_id, movie_id, title, predicted_rating in recommendations:
        assert predicted_rating > 0


if __name__ == "__main__":
    pytest.main([__file__])
