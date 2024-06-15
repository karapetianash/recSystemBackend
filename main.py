import sqlite3
import pandas as pd
from database import create_tables, insert_movies, insert_ratings
from preprocess import normalize_ratings, check_data_quality
from similarity import calculate_similarity
from recommendations import get_recommendations, save_recommendations, fetch_recommendations


def main():
    conn = sqlite3.connect('../data/recommendations.db')
    create_tables(conn)

    movies_df = pd.read_csv('../data/movies.csv')
    ratings_df = pd.read_csv('../data/ratings.csv')

    check_data_quality(ratings_df)
    ratings_df = normalize_ratings(ratings_df)

    insert_movies(conn, movies_df)
    insert_ratings(conn, ratings_df)

    similarity_df = calculate_similarity(ratings_df)

    all_recommendations = {}
    for user_id in ratings_df['userId'].unique():
        all_recommendations[user_id] = get_recommendations(user_id, ratings_df, similarity_df, movies_df)

    save_recommendations(conn, all_recommendations)
    recommendations = fetch_recommendations(conn)
    print(recommendations)
    conn.close()


if __name__ == "__main__":
    main()
