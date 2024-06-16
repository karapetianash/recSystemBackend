import sqlite3
import pandas as pd
from database import create_tables, insert_movies, insert_ratings
from preprocess import normalize_ratings, check_data_quality, split_data
from similarity import calculate_similarity
from recommendations import get_recommendations, save_recommendations, fetch_recommendations, precision_at_k, recall_at_k
from sklearn.metrics import mean_squared_error


def evaluate_model(test_df, recommendations, k=10):
    y_true = []
    y_pred = []
    precision_list = []
    recall_list = []

    for user_id in test_df['userId'].unique():
        user_test_df = test_df[test_df['userId'] == user_id]
        actual_movies = user_test_df['movieId'].tolist()
        recommended_movies = [rec[0] for rec in recommendations.get(user_id, [])]

        y_true.extend(user_test_df['rating'])
        y_pred.extend([next((r[1] for r in recommendations.get(user_id, []) if r[0] == movie_id), 0) for movie_id in
                       actual_movies])

        precision_list.append(precision_at_k(actual_movies, recommended_movies, k))
        recall_list.append(recall_at_k(actual_movies, recommended_movies, k))

    rmse = mean_squared_error(y_true, y_pred, squared=False)
    precision = sum(precision_list) / len(precision_list)
    recall = sum(recall_list) / len(recall_list)

    print(f'RMSE: {rmse}')
    print(f'Precision@{k}: {precision}')
    print(f'Recall@{k}: {recall}')


def main():
    print("Creating database and tables...")
    conn = sqlite3.connect('../data/recommendations.db')
    create_tables(conn)

    print("Loading data...")
    movies_df = pd.read_csv('../data/movies.csv')
    train_ratings_df = pd.read_csv('../data/train_ratings.csv')
    test_ratings_df = pd.read_csv('../data/test_ratings.csv')

    print("Checking data quality...")
    check_data_quality(train_ratings_df)

    print("Normalizing ratings...")
    train_ratings_df = normalize_ratings(train_ratings_df)

    print("Inserting data into database...")
    movies_df = movies_df[['movieId', 'title']]  # Remove the genres column
    insert_movies(conn, movies_df)
    insert_ratings(conn, train_ratings_df)

    print("Calculating user similarity...")
    similarity_df = calculate_similarity(train_ratings_df)

    print("Generating recommendations...")
    all_recommendations = {}
    for user_id in train_ratings_df['userId'].unique():
        all_recommendations[user_id] = get_recommendations(user_id, train_ratings_df, similarity_df, movies_df)

    print("Saving recommendations...")
    save_recommendations(conn, all_recommendations)

    print("Fetching recommendations from database...")
    recommendations = fetch_recommendations(conn)
    print(recommendations)

    print("Evaluating model...")
    evaluate_model(test_ratings_df, all_recommendations)

    conn.close()


if __name__ == "__main__":
    main()
