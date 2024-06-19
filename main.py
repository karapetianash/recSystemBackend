import sqlite3
import pandas as pd
from database import create_tables, insert_movies, insert_ratings
from preprocess import normalize_ratings, check_data_quality, split_data
from similarity import calculate_similarity
from recommendations import get_recommendations, save_recommendations, fetch_recommendations, precision_at_k, recall_at_k
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


def evaluate_model(test_df, recommendations, k=5):
    y_true = []
    y_pred = []
    precision_list = []
    recall_list = []

    user_ids = test_df['userId'].unique()

    print(f"Total users to evaluate: {len(user_ids)}")

    with tqdm(total=len(user_ids), desc="Evaluating model", unit="user") as pbar:
        for user_id in user_ids:
            user_test_df = test_df[test_df['userId'] == user_id]
            actual_movies = user_test_df['movieId'].tolist()
            recommended_movies = recommendations.get(user_id, [])

            y_true.extend(user_test_df['rating'])
            y_pred.extend([next((r[1] for r in recommended_movies if r[0] == movie_id), 0) for movie_id in actual_movies])

            if recommended_movies:
                precision = precision_at_k(actual_movies, [rec[0] for rec in recommended_movies], k)
                recall = recall_at_k(actual_movies, [rec[0] for rec in recommended_movies], k)
                precision_list.append(precision)
                recall_list.append(recall)

            pbar.update(1)

    if y_pred:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    else:
        rmse = 0
    precision = sum(precision_list) / len(precision_list) if precision_list else 0
    recall = sum(recall_list) / len(recall_list) if recall_list else 0

    print(f'RMSE: {rmse}')
    print(f'Precision@{k}: {precision}')
    print(f'Recall@{k}: {recall}')


def fetch_recommendations_with_progress(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM recommendations')
    total_recommendations = cursor.fetchone()[0]

    cursor.execute('SELECT * FROM recommendations')
    recommendations = []
    with tqdm(total=total_recommendations, desc="Fetching recommendations from database") as pbar:
        for row in cursor:
            recommendations.append(row)
            pbar.update(1)
    return recommendations


def create_all_recommendations_dict(recommendations):
    user_ids = set(rec[0] for rec in recommendations)
    all_recommendations_dict = {}
    with tqdm(total=len(user_ids), desc="Creating recommendations dictionary", unit="user") as pbar:
        for user_id in user_ids:
            all_recommendations_dict[user_id] = [(rec[1], rec[3], rec[2]) for rec in recommendations if rec[0] == user_id]
            pbar.update(1)
    return all_recommendations_dict


def main(fetch_only=False):
    conn = sqlite3.connect('../data/recommendations.db')

    if not fetch_only:
        print("Creating database and tables...")
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
        with tqdm(total=len(movies_df) + len(train_ratings_df), desc="Inserting data into database") as pbar:
            movies_df = movies_df[['movieId', 'title']]  # Remove the genres column
            train_ratings_df = train_ratings_df[['userId', 'movieId', 'rating']]  # Remove the timestamp column
            insert_movies(conn, movies_df)
            pbar.update(len(movies_df))
            insert_ratings(conn, train_ratings_df)
            pbar.update(len(train_ratings_df))

        print("Calculating user similarity...")
        similarity_dict = calculate_similarity(train_ratings_df, batch_size=100, top_n=5)

        print("Generating recommendations...")
        all_recommendations = {}
        total_users = len(train_ratings_df['userId'].unique())
        for idx, user_id in enumerate(tqdm(train_ratings_df['userId'].unique(), desc="Generating recommendations")):
            all_recommendations[user_id] = get_recommendations(user_id, train_ratings_df, similarity_dict, movies_df,
                                                               top_n=5)

        print("Saving recommendations...")
        save_recommendations(conn, all_recommendations)

    print("Fetching recommendations from database...")
    recommendations = fetch_recommendations_with_progress(conn)
    print(f"Fetched {len(recommendations)} recommendations")

    print("Creating recommendations dictionary...")
    all_recommendations_dict = create_all_recommendations_dict(recommendations)

    print("Evaluating model...")
    test_ratings_df = pd.read_csv('../data/test_ratings.csv')  # Reload test ratings if necessary
    evaluate_model(test_ratings_df, all_recommendations_dict)

    conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run recommendation system.')
    parser.add_argument('--fetch_only', action='store_true', help='Only fetch recommendations and evaluate the model.')
    args = parser.parse_args()

    main(fetch_only=args.fetch_only)
