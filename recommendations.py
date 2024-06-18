import sqlite3
import pandas as pd
from tqdm import tqdm


def get_recommendations(user_id, ratings_df, similarity_dict, movies_df, top_n=5):
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    user_movie_ids = set(user_ratings['movieId'].values)
    other_users = similarity_dict[user_id]

    movie_scores = {}
    similarity_sums = {}

    for other_user, similarity_score in other_users:
        other_user_ratings = ratings_df[ratings_df['userId'] == other_user]

        for _, row in other_user_ratings.iterrows():
            if row['movieId'] not in user_movie_ids:
                if row['movieId'] not in movie_scores:
                    movie_scores[row['movieId']] = 0
                    similarity_sums[row['movieId']] = 0
                movie_scores[row['movieId']] += row['rating'] * similarity_score
                similarity_sums[row['movieId']] += similarity_score

    for movie_id in movie_scores:
        movie_scores[movie_id] /= similarity_sums[movie_id] if similarity_sums[movie_id] != 0 else 1

    recommended_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return [(movie_id, movie_scores[movie_id], movies_df[movies_df['movieId'] == movie_id]['title'].values[0]) for
            movie_id, _ in recommended_movies]


def save_recommendations(conn, recommendations):
    cursor = conn.cursor()
    total_users = len(recommendations)
    for idx, (user_id, movie_recommendations) in enumerate(
            tqdm(recommendations.items(), desc="Saving recommendations")):
        for movie_id, predicted_rating, title in movie_recommendations:
            cursor.execute('''
                INSERT OR REPLACE INTO recommendations (userId, movieId, title, predicted_rating)
                VALUES (?, ?, ?, ?)
            ''', (user_id, movie_id, title, predicted_rating))
    conn.commit()


def fetch_recommendations(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM recommendations')
    recommendations = cursor.fetchall()
    return recommendations


def precision_at_k(y_true, y_pred, k):
    if len(y_pred) > k:
        y_pred = y_pred[:k]
    num_relevant = len(set(y_true) & set(y_pred))
    return num_relevant / min(k, len(y_pred))


def recall_at_k(y_true, y_pred, k):
    if len(y_pred) > k:
        y_pred = y_pred[:k]
    num_relevant = len(set(y_true) & set(y_pred))
    return num_relevant / len(y_true)


if __name__ == "__main__":
    conn = sqlite3.connect('../data/recommendations.db')
    ratings_df = pd.read_csv('../data/ratings_normalized.csv')
    similarity_df = pd.read_csv('../data/user_similarity.csv', index_col=0)
    all_recommendations = {}
    for user_id in ratings_df['userId'].unique():
        all_recommendations[user_id] = get_recommendations(user_id, ratings_df, similarity_df)
    save_recommendations(conn, all_recommendations)
    recommendations = fetch_recommendations(conn)
    print(recommendations)
    conn.close()
