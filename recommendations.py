import pandas as pd
import sqlite3


def get_recommendations(user_id, ratings_df, similarity_df, top_n=10):
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    user_movie_ids = set(user_ratings['movieId'].values)
    other_users = similarity_df[user_id].sort_values(ascending=False).index[1:]

    movie_scores = {}
    similarity_sums = {}

    print(f"User {user_id} rated movies: {user_movie_ids}")

    for other_user in other_users:
        other_user_ratings = ratings_df[ratings_df['userId'] == other_user]
        similarity_score = similarity_df.loc[user_id, other_user]

        print(f"Other user {other_user} with similarity score {similarity_score}")

        for _, row in other_user_ratings.iterrows():
            if row['movieId'] not in user_movie_ids:
                if row['movieId'] not in movie_scores:
                    movie_scores[row['movieId']] = 0
                    similarity_sums[row['movieId']] = 0
                movie_scores[row['movieId']] += row['rating'] * similarity_score
                similarity_sums[row['movieId']] += similarity_score

    for movie_id in movie_scores:
        movie_scores[movie_id] /= similarity_sums[movie_id] if similarity_sums[movie_id] != 0 else 1

    recommended_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"Recommended movies for user {user_id}: {recommended_movies}")

    return recommended_movies[:top_n]


def save_recommendations(conn, recommendations):
    cursor = conn.cursor()
    for user_id, movie_recommendations in recommendations.items():
        for movie_id, predicted_rating in movie_recommendations:
            cursor.execute('''
                INSERT INTO recommendations (userId, movieId, predicted_rating)
                VALUES (?, ?, ?)
            ''', (user_id, movie_id, predicted_rating))
    conn.commit()


if __name__ == "__main__":
    conn = sqlite3.connect('../data/recommendations.db')
    ratings_df = pd.read_csv('../data/ratings_normalized.csv')
    similarity_df = pd.read_csv('../data/user_similarity.csv', index_col=0)
    all_recommendations = {}
    for user_id in ratings_df['userId'].unique():
        all_recommendations[user_id] = get_recommendations(user_id, ratings_df, similarity_df)
    save_recommendations(conn, all_recommendations)
    conn.close()
