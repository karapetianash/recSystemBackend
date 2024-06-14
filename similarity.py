import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(ratings_df):
    user_movie_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    similarity_matrix = cosine_similarity(user_movie_matrix)
    similarity_df = pd.DataFrame(similarity_matrix, index=user_movie_matrix.index, columns=user_movie_matrix.index)
    return similarity_df


if __name__ == "__main__":
    ratings_df = pd.read_csv('../data/ratings_normalized.csv')
    similarity_df = calculate_similarity(ratings_df)
    similarity_df.to_csv('../data/user_similarity.csv')
