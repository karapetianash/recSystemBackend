import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import os
from tqdm import tqdm


def calculate_similarity(ratings_df, batch_size=100, top_n=5):
    # Преобразуем данные в разреженную матрицу пользователь-фильм
    user_ids = ratings_df['userId'].unique()
    movie_ids = ratings_df['movieId'].unique()
    user_idx = {user: idx for idx, user in enumerate(user_ids)}
    movie_idx = {movie: idx for idx, movie in enumerate(movie_ids)}

    row = ratings_df['userId'].map(user_idx)
    col = ratings_df['movieId'].map(movie_idx)
    data = ratings_df['rating'].astype(np.float32)
    user_movie_matrix_sparse = csr_matrix((data, (row, col)), shape=(len(user_ids), len(movie_ids)))

    # Создаем директорию для хранения промежуточных результатов
    if not os.path.exists('similarity_chunks'):
        os.makedirs('similarity_chunks')

    # Функция для вычисления косинусного сходства по частям
    def compute_similarity_by_batch(start_idx, end_idx):
        batch = user_movie_matrix_sparse[start_idx:end_idx]
        return cosine_similarity(batch, user_movie_matrix_sparse)

    # Разделение матрицы на мини-пакеты и вычисление косинусного сходства
    n_users = user_movie_matrix_sparse.shape[0]

    top_recommendations = {}

    for start in tqdm(range(0, n_users, batch_size), desc="Calculating user similarity", unit="batch"):
        end = min(start + batch_size, n_users)
        similarity_chunk = compute_similarity_by_batch(start, end)

        for i, user_similarities in enumerate(similarity_chunk):
            user_id = user_ids[start + i]
            similar_users = np.argsort(-user_similarities)[:top_n]
            top_recommendations[user_id] = [(user_ids[sim_user], user_similarities[sim_user]) for sim_user in
                                            similar_users]

    # Очистка промежуточных файлов
    if os.path.exists('similarity_chunks'):
        for file in os.listdir('similarity_chunks'):
            os.remove(os.path.join('similarity_chunks', file))

    return top_recommendations


if __name__ == "__main__":
    ratings_df = pd.read_csv('../data/ratings_normalized.csv')
    similarity_df = calculate_similarity(ratings_df)
    similarity_df.to_csv('../data/user_similarity.csv')
