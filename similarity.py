import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(ratings_df, chunk_size=1000):
    # Преобразуем данные в разреженную матрицу пользователь-фильм
    row = ratings_df['userId'].astype('category').cat.codes
    col = ratings_df['movieId'].astype('category').cat.codes
    data = ratings_df['rating'].astype(np.float32)
    user_movie_matrix_sparse = csr_matrix((data, (row, col)), shape=(row.max() + 1, col.max() + 1))

    # Функция для вычисления косинусного сходства по частям
    def compute_similarity_by_chunk(start_idx, end_idx):
        chunk = user_movie_matrix_sparse[start_idx:end_idx]
        return cosine_similarity(chunk, user_movie_matrix_sparse)

    # Разделение матрицы на части и вычисление косинусного сходства
    n_users = user_movie_matrix_sparse.shape[0]
    similarity_chunks = []

    for start in range(0, n_users, chunk_size):
        end = min(start + chunk_size, n_users)
        similarity_chunk = compute_similarity_by_chunk(start, end)
        similarity_chunks.append(similarity_chunk)

    # Объединяем части в одну матрицу
    user_similarity = np.vstack(similarity_chunks)
    user_ids = ratings_df['userId'].astype('category').cat.categories
    user_similarity_df = pd.DataFrame(user_similarity, index=user_ids, columns=user_ids)

    return user_similarity_df


if __name__ == "__main__":
    ratings_df = pd.read_csv('../data/ratings_normalized.csv')
    similarity_df = calculate_similarity(ratings_df)
    similarity_df.to_csv('../data/user_similarity.csv')
