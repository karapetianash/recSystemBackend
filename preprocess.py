import pandas as pd


def normalize_ratings(ratings_df):
    min_rating = ratings_df['rating'].min()
    max_rating = ratings_df['rating'].max()
    ratings_df['rating'] = (ratings_df['rating'] - min_rating) / (max_rating - min_rating)
    return ratings_df


def check_data_quality(ratings_df):
    assert ratings_df.isnull().sum().sum() == 0, "Data contains null values"
    assert ratings_df['rating'].between(0, 5).all(), "Ratings are out of bounds"


if __name__ == "__main__":
    ratings_df = pd.read_csv('../data/ratings.csv')
    check_data_quality(ratings_df)
    ratings_df = normalize_ratings(ratings_df)
    ratings_df.to_csv('../data/ratings_normalized.csv', index=False)
