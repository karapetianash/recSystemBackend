import pandas as pd
from sklearn.model_selection import train_test_split


def normalize_ratings(ratings_df):
    min_rating = ratings_df['rating'].min()
    max_rating = ratings_df['rating'].max()
    ratings_df['rating'] = (ratings_df['rating'] - min_rating) / (max_rating - min_rating)
    return ratings_df


def check_data_quality(ratings_df):
    assert ratings_df.isnull().sum().sum() == 0, "Data contains null values"
    assert ratings_df['rating'].between(0, 5).all(), "Ratings are out of bounds"


def split_data(ratings_df, test_size=0.2):
    train_df, test_df = train_test_split(ratings_df, test_size=test_size, random_state=42)
    return train_df, test_df


if __name__ == "__main__":
    ratings_df = pd.read_csv('../data/ratings.csv')
    check_data_quality(ratings_df)
    ratings_df = normalize_ratings(ratings_df)
    train_df, test_df = split_data(ratings_df)
    train_df.to_csv('../data/train_ratings.csv', index=False)
    test_df.to_csv('../data/test_ratings.csv', index=False)
