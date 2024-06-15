import sqlite3
import pandas as pd


def create_tables(conn):
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS movies (
            movieId INTEGER PRIMARY KEY,
            title TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ratings (
            userId INTEGER,
            movieId INTEGER,
            rating REAL,
            PRIMARY KEY (userId, movieId)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recommendations (
            userId INTEGER,
            movieId INTEGER,
            title TEXT,
            predicted_rating REAL,
            PRIMARY KEY (userId, movieId)
        )
    ''')
    conn.commit()


def insert_movies(conn, movies_df):
    movies_df.to_sql('movies', conn, if_exists='append', index=False)


def insert_ratings(conn, ratings_df):
    ratings_df.to_sql('ratings', conn, if_exists='append', index=False)


if __name__ == "__main__":
    conn = sqlite3.connect('../data/recommendations.db')
    create_tables(conn)
    movies_df = pd.read_csv('../data/movies.csv')
    ratings_df = pd.read_csv('../data/ratings.csv')
    insert_movies(conn, movies_df)
    insert_ratings(conn, ratings_df)
    conn.close()
