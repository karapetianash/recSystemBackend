import sqlite3
import pandas as pd


def create_connection(db_file):
    """Создание подключения к SQLite базе данных"""
    conn = sqlite3.connect(db_file)
    return conn


def create_tables(conn):
    """Создание таблиц movies и ratings"""
    with conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS movies (
                movieId INTEGER PRIMARY KEY,
                title TEXT NOT NULL
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
    conn.commit()


def load_data_to_db(conn, movies_file, ratings_file):
    """Загрузка данных из CSV файлов в таблицы SQLite"""
    movies = pd.read_csv(movies_file)
    ratings = pd.read_csv(ratings_file)

    movies.to_sql('movies', conn, if_exists='replace', index=False)
    ratings.to_sql('ratings', conn, if_exists='replace', index=False)
