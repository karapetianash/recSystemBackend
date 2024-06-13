from database import create_connection, create_tables, load_data_to_db
from preprocess import preprocess_data
from similarity import calculate_similarity
from recommendations import generate_recommendations, save_recommendations


def main():
    conn = create_connection('/data/recommendations.db')
    create_tables(conn)
    load_data_to_db(conn, '/data/movies.csv', '/data/ratings.csv')
    preprocess_data(conn)
    calculate_similarity(conn)
    recommendations = generate_recommendations(conn)
    save_recommendations(conn, recommendations)
    conn.close()


if __name__ == "__main__":
    main()
