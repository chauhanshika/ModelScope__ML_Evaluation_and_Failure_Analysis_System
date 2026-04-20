import sqlite3
import pandas as pd


DB_PATH = "data/predictions.db"


def create_connection():
    return sqlite3.connect(DB_PATH)


def create_table():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        actual INTEGER,
        predicted INTEGER,
        probability REAL
    )
    """)

    conn.commit()
    conn.close()


def insert_data(df: pd.DataFrame):
    conn = create_connection()
    df.to_sql("predictions", conn, if_exists="replace", index=False)
    conn.close()


def query_prediction_distribution():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT predicted, COUNT(*) 
    FROM predictions 
    GROUP BY predicted
    """)

    result = cursor.fetchall()
    conn.close()
    return resultdb.py
