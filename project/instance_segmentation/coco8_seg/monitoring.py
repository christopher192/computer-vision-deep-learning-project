import io
import psycopg

create_table_statement = """
    DROP TABLE IF EXISTS cv_metrics;
    CREATE TABLE cv_metrics(
        timestamp timestamp,
        image_metadata JSON
    )
"""
host = "localhost"
port = "5432"
user = "postgres"
password = "Password123"
database = "cvops"

def prep_database():
    with psycopg.connect(f"host = {host} port = {port} user = {user} password = {password}", autocommit = True) as conn:
        res = conn.execute(f"SELECT 1 FROM pg_database WHERE datname = '{database}'")
        if len(res.fetchall()) == 0:
            conn.execute("CREATE DATABASE cvops;")
        with psycopg.connect(f"host = {host} port = {port} dbname = {database} user = {user} password = {password}") as conn:
            conn.execute(create_table_statement)

if __name__ == '__main__':
    prep_database()
    print("Hello World!")