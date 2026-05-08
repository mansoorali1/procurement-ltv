import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()

DB_URI = os.getenv("SUPABASE_DB_URI")

# Parse the URI for psycopg2
parsed = urlparse(DB_URI)
conn_params = {
    "dbname": parsed.path[1:],
    "user": parsed.username,
    "password": parsed.password,
    "host": parsed.hostname,
    "port": parsed.port or 5432,
}

CSV_TABLE_MAP = {
    "olist_orders_dataset.csv": "olist_orders",
    "olist_order_items_dataset.csv": "olist_order_items",
    "olist_customers_dataset.csv": "olist_customers",
    "olist_sellers_dataset.csv": "olist_sellers",
    "olist_products_dataset.csv": "olist_products",
    "olist_order_payments_dataset.csv": "olist_order_payments",
    "olist_order_reviews_dataset.csv": "olist_order_reviews",
    "olist_geolocation_dataset.csv": "olist_geolocation",
    "product_category_name_translation.csv": "product_category_translation",
}

RAW_DATA_PATH = "data/raw/"

def get_create_table_sql(df, table_name):
    """Generate CREATE TABLE SQL based on DataFrame dtypes."""
    type_map = {
        "int64": "BIGINT",
        "float64": "DOUBLE PRECISION",
        "bool": "BOOLEAN",
        "datetime64[ns]": "TIMESTAMP",
        "object": "TEXT",
    }
    cols = []
    for col, dtype in zip(df.columns, df.dtypes):
        pg_type = type_map.get(str(dtype), "TEXT")
        cols.append(f'"{col}" {pg_type}')
    return f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(cols)});'

def upload_csv(csv_file: str, table_name: str):
    filepath = os.path.join(RAW_DATA_PATH, csv_file)
    print(f"Loading {csv_file}...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"  Shape: {df.shape}")

    # Replace NaN with None for proper NULL handling in PostgreSQL
    df = df.where(pd.notnull(df), None)

    print(f"  Uploading to table '{table_name}'...")
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()

    # Drop and recreate table
    cur.execute(f'DROP TABLE IF EXISTS "{table_name}";')
    cur.execute(get_create_table_sql(df, table_name))

    # Insert data in chunks
    columns = [f'"{c}"' for c in df.columns]
    insert_sql = f'INSERT INTO "{table_name}" ({", ".join(columns)}) VALUES %s'
    rows = [tuple(row) for row in df.itertuples(index=False, name=None)]

    chunk_size = 1000
    for i in range(0, len(rows), chunk_size):
        execute_values(cur, insert_sql, rows[i:i + chunk_size])
        print(f"  Inserted rows {i} to {min(i + chunk_size, len(rows))}...")

    conn.commit()
    cur.close()
    conn.close()
    print(f"  ✓ Done — {len(df)} rows uploaded.\n")

if __name__ == "__main__":
    for csv_file, table_name in CSV_TABLE_MAP.items():
        upload_csv(csv_file, table_name)
    print("All tables uploaded successfully.")