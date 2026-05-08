import os
import psycopg2
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv("SUPABASE_DB_URI")
parsed = urlparse(DB_URI)
conn_params = {
    "dbname": parsed.path[1:],
    "user": parsed.username,
    "password": parsed.password,
    "host": parsed.hostname,
    "port": parsed.port or 5432,
}

tables = [
    "olist_orders", "olist_order_items", "olist_customers",
    "olist_sellers", "olist_products", "olist_order_payments",
    "olist_order_reviews", "olist_geolocation", "product_category_translation"
]

conn = psycopg2.connect(**conn_params)
cur = conn.cursor()

for table in tables:
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    count = cur.fetchone()[0]
    print(f"{table}: {count:,} rows")

cur.close()
conn.close()