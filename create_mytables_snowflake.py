# create_mytables_snowflake.py
from pathlib import Path
import os
import pandas as pd
from sqlalchemy import create_engine, text
from snowflake.sqlalchemy import URL, TIMESTAMP_NTZ, VARCHAR, NUMBER
from snowflake.connector.pandas_tools import pd_writer

# --- .env ---
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.getenv("ENV_FILE", ".env"), override=False)
except Exception:
    pass

# ---------- CONFIG ----------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_DIR = BASE_DIR / "csv_files"              # 👈 default to ./csv_files
DATA_DIR = Path(os.getenv("DATA_DIR", DEFAULT_CSV_DIR)).resolve()  # allow override via .env

# Read Snowflake creds & context from env (required)
SF_ACCOUNT   = (os.getenv("SF_ACCOUNT")   or "").strip()
SF_USER      = (os.getenv("SF_USER")      or "").strip()
SF_PASSWORD  = (os.getenv("SF_PASSWORD")  or "").strip()
SF_WAREHOUSE = (os.getenv("SF_WAREHOUSE") or "").strip()
SF_DATABASE  = (os.getenv("SF_DATABASE")  or "").strip()
SF_SCHEMA    = (os.getenv("SF_SCHEMA")    or "").strip()
SF_ROLE      = (os.getenv("SF_ROLE")      or "").strip()  # optional

REQUIRED = {
    "SF_ACCOUNT": SF_ACCOUNT, "SF_USER": SF_USER, "SF_PASSWORD": SF_PASSWORD,
    "SF_WAREHOUSE": SF_WAREHOUSE, "SF_DATABASE": SF_DATABASE, "SF_SCHEMA": SF_SCHEMA,
}
missing = [k for k, v in REQUIRED.items() if not v]
if missing:
    raise ValueError(f"Missing Snowflake env vars: {', '.join(missing)}")

DB_NAME = SF_DATABASE
SCHEMA  = SF_SCHEMA

def p(name: str) -> str:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return str(path)

def make_url(**kwargs):
    clean = {k: v for k, v in kwargs.items() if v}
    return URL(**clean)

def to_upper(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).upper() for c in df.columns]
    return df

def to_iso_timestamps(df: pd.DataFrame, cols) -> pd.DataFrame:
    """
    Parse timestamps -> ISO 'YYYY-MM-DD HH:MM:SS' strings. NULLs preserved.
    """
    out = df.copy()
    for c in cols:
        if c in out.columns:
            s = pd.to_datetime(out[c], errors="coerce")
            try:
                s = s.dt.tz_localize(None)
            except Exception:
                pass
            out[c] = s.dt.strftime("%Y-%m-%d %H:%M:%S")
            out.loc[s.isna(), c] = None
    return out

print(f"📁 Using DATA_DIR: {DATA_DIR}")

# ---------- ENSURE WAREHOUSE / DB / SCHEMA ----------
server_engine = create_engine(make_url(
    account=SF_ACCOUNT, user=SF_USER, password=SF_PASSWORD,
    warehouse=SF_WAREHOUSE, role=SF_ROLE or None,
))
with server_engine.begin() as conn:
    conn.execute(text("ALTER SESSION SET TIMESTAMP_TYPE_MAPPING = 'TIMESTAMP_NTZ'"))
    conn.execute(text("ALTER SESSION SET DATE_INPUT_FORMAT = 'AUTO'"))
    conn.execute(text("ALTER SESSION SET TIMESTAMP_INPUT_FORMAT = 'AUTO'"))

    try:
        conn.execute(text(
            f"CREATE WAREHOUSE IF NOT EXISTS {SF_WAREHOUSE} "
            "WITH WAREHOUSE_SIZE='XSMALL' AUTO_SUSPEND=60 AUTO_RESUME=TRUE"
        ))
    except Exception as e:
        print(f"ℹ️ Skipping CREATE WAREHOUSE (permission?): {e}")
    try:
        conn.execute(text(f"ALTER WAREHOUSE {SF_WAREHOUSE} RESUME"))
    except Exception as e:
        print(f"ℹ️ Skipping RESUME WAREHOUSE: {e}")

    conn.execute(text(f"USE WAREHOUSE {SF_WAREHOUSE}"))
    try:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}"))
    except Exception as e:
        print(f"ℹ️ Could not CREATE DATABASE {DB_NAME}: {e}")
    try:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {DB_NAME}.{SCHEMA}"))
    except Exception as e:
        print(f"ℹ️ Could not CREATE SCHEMA {DB_NAME}.{SCHEMA}: {e}")

    conn.execute(text(f"USE DATABASE {DB_NAME}"))
    conn.execute(text(f"USE SCHEMA {SCHEMA}"))
    role, wh, db, sch = conn.execute(text(
        "select current_role(), current_warehouse(), current_database(), current_schema()"
    )).fetchone()
    print(f"🔎 Context — role={role}, warehouse={wh}, db={db}, schema={sch}")

print(f"✅ Ensured: {DB_NAME}.{SCHEMA} and warehouse {SF_WAREHOUSE}")

# ---------- CONNECT TO TARGET CONTEXT ----------
engine = create_engine(make_url(
    account=SF_ACCOUNT, user=SF_USER, password=SF_PASSWORD,
    warehouse=SF_WAREHOUSE, role=SF_ROLE or None,
    database=DB_NAME, schema=SCHEMA,
))
WRITE_KW = dict(if_exists="replace", index=False, method=pd_writer)

# ---- ORDERS ----
orders = pd.read_csv(p('olist_orders_dataset.csv'))
orders = to_upper(orders)
orders = to_iso_timestamps(orders, [
    'ORDER_PURCHASE_TIMESTAMP','ORDER_APPROVED_AT',
    'ORDER_DELIVERED_CARRIER_DATE','ORDER_DELIVERED_CUSTOMER_DATE',
    'ORDER_ESTIMATED_DELIVERY_DATE'
])
orders.to_sql('orders', engine, dtype={
    'ORDER_ID': VARCHAR(64),
    'CUSTOMER_ID': VARCHAR(64),
    'ORDER_STATUS': VARCHAR(32),
    'ORDER_PURCHASE_TIMESTAMP': TIMESTAMP_NTZ(),
    'ORDER_APPROVED_AT': TIMESTAMP_NTZ(),
    'ORDER_DELIVERED_CARRIER_DATE': TIMESTAMP_NTZ(),
    'ORDER_DELIVERED_CUSTOMER_DATE': TIMESTAMP_NTZ(),
    'ORDER_ESTIMATED_DELIVERY_DATE': TIMESTAMP_NTZ(),
}, **WRITE_KW)
print("✅ Loaded: orders")

# ---- ORDER_PAYMENTS ----
op = pd.read_csv(p('olist_order_payments_dataset.csv'))
op = to_upper(op)
op.to_sql('order_payments', engine, dtype={
    'ORDER_ID': VARCHAR(64),
    'PAYMENT_SEQUENTIAL': NUMBER(38, 0),
    'PAYMENT_TYPE': VARCHAR(32),
    'PAYMENT_INSTALLMENTS': NUMBER(38, 0),
    'PAYMENT_VALUE': NUMBER(12, 2),
}, **WRITE_KW)
print("✅ Loaded: order_payments")

# ---- ORDER_ITEMS ----
oi = pd.read_csv(p('olist_order_items_dataset.csv'))
oi = to_upper(oi)
oi = to_iso_timestamps(oi, ['SHIPPING_LIMIT_DATE'])
oi.to_sql('order_items', engine, dtype={
    'ORDER_ID': VARCHAR(64),
    'ORDER_ITEM_ID': NUMBER(38, 0),
    'PRODUCT_ID': VARCHAR(64),
    'SELLER_ID': VARCHAR(64),
    'SHIPPING_LIMIT_DATE': TIMESTAMP_NTZ(),
    'PRICE': NUMBER(12, 2),
    'FREIGHT_VALUE': NUMBER(12, 2),
}, **WRITE_KW)
print("✅ Loaded: order_items")

# ---- ORDER_REVIEWS ----
orv = pd.read_csv(p('olist_order_reviews_dataset.csv'))
orv = to_upper(orv)
orv = to_iso_timestamps(orv, ['REVIEW_CREATION_DATE','REVIEW_ANSWER_TIMESTAMP'])
orv.to_sql('order_reviews', engine, dtype={
    'REVIEW_ID': VARCHAR(64),
    'ORDER_ID': VARCHAR(64),
    'REVIEW_SCORE': NUMBER(38, 0),
    'REVIEW_COMMENT_TITLE': VARCHAR(255),
    'REVIEW_COMMENT_MESSAGE': VARCHAR(16777216),
    'REVIEW_CREATION_DATE': TIMESTAMP_NTZ(),
    'REVIEW_ANSWER_TIMESTAMP': TIMESTAMP_NTZ(),
}, **WRITE_KW)
print("✅ Loaded: order_reviews")

# ---- CUSTOMER ----
cust = pd.read_csv(p('olist_customers_dataset.csv'))
cust = to_upper(cust)
cust.to_sql('customer', engine, dtype={
    'CUSTOMER_ID': VARCHAR(64),
    'CUSTOMER_UNIQUE_ID': VARCHAR(64),
    'CUSTOMER_ZIP_CODE_PREFIX': NUMBER(38, 0),
    'CUSTOMER_CITY': VARCHAR(128),
    'CUSTOMER_STATE': VARCHAR(4),
}, **WRITE_KW)
print("✅ Loaded: customer")

# ---- PRODUCTS ----
prod = pd.read_csv(p('olist_products_dataset.csv'))
for col in [
    'product_name_lenght','product_description_lenght','product_photos_qty',
    'product_weight_g','product_length_cm','product_height_cm','product_width_cm'
]:
    prod[col] = pd.to_numeric(prod[col], errors='coerce').astype('Int64')
prod = to_upper(prod)
prod.to_sql('products', engine, dtype={
    'PRODUCT_ID': VARCHAR(64),
    'PRODUCT_CATEGORY_NAME': VARCHAR(128),
    'PRODUCT_NAME_LENGHT': NUMBER(38, 0),
    'PRODUCT_DESCRIPTION_LENGHT': NUMBER(38, 0),
    'PRODUCT_PHOTOS_QTY': NUMBER(38, 0),
    'PRODUCT_WEIGHT_G': NUMBER(38, 0),
    'PRODUCT_LENGTH_CM': NUMBER(38, 0),
    'PRODUCT_HEIGHT_CM': NUMBER(38, 0),
    'PRODUCT_WIDTH_CM': NUMBER(38, 0),
}, **WRITE_KW)
print("✅ Loaded: products")

# ---- SELLERS ----
sel = pd.read_csv(p('olist_sellers_dataset.csv'))
sel = to_upper(sel)
sel.to_sql('sellers', engine, dtype={
    'SELLER_ID': VARCHAR(64),
    'SELLER_ZIP_CODE_PREFIX': NUMBER(38, 0),
    'SELLER_CITY': VARCHAR(128),
    'SELLER_STATE': VARCHAR(4),
}, **WRITE_KW)
print("✅ Loaded: sellers")

# ---- CATEGORY_TRANSLATION ----
ct = pd.read_csv(p('product_category_name_translation.csv'))
ct = to_upper(ct)
ct.to_sql('category_translation', engine, dtype={
    'PRODUCT_CATEGORY_NAME': VARCHAR(128),
    'PRODUCT_CATEGORY_NAME_ENGLISH': VARCHAR(128),
}, **WRITE_KW)
print("✅ Loaded: category_translation")

# ---------- POST-LOAD: KEYS & RELATIONSHIPS ----------
ddl = [
    "ALTER TABLE ORDERS                 ADD CONSTRAINT PK_ORDERS                 PRIMARY KEY (ORDER_ID)",
    "ALTER TABLE CUSTOMER               ADD CONSTRAINT PK_CUSTOMER               PRIMARY KEY (CUSTOMER_ID)",
    "ALTER TABLE PRODUCTS               ADD CONSTRAINT PK_PRODUCTS               PRIMARY KEY (PRODUCT_ID)",
    "ALTER TABLE SELLERS                ADD CONSTRAINT PK_SELLERS                PRIMARY KEY (SELLER_ID)",
    "ALTER TABLE CATEGORY_TRANSLATION   ADD CONSTRAINT PK_CATEGORY_TRANSLATION   PRIMARY KEY (PRODUCT_CATEGORY_NAME)",
    "ALTER TABLE ORDER_PAYMENTS         ADD CONSTRAINT PK_ORDER_PAYMENTS         PRIMARY KEY (ORDER_ID, PAYMENT_SEQUENTIAL)",
    "ALTER TABLE ORDER_ITEMS            ADD CONSTRAINT PK_ORDER_ITEMS            PRIMARY KEY (ORDER_ID, ORDER_ITEM_ID)",
    "ALTER TABLE ORDER_REVIEWS          ADD CONSTRAINT PK_ORDER_REVIEWS          PRIMARY KEY (REVIEW_ID)",

    "ALTER TABLE ORDERS           ADD CONSTRAINT FK_ORDERS_CUSTOMER        FOREIGN KEY (CUSTOMER_ID) REFERENCES CUSTOMER(CUSTOMER_ID)",
    "ALTER TABLE ORDER_PAYMENTS   ADD CONSTRAINT FK_OP_ORDER               FOREIGN KEY (ORDER_ID)    REFERENCES ORDERS(ORDER_ID)",
    "ALTER TABLE ORDER_ITEMS      ADD CONSTRAINT FK_OI_ORDER               FOREIGN KEY (ORDER_ID)    REFERENCES ORDERS(ORDER_ID)",
    "ALTER TABLE ORDER_ITEMS      ADD CONSTRAINT FK_OI_PRODUCT             FOREIGN KEY (PRODUCT_ID)  REFERENCES PRODUCTS(PRODUCT_ID)",
    "ALTER TABLE ORDER_ITEMS      ADD CONSTRAINT FK_OI_SELLER              FOREIGN KEY (SELLER_ID)   REFERENCES SELLERS(SELLER_ID)",
    "ALTER TABLE ORDER_REVIEWS    ADD CONSTRAINT FK_ORV_ORDER              FOREIGN KEY (ORDER_ID)    REFERENCES ORDERS(ORDER_ID)",
    "ALTER TABLE PRODUCTS         ADD CONSTRAINT FK_PRODUCT_CATEGORY       FOREIGN KEY (PRODUCT_CATEGORY_NAME) REFERENCES CATEGORY_TRANSLATION(PRODUCT_CATEGORY_NAME)",

    "ALTER TABLE CUSTOMER         ADD CONSTRAINT UQ_CUSTOMER_UNIQUE_ID     UNIQUE (CUSTOMER_UNIQUE_ID)"
]

with engine.begin() as conn:
    for stmt in ddl:
        try:
            conn.execute(text(stmt))
        except Exception as e:
            print(f"ℹ️ Constraint DDL skipped ({e}) -> {stmt}")

print(f"🔗 Keys & relationships declared on {DB_NAME}.{SCHEMA}.")

# ---------- OPTIONAL: quick integrity checks ----------
with engine.connect() as conn:
    checks = {
        "orders -> customer": """
            SELECT COUNT(*) AS orphans
            FROM ORDERS o
            LEFT JOIN CUSTOMER c ON c.CUSTOMER_ID = o.CUSTOMER_ID
            WHERE c.CUSTOMER_ID IS NULL
        """,
        "order_items -> orders": """
            SELECT COUNT(*) FROM ORDER_ITEMS oi
            LEFT JOIN ORDERS o ON o.ORDER_ID = oi.ORDER_ID
            WHERE o.ORDER_ID IS NULL
        """,
        "order_items -> products": """
            SELECT COUNT(*) FROM ORDER_ITEMS oi
            LEFT JOIN PRODUCTS p ON p.PRODUCT_ID = oi.PRODUCT_ID
            WHERE p.PRODUCT_ID IS NULL
        """,
        "order_items -> sellers": """
            SELECT COUNT(*) FROM ORDER_ITEMS oi
            LEFT JOIN SELLERS s ON s.SELLER_ID = oi.SELLER_ID
            WHERE s.SELLER_ID IS NULL
        """,
        "order_payments -> orders": """
            SELECT COUNT(*) FROM ORDER_PAYMENTS op
            LEFT JOIN ORDERS o ON o.ORDER_ID = op.ORDER_ID
            WHERE o.ORDER_ID IS NULL
        """,
        "order_reviews -> orders": """
            SELECT COUNT(*) FROM ORDER_REVIEWS r
            LEFT JOIN ORDERS o ON o.ORDER_ID = r.ORDER_ID
            WHERE o.ORDER_ID IS NULL
        """,
        "products -> category_translation": """
            SELECT COUNT(*) FROM PRODUCTS p
            LEFT JOIN CATEGORY_TRANSLATION c
              ON c.PRODUCT_CATEGORY_NAME = p.PRODUCT_CATEGORY_NAME
            WHERE p.PRODUCT_CATEGORY_NAME IS NOT NULL
              AND c.PRODUCT_CATEGORY_NAME IS NULL
        """
    }
    for name, sql in checks.items():
        n = conn.execute(text(sql)).scalar()
        print(f"🔎 Orphans ({name}): {n}")

print(f"🎉 All tables loaded into {DB_NAME}.{SCHEMA} with keys/relationships.")
