# =============================================================================
# build_knowledgebase.py — Introspects Snowflake to gather column names/dtypes/
# samples per table and asks the LLM to write practical descriptions.
# Writes a pickle: {table_name: [table_desc, [[col, desc], ...]]}
# =============================================================================

import os
import re
import json
import time
import pickle
import tqdm
import pandas as pd
from sqlalchemy import text
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser

from config import get_llm, get_engine, get_knowledgebase_path

# ---- LLM & DB ----
llm = get_llm()
engine = get_engine()
if engine.dialect.name != "snowflake":
    raise RuntimeError("This builder is Snowflake-only. Current engine is not Snowflake.")

# Human-written base descriptions (dataset-specific seeds)
table_description = {
    'order_items': """Contains item-level rows for each order, including seller_id, product_id, item price, and freight value. Each order can have multiple items (and sellers).""",
    'customer': """Contains customer records and their location (city, state).""",
    'order_payments': """Contains payment transactions for orders. A single order can have multiple payments (installments or split methods). payment_value is the amount of that transaction.""",
    'order_reviews': """Contains review scores and timestamps for orders.""",
    'orders': """Contains order lifecycle timestamps (purchase, approved, delivered, estimated), status, and customer_id.""",
    'products': """Contains product-level attributes, including product_category_name (in Portuguese).""",
    'sellers': """Contains seller_id and seller location.""",
    'category_translation': """Maps Portuguese product_category_name to English in product_category_name_english."""
}

# ---- Helpers ----
def _safe_table_name(name: str) -> str:
    if not re.match(r"^[A-Za-z0-9_]+$", name or ""):
        raise ValueError(f"Unsafe table name: {name!r}")
    return name  # unquoted; Snowflake resolves case-insensitively


def sample_table_df(table: str, limit: int = 100) -> pd.DataFrame:
    """Return a small random sample from a Snowflake table."""
    tbl = _safe_table_name(table)
    q = text(f"SELECT * FROM {tbl} ORDER BY RANDOM() LIMIT {int(limit)}")
    return pd.read_sql(q, con=engine)


def column_specs(table: str, df: pd.DataFrame):
    """
    Get column names and data types from INFORMATION_SCHEMA, then attach up to
    5 sample values (from the sample df) for better LLM descriptions.
    Robust to driver/pandas casing differences.
    """
    cols = []

    # Explicit aliases ensure stable labels; quoted to preserve exact case.
    meta_sql = text("""
        SELECT
            COLUMN_NAME AS "COLUMN_NAME",
            DATA_TYPE   AS "DATA_TYPE"
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_CATALOG = CURRENT_DATABASE()
          AND TABLE_SCHEMA  = CURRENT_SCHEMA()
          AND TABLE_NAME    = :t
        ORDER BY ORDINAL_POSITION
    """)
    tval = table.upper()
    with engine.begin() as conn:
        meta = pd.read_sql(meta_sql, conn, params={"t": tval})

    if meta.empty:
        return cols

    # Normalize metadata column labels to uppercase to be extra safe
    meta.columns = [str(c).strip().upper() for c in meta.columns]

    # Prepare df column maps for robust lookup
    df_cols_upper = {c.upper(): c for c in df.columns}
    df_cols_lower = {c.lower(): c for c in df.columns}

    for _, row in meta.iterrows():
        name = str(row["COLUMN_NAME"])
        dtype = str(row["DATA_TYPE"]).upper()

        if name in df.columns:
            col = name
        elif name in df_cols_upper:
            col = df_cols_upper[name]
        elif name.lower() in df_cols_lower:
            col = df_cols_lower[name.lower()]
        else:
            col = None

        if col is not None:
            samples = df[col].dropna().astype(str).drop_duplicates().head(5).tolist()
        else:
            samples = []

        cols.append({"name": name, "dtype": dtype, "samples": samples})
    return cols

# ---- Prompt (Snowflake-aware) ----
template = ChatPromptTemplate.from_messages([
    ("system", """
You are a precise SQL data annotator.

STRICT OUTPUT:
- Return ONLY valid JSON, no prose, no code fences.
- Shape:
  {{
    "table_description": "<concise, factual table description based on inputs>",
    "columns": [
      ["<column_name>", "<detailed, practical description including datatype and 1-2 sample values (and mention there are more values)>"]
    ]
  }}

RULES:
- Do NOT invent column names. Use exactly the names provided in "column_specs".
- Use the datatype provided; do not guess others.
- Be specific and practical; avoid generic marketing phrases.
- Keep each column description helpful for a Text-to-SQL agent (what it represents, how it is used).
- Include 1-2 sample values inline and say more values exist.

DATASET-SPECIFIC MAPPING HINTS (very important; weave these into relevant column descriptions when present):
- "total sales" / "revenue" / "GMV" → sum of order_payments.payment_value.
- Time trends → use orders.order_purchase_timestamp (e.g., DATE_TRUNC('month', orders.order_purchase_timestamp)).
- Optional month label → TO_VARCHAR(orders.order_purchase_timestamp, 'YYYY-MM').
- Seller performance (# orders) → derive from order_items.seller_id with COUNT(DISTINCT order_id).
- Delivery time → orders.order_delivered_customer_date minus orders.order_purchase_timestamp; "late" compares to orders.order_estimated_delivery_date.
- Reviews → order_reviews.review_score; link to orders by order_id.
- English category → category_translation.product_category_name_english joined to products.product_category_name.
"""),
    ("human", """
SQL table description:
{table_desc}

Column specs (names are authoritative — do not invent new names):
{column_specs}

A few random sample rows from this table:
{table_samples}

Return ONLY JSON with keys "table_description" and "columns" as specified.
""")
])

chain = (
    RunnableMap({
        "table_desc": lambda x: x["table_desc"],
        "column_specs": lambda x: x["column_specs"],
        "table_samples": lambda x: x["table_samples"]
    })
    | template
    | llm
    | StrOutputParser()
)

kb_final = {}
for table, tdesc in tqdm.tqdm(table_description.items()):
    df = sample_table_df(table, limit=100)
    specs = column_specs(table, df)
    specs_json = json.dumps(specs, ensure_ascii=False)
    sample_json = df.head(10).to_json(orient="records", force_ascii=False)

    raw = chain.invoke({
        "table_desc": tdesc,
        "column_specs": specs_json,
        "table_samples": sample_json
    }).strip()

    # Be tolerant to occasional extra tokens from the model
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            raise
        obj = json.loads(m.group(0))

    table_desc_final = obj.get("table_description", "").strip()
    columns_pairs = obj.get("columns", [])
    kb_final[table] = [table_desc_final, columns_pairs]
    time.sleep(0.5)  # gentle on API

# Write to configured knowledgebase path
OUT_PATH = get_knowledgebase_path()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "wb") as f:
    pickle.dump(kb_final, f)

print(f"✅ Wrote knowledgebase to: {OUT_PATH}  (tables: {len(kb_final)})")
