#  LLM-Powered AI Agent to facilitate Natural Language Queries in SQL Databases

A small, production‑lean pipeline that turns natural‑language questions into **validated Snowflake SQL**, executes the query, and auto‑generates a **sensible Plotly visualization or table**. It ships with:
- A **knowledge‑base builder** that introspects your Snowflake schema and lets an LLM write practical table/column descriptions.
- A set of **LangChain / LangGraph agents** that route questions, pick tables/columns, extract filters, generate SQL, validate/fix it, and render a chart.
- A minimal **Streamlit UI** to try questions interactively and download the results/SQL/code.
- A **CSV → Snowflake loader** that creates the demo Olist schema in your account from files under `csv_files/`.

> This README uses **placeholders** — replace them with your values locally.

---

## Contents

- [Quick start](#quick-start)
- [Environment (.env) template](#environment-env-template)
- [Load the demo data from CSVs](#load-the-demo-data-from-csvs)
- [How it works](#how-it-works)
- [Project layout](#project-layout)
- [Common tasks](#common-tasks)
- [Troubleshooting](#troubleshooting)
- [Design notes & extensibility](#design-notes--extensibility)
- [Security & data governance](#security--data-governance)
- [License](#license)

---

## Quick start

### 1) Prerequisites
- **Python** 3.10+ (3.11 recommended)
- Network access to **Snowflake** (read‑only is fine for the app; loader needs create table/constraints)
- An **Azure OpenAI** deployment (e.g., `o4-mini`, `gpt-4o`, etc.) compatible with LangChain’s `AzureChatOpenAI`

### 2) Install dependencies
Prefer using the provided requirements file:
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
```
If you install manually, you’ll need at least:
```
pandas tqdm streamlit plotly
langchain-core langchain-openai langgraph
SQLAlchemy snowflake-connector-python snowflake-sqlalchemy
python-dotenv rapidfuzz
```

### 3) Create your `.env`
See the template below. Keep it **local** and **private**.

### 4) Download the CSVs (Olist / Brazilian E‑commerce)
**Download from Kaggle** and put all CSV files into the local folder **`./csv_files`** (create it if missing).  
Dataset: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

- You’ll need a Kaggle account to download. Follow the dataset’s license/terms.
- Our loader **requires** the eight files listed below (geolocation is **optional** and **not used** by the app).

**Alternative via Kaggle CLI** (optional):
```bash
pip install kaggle
# Place your Kaggle API token at:  %USERPROFILE%\.kaggle\kaggle.json  (Windows)
# or ~/.kaggle/kaggle.json (Linux/Mac). File must be readable only by you.
kaggle datasets download -d olistbr/brazilian-ecommerce -p csv_files --unzip
```
> The download includes extra files; the loader will only read the eight required ones.

**Do not commit CSVs** to GitHub. Add this to `.gitignore`:
```
csv_files/
```
If you *must* version large data, consider Git LFS.

### 5) (First‑time) Load the demo data into Snowflake
With the CSVs in `./csv_files/` (or point `DATA_DIR` to another folder), run:
```bash
python create_mytables_snowflake.py
```
This creates the warehouse (if permitted), DB/schema, loads tables, adds PK/FK constraints, and prints integrity checks.

### 6) Build the knowledge‑base (one‑time per schema change)
```bash
python build_knowledgebase.py
```
Expect:
```
✅ Wrote knowledgebase to: ./knowledgebase.pkl  (tables: 8)
```

### 7) Run the Streamlit app
```bash
streamlit run streamlit_chat.py
```
Ask questions like:
- “What is the **monthly trend of total sales**?”  
- “Top **10 sellers** by number of **orders** in **2018**.”  
- “Average **delivery time in days** by **state** for **last month**.”

---

## Environment (.env) template

> The code accepts either a full Azure endpoint path **or** the resource base URL; it will normalize automatically.

```dotenv
# ---------- Azure OpenAI ----------
# Either: https://<resource>.cognitiveservices.azure.com/
# Or a full path (the code will normalize to the base URL):
AZURE_OPENAI_ENDPOINT=<<YOUR_AZURE_OPENAI_ENDPOINT>>
AZURE_OPENAI_DEPLOYMENT=<<YOUR_DEPLOYMENT_NAME>>           # e.g., o4-mini
AZURE_OPENAI_API_VERSION=2025-04-01-preview                 # match your deployment
AZURE_OPENAI_API_KEY=<<YOUR_AZURE_OPENAI_API_KEY>>

# ---------- Snowflake (loader needs create rights; app can be read-only) ----------
SF_ACCOUNT=<<ORG-ACCOUNT>>          # e.g., ABCD-XY12345
SF_USER=<<USERNAME>>
SF_PASSWORD=<<PASSWORD>>
SF_WAREHOUSE=<<WAREHOUSE>>
SF_DATABASE=<<DATABASE>>
SF_SCHEMA=<<SCHEMA>>
SF_ROLE=<<ROLE>>                    # optional

# ---------- Data & knowledge-base ----------
# CSV directory for the loader script. Defaults to ./csv_files if unset.
DATA_DIR=./csv_files

# Where to write/read the built knowledge-base (defaults to ./knowledgebase.pkl)
KNOWLEDGEBASE_PATH=./knowledgebase.pkl
```

**Do not** commit `.env`. Consider an `.env.example` with placeholders for teammates.

---

## Load the demo data from CSVs

The loader script
[`create_mytables_snowflake.py`](create_mytables_snowflake.py) reads files from `DATA_DIR`
(defaults to `./csv_files`) and writes Snowflake tables with appropriate types and
constraints. **Required filenames:**

- `olist_orders_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_order_reviews_dataset.csv`
- `olist_customers_dataset.csv`
- `olist_products_dataset.csv`
- `olist_sellers_dataset.csv`
- `product_category_name_translation.csv`

**Optional (not used by the app)**  
- `olist_geolocation_dataset.csv` — large (~58MB). The loader does not import it.

Run:
```bash
python create_mytables_snowflake.py
```
What it does:
- Ensures session settings (`TIMESTAMP_NTZ`, auto input formats).
- **Creates/resumes** the warehouse (if role permits), creates DB/Schema if needed, then switches context.
- Loads the eight tables using `pandas` + `pd_writer`, mapping to Snowflake types and normalizing timestamps to **TIMESTAMP_NTZ**.
- Declares **primary keys** and **foreign keys**:
  - `orders(order_id)`; `customer(customer_id)`; `products(product_id)`; `sellers(seller_id)`; `category_translation(product_category_name)`
  - `order_payments(order_id, payment_sequential)`; `order_items(order_id, order_item_id)`; `order_reviews(review_id)`
  - FKs across standard join paths (orders↔items/payments/reviews, orders→customer, items→products/sellers, products→category_translation)
- Runs quick **integrity checks** and prints orphan counts.
- Prints confirmations like:
  - `📁 Using DATA_DIR: <path>`
  - `🔎 Context — role=..., warehouse=..., db=..., schema=...`
  - `✅ Loaded: <table>`
  - `🔗 Keys & relationships declared...`
  - `🎉 All tables loaded ...`

**Notes**
- If you see “Constraint DDL skipped (permission?)”, your role may lack `ALTER TABLE`. The app still works without constraints.
- You can point to a different folder via `.env` → `DATA_DIR=/absolute/or/relative/path`.
- Column names are loaded **UPPERCASE** to match Snowflake’s unquoted semantics.

---

## How it works

### High‑level pipeline
1. **Knowledge‑base builder** (`build_knowledgebase.py`)
   - Samples each table (`SELECT * ORDER BY RANDOM() LIMIT 100`), reads `INFORMATION_SCHEMA.COLUMNS`, and asks the LLM to produce **concise, practical descriptions** for the table and each column (with 1–2 sample values).  
   - Output: `knowledgebase.pkl` mapping `{table_name: [table_desc, [[col, desc], ...]]}`.

2. **Routing & planning** (`agents.py`)
   - **Router** picks agent groups (`customer`, `orders`, `product`) relevant to the question.
   - **Subquestion generator** splits the question into minimal parts and assigns each to a single best table (names are canonicalized).
   - **Column selector** chooses only the columns needed (plus join keys) to answer/link subquestions.

3. **Filter extraction** (`agents.py` ➜ `chain_filter_extractor`)
   - Extracts WHERE‑like predicates from NL (e.g., `"last month"`, `"delivered"`, `"São Paulo"`).  
   - `utils.fuzzy_match_filters` resolves categorical values to real DB values and can map **city phrases to state abbreviations** (e.g., “São Paulo” → `SP`).

4. **SQL generation & validation**
   - **Generator** produces a single **Snowflake** query using known join keys and hints.
   - **Validator** fixes common mistakes (CTEs, join paths, location fields) and returns a ready‑to‑run statement.
   - Execution uses a safety **`LIMIT 2000`** (appended for the run only).

5. **Execution & visualization** (`sql_viz_workflow.py`)
   - Executes SQL. If Snowflake errors, a **SQL‑fixer** prompt repairs it (bounded retries).
   - A **BI expert** prompt recommends the best chart/table.
   - Plotly code is generated and then **silently auto‑fixed** until it runs. The Streamlit UI renders `fig`, `df_viz`, or a short `string_viz_result`.

### Dataset‑specific mapping hints (embedded)
- “Total sales / revenue / GMV” → `SUM(order_payments.payment_value)`
- Time trends → `orders.order_purchase_timestamp` (e.g., `DATE_TRUNC('month', ...)` with optional `TO_VARCHAR(..., 'YYYY-MM')`)
- Reviews → `order_reviews.review_score`
- Seller performance → `COUNT(DISTINCT order_items.order_id)` by `order_items.seller_id`
- English category → join `products.product_category_name` to `category_translation.product_category_name_english`

### Known join keys
- `orders.order_id` ↔ `order_items.order_id` ↔ `order_payments.order_id` ↔ `order_reviews.order_id`  
- `orders.customer_id` ↔ `customer.customer_id`  
- `order_items.product_id` ↔ `products.product_id`  
- `products.product_category_name` ↔ `category_translation.product_category_name`  
- `order_items.seller_id` ↔ `sellers.seller_id`

---

## Project layout

```
.
├─ csv_files/                           # ← default CSV source for loader
│  ├─ olist_orders_dataset.csv
│  ├─ olist_order_payments_dataset.csv
│  ├─ olist_order_items_dataset.csv
│  ├─ olist_order_reviews_dataset.csv
│  ├─ olist_customers_dataset.csv
│  ├─ olist_products_dataset.csv
│  ├─ olist_sellers_dataset.csv
│  └─ product_category_name_translation.csv
├─ create_mytables_snowflake.py         # CSV → Snowflake loader (creates DB/schema, tables, PK/FKs, checks)
├─ build_knowledgebase.py               # Introspects Snowflake + LLM descriptions → knowledgebase.pkl
├─ agents.py                            # Router, subquestions, column selection, filters, SQL gen + validator
├─ config.py                            # Azure OpenAI + Snowflake engine setup; .env loader; KB path
├─ nlq_to_viz_workflow.py               # Orchestrates end‑to‑end (router → columns → filters → SQL → viz)
├─ sql_viz_workflow.py                  # Validate/execute SQL; BI advice; generate/fix Plotly; run code
├─ streamlit_chat.py                    # Streamlit UI: ask, view SQL, viz, download results/code
├─ utils.py                             # Parsing helpers, code/SQL extraction, fuzzy filter matching
├─ knowledgebase.pkl                    # (generated) table/column descriptions
├─ requirements.txt                     # Dependencies
└─ .env                                 # (local, not committed) credentials and config
```

---

## Common tasks

### Rebuild the knowledge‑base after schema changes
```bash
python build_knowledgebase.py
```

### Reload the demo data (or load from a different folder)
```bash
# Option A: keep files in ./csv_files (default)
python create_mytables_snowflake.py

# Option B: point to another folder
# in .env: DATA_DIR=/path/to/my/csvs
python create_mytables_snowflake.py
```

### Tune LLM behavior
Edit `config.get_llm()`:
- `extra_body.reasoning_effort`: `low | medium | high`
- `extra_body.max_completion_tokens`: raise for longer tables/questions

### Limit/enable retries
- In `nlq_to_viz_workflow.run(question, max_retries=3)` (exposed in Streamlit “Advanced”)
- Applies to both **SQL repair** and **viz code repair** loops.

---

## Troubleshooting

**`CSV not found: <path>`**  
Place files in `./csv_files` or set `DATA_DIR` to a folder that contains all eight CSVs listed above.

**Permission errors creating warehouse/database/schema**  
The loader prints `ℹ️ ... (permission?)` and continues. Ensure your role has the right grants, or pre‑create the objects and re‑run.

**`knowledgebase.pkl not found ...`**  
Run `python build_knowledgebase.py`, or set `KNOWLEDGEBASE_PATH` correctly. The code also checks a couple of fallback locations.

**Missing Azure OpenAI or Snowflake env vars**  
Check names/values in `.env`. For Azure, the endpoint can be the resource base URL; the code normalizes full paths.

**Snowflake connection errors**  
- Verify network/VPN/allow‑lists and role/warehouse privileges.  
- Ensure `snowflake-connector-python` + `snowflake-sqlalchemy` are installed.  
- Account string format must match your org/region (e.g., `ORG-ACC` style).

**SQL generated but execution fails**  
The workflow will try to fix SQL automatically. If it still fails:
- Rephrase your question more narrowly, add date ranges, or start simpler.
- Copy the **Generated SQL** from the UI and tweak manually.

**Visualization not rendering**  
If SQL returns **zero rows**, the app will set `string_viz_result` to explain there’s no data to visualize.

---

## Design notes & extensibility

- **Safety:** Only `SELECT`/CTE queries are allowed; execution appends `LIMIT 2000`.
- **Robust parsing:** Utilities extract fenced code/SQL and parse list‑like JSON/py‑literals defensively.
- **Fuzzy filters:** Unicode‑aware matching, initials for Brazilian state abbreviations, and optional `rapidfuzz` acceleration.
- **Pluggable BI:** The BI “what to plot” step is concise; swap in your house style or chart lib if you like.
- **Adding new tables:** Re‑run the builder; if you add new logical **agents**, wire them in `AGENT_TABLES` and update the router prompt.

---

## Security & data governance

- Use a **read‑only** Snowflake role for the app, and a separate role with create/alter for the loader.
- Keep `.env` and `csv_files/` out of version control (`.gitignore`), or use a secret manager / data registry in deployment.
- Rotate Azure OpenAI/Snowflake secrets regularly and **immediately** if shared outside your org.
- The Streamlit app is for internal use; don’t expose it publicly without authentication and rate limits.

---

## License

Choose a license that fits your org; for open source, **MIT** is a common default.

```text
MIT License
Copyright (c) 2025 <Your Name/Org>
Permission is hereby granted, free of charge, to any person obtaining a copy...
```
