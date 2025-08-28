# agents.py

# =============================================================================
# agents.py — Orchestrates routing, subquestion/column selection, filter parsing,
# SQL generation, and validation for a Text-to-SQL system built on LangChain
# and LangGraph. A single shared LLM client is used across chains.
# =============================================================================

from __future__ import annotations
import os
import pickle
import re
from typing import TypedDict, Annotated, List
from operator import add

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from langgraph.graph import StateGraph, START, END

from config import get_llm, get_knowledgebase_path
from utils import parse_nested_list, normalize_subquestions

# Single shared LLM client (cached by config.get_llm)
llm = get_llm()

# ===========================
# Router 
# ===========================
# Chooses which high-level agent groups (customer / orders / product) are
# relevant for a given natural language question. Output MUST be a Python-like
# list literal of agent names (strings). Downstream code parses it with
# ast.literal_eval.
_router_template = ChatPromptTemplate.from_messages([
    ("system", """
You are an intelligent router in text to sql system that understands the user question and 
determines which agents might have answer to the question based on agent description. Multiple agents might answer a given user question. OUTPUT SHOULD BE IN FORM OF LIST OF strings.
Dont give any explanation or any other verbose in the output.
"""),
    ("human", '''
Below are descriptions of different agents.
customer agent : It contains all the details about customer and seller locations and their unique identifiers
orders agent : It contains details about all the orders like product identifier, order identifier, products in an order, no. of items of a product in order, price of order, frieght value, order time, delivery status and its time, payment etc.
product agent : It contains details about product like product identifier, product category, description, dimensions of product

STEP BY STEP TABLE SELECTION PROCESS:
- Split the question into different subquestions.
- For each subquestion, very carefully go through each and every AGENT description, think which agent might have answer to this subquestion.
- At the end collect all the agents that you thought can answer the whole question in form of list of strings
- For a give question, if customer and orders agents can answer question, give output like below without any verbose.
['customer', 'orders']
- If only customer can answer a question , give output like below with one table in list
['customer']
- For a give question, if customer and orders and product agent can answer question, give output like below without any verbose.
['customer', 'orders', 'product']
     
User question:
{question}
''')
])
_router_chain = (RunnableMap({"question": lambda x: x["question"]}) | _router_template | llm | StrOutputParser())


def agent_router(question: str) -> str:
    """Invoke router chain and strip newlines (router promises a list literal).

    Returns: e.g., "['customer','orders']"
    """
    return _router_chain.invoke({"question": question}).replace("\n", "")


# =========================================
# Subquestion + Column selection 
# =========================================
# LLM creates minimal subquestions and assigns each to a single best table.
# Note: downstream logic can link across tables later via known join keys.
template_subquestion = ChatPromptTemplate.from_messages([
    ("system", """
You are an intelligent subquestion generator that creates subquestions based on human instructions and the provided CONTEXT. You operate as part of a Text-to-SQL agent.

STRICT OUTPUT CONTRACT (read carefully):
- Return ONLY a JSON array (no backticks, no markdown, no extra text).
- Each element MUST be a 2-item array: ["<subquestion>", "<table_name>"].
- DO NOT group multiple subquestions into one element. If multiple subquestions map to the same table, emit multiple 2-item elements that reuse that table.
- Use double quotes for all strings.
- If no valid subquestions exist, return [] (an empty JSON array).

LINKING MINDSET:
You may choose a table even if it cannot independently answer a subquestion, as long as it serves as a link to another table (e.g., order_id links orders ↔ order_items ↔ order_payments). Aim to select the single best table for each subquestion, while keeping potential links in mind.

DATASET-SPECIFIC MAPPING HINTS (very important):
- Seller performance (e.g., “Which seller received the most orders?”) → use the order_items table (it has seller_id) and count DISTINCT order_id per seller_id.
- Time trends for orders/sales → use orders.order_purchase_timestamp (e.g., monthly with TO_VARCHAR(orders.order_purchase_timestamp, 'YYYY-MM')).
- Total sales / revenue → SUM(order_payments.payment_value), joined to orders via order_id.
- Reviews → order_reviews.review_score linked by order_id.
- English category → category_translation.product_category_name_english joined to products.product_category_name.
"""),
    ("human", '''
CONTEXT:
This dataset is from Olist, the largest department store on Brazilian marketplaces. 
When a customer purchases a product from Olist (via a specific seller and location), the seller is notified to fulfill the order. 
After the customer receives the product or the estimated delivery date has passed, they receive an email survey to rate their purchase experience and leave comments.

You are given:
- A user question
- A list of table names with descriptions

Instructions:
Think like a Text-to-SQL agent. When selecting tables, carefully determine whether multiple tables need to be joined, and choose only those necessary to answer the user’s question. A table may not directly answer a subquestion but could serve as a link to another table selected by a different agent. Keep this in mind when making your selection. If one table contains all the required information, do not include others.

Your task:
1. Break the user question into minimal, precise subquestions that cover distinct parts of the requested information.
2. For each subquestion, identify the single best table whose description clearly shows it contains the needed information
3. Exclude any subquestion that cannot be answered using the provided tables.
4. Include only subquestions that directly contribute to answering the main user question.
5. If multiple tables could answer a subquestion, select the one most appropriate based on the descriptions.
6. Be highly specific, avoiding redundancy and irrelevant details (e.g., if the number of orders is requested, use order IDs without adding extra details).

Output format (STRICT):
- Return ONLY valid JSON (no code fences, no prose).
- A JSON array of 2-item arrays: [["subquestion1","table1"], ["subquestion2","table2"], ...]
- DO NOT group multiple subquestions into a single element.
- If multiple subquestions map to the same table, repeat that table in separate elements.
- If no valid subquestions: []

Table List:
{tables}

User question:
{user_query}
''')
])

# Runnable that feeds the prompt, runs the LLM, and parses as a string
chain_subquestion = (
    RunnableMap({
        "tables": lambda x: x["tables"],
        "user_query": lambda x: x["user_query"]
    })
    | template_subquestion
    | llm
    | StrOutputParser()
)

# Column selection for each subquestion. Picks only the columns needed for
# correct SQL generation and linking; emphasizes identifiers and location fields.
template_column = ChatPromptTemplate.from_messages([
    ("system", """
You are an intelligent data column selector that chooses the most relevant columns from a list of available column descriptions to help answer a subquestion ONLY.
Your selections will be used by a SQL generation agent, so choose **only those columns** that will help write the correct SQL query for a subquestion based on main question.

Act like you're preparing the exact inputs required to build the SQL logic. Also, look at main user question before selecting columns.
BUT main PRIORITY IS TO SELECT columns for subquestion.

HOW TO THINK STEP BY STEP:
- For each subquestion mentioned in subquestion below, think if <column1> in Column list might help in answering the question based on column description below. If no, check if this column can be used to answer any part of main question below.
- There can be critical dependencies between columns (e.g., totals need identifiers like order_id; multi-row facts like installments/items must be combined).
- Include supporting columns that help define or group the main entity (e.g., order_id if the question asks for order-level info).
- Only after processing the subquestion completely, look at main question to see if it adds any more relevant columns.

RULES:
1. ALWAYS include any unique identifiers related to the entity being queried (e.g., order_id, product_id, customer_id).
2. NEVER select the customer_unique_id column — it must always be ignored.
3. When a value depends on multiple rows/parts, include all columns required to fully calculate or group that metric.
4. Output must be a list of pairs: [["<column name>", "<description and how it is used>"], ...] (each inner list length == 2).

LOCATION HINT (mandatory):
- If the question mentions *city*, *state*, or *location* for customers:
  - Select customer.customer_city and/or customer.customer_state.
  - Also include orders.customer_id so the query can join orders → customer.
- If the question mentions seller location:
  - Select sellers.seller_city and/or sellers.seller_state and include order_items.seller_id to join order_items → sellers.

Hints:
- For seller-level counts (e.g., “Which seller received the most orders?”), select order_items.seller_id and order_items.order_id (COUNT DISTINCT order_id per seller_id).
- For total sales/revenue, ensure order_payments.payment_value is selected; for time trends also include orders.order_purchase_timestamp (e.g., TO_VARCHAR(timestamp, 'YYYY-MM') for month label).
"""),
    ("human", '''
Column list:
{columns}
     
subquestion:
{query}
     
Main question:
{main_question}
''')
])

chain_column_extractor = (
    RunnableMap({
        "columns": lambda x: x["columns"],
        "query": lambda x: x["query"],
        "main_question": lambda x: x["main_question"]
    })
    | template_column
    | llm
    | StrOutputParser()
)

# ----------------------------
# Filter / SQL / Validation
# ----------------------------
# Extract WHERE-like filters implied by the question, returned as strict JSON.
template_filter_check = ChatPromptTemplate.from_messages([
    ("system", """
You help a text-to-SQL agent decide WHAT filters are implied by a user's question.
Return a STRICT JSON array:
- If no filter: ["no"]
- If filters exist: ["yes", ["<table>", "<column>", "<predicate>"], ...]
Where <predicate> is one of:
  - simple equality for categorical columns, e.g., "credit_card", "SP", "delivered"
  - numerical or date conditions, e.g., ">= 5", "< 100", "between 2017-01-01 and 2017-01-31", "after 2018-10-01", "before 2018-10-01"
Rules:
- Include only filters that truly narrow the dataset (e.g., city/state, payment_type, status, date ranges, numeric thresholds).
- If the question gives a natural language date like "last month", translate it into a relative predicate string, e.g., "last month" (the downstream agent will resolve it).
- Prefer equality for categorical values; use ranges for dates and numbers.
- When the question mentions **customer location** (state or city), prefer filters on **customer.customer_state** when the place is a Brazilian state name (e.g., "São Paulo" → state), otherwise use **customer.customer_city**. Remember the join path to customers is **orders.customer_id = customer.customer_id** when needed.

IMPORTANT OUTPUT FORMATTING (do not skip):
- Return the predicate exactly as it appears in the user's natural language. Do NOT abbreviate or translate.
  For Brazilian states, if the user says "São Paulo", output "São Paulo" (not "SP").
  The downstream fuzzy-matching stage will normalize values to DB abbreviations if needed.

Return ONLY the JSON array, no prose.
"""),
    ("human", '''
User question:
{query}

Available tables and columns (with sample values):
{columns}
''')
])

chain_filter_extractor = (
    RunnableMap({
        "columns": lambda x: x["columns"],
        "query": lambda x: x["query"]
    })
    | template_filter_check
    | llm
    | StrOutputParser()
)

# ----------- SQL GENERATOR  (Snowflake) -----------
template_sql_query = ChatPromptTemplate.from_messages([
    ("system", """
You are an intelligent Snowflake SQL query generator.

STRICT OUTPUT CONTRACT
- Return ONLY a single Snowflake SQL query as plain text. No prose, no explanations, no markdown/code fences.
- The output MUST be a single syntactically valid statement. If you use a CTE, include the full WITH <name> AS (...) and close all parentheses.
- Do NOT include any leading commentary such as "Assuming...".
- Output must be ready to run as-is.

SCHEMA COMPLIANCE
- Prefer the tables and columns listed under "Relevant tables and columns" below.
- If a standard join key or column is obviously required to connect the listed tables (see Known Join Keys) but is missing from the list, you may include it to produce a correct query.
- Do not introduce unrelated tables or columns not needed to answer the question.

KNOWN JOIN KEYS (helpful guidance, use only when present in the provided columns):
- orders.order_id ↔ order_items.order_id ↔ order_payments.order_id ↔ order_reviews.order_id
- orders.customer_id ↔ customer.customer_id
- order_items.product_id ↔ products.product_id
- products.product_category_name ↔ category_translation.product_category_name
- order_items.seller_id ↔ sellers.seller_id

SCHEMA MAPPING HINTS (Snowflake):
- Customer city/state come from customer.customer_city / customer.customer_state via orders.customer_id = customer.customer_id. Do NOT use non-existent columns like orders.city or orders.state.
- Monthly bucket: DATE_TRUNC('month', orders.order_purchase_timestamp); optional label: TO_VARCHAR(orders.order_purchase_timestamp, 'YYYY-MM').
- String aggregation: LISTAGG(expr, ',') [WITHIN GROUP (ORDER BY ...)].
- Date math: DATEADD(<unit>, <n>, <timestamp>), DATEDIFF(<unit>, start, end).

COLUMN USAGE POLICY
- Use only the columns you actually need to answer the question clearly.
- Always include required identifiers and join keys so the query is correct and traceable.
- Do not feel obligated to include every listed column in SELECT if it is not required.

FILTERS
- Apply exactly the predicates given in "Applicable filters" below. Treat them literally (e.g., "between 2017-01-01 and 2017-01-31", ">= 5", "delivered").
- Translate relative dates like "today", "yesterday", "last 7 days", "last month", and "this year"
  into Snowflake ranges using CURRENT_DATE and DATE_TRUNC. Examples:
  - last 7 days → orders.order_purchase_timestamp >= DATEADD(day, -7, CURRENT_DATE)
  - last month → orders.order_purchase_timestamp >= DATE_TRUNC('month', DATEADD(month, -1, CURRENT_DATE))
                 AND orders.order_purchase_timestamp < DATE_TRUNC('month', CURRENT_DATE)
  - this year  → orders.order_purchase_timestamp >= DATE_TRUNC('year', CURRENT_DATE)
                 AND orders.order_purchase_timestamp < DATE_TRUNC('year', DATEADD(year, 1, CURRENT_DATE))

AGGREGATION & DISTINCT
- When counting logical entities that can repeat across rows (e.g., multiple items per order), use COUNT(DISTINCT <entity_id>) as appropriate to match the user’s intent.
- For seller-level order counts from item-level data, compute COUNT(DISTINCT order_items.order_id) grouped by order_items.seller_id; order by that count DESC and limit as needed.

STYLE & SAFETY
- Use meaningful, short aliases (never SQL reserved words like 'or', 'and', 'as').
- Prefer CTEs for readability if the query is long/complex, but ensure the CTE is fully defined and referenced.
- For “average delivery time”, default to DAYS:
  use DATEDIFF('day', orders.order_purchase_timestamp, orders.order_delivered_customer_date)
  and exclude NULL timestamps unless the user explicitly asks for hours.
- Do NOT use backticks. Use unquoted identifiers (case-insensitive) or double quotes only when necessary.

Return ONLY the final SQL statement.
"""),
    ("human", '''
User question:
{query}

Relevant tables and columns:
{columns}

Applicable filters:
{filters}
''')
])

chain_query_extractor = (
    RunnableMap({
        "columns": lambda x: x["columns"],
        "query": lambda x: x["query"],
        "filters": lambda x: x["filters"]
    })
    | template_sql_query
    | llm
    | StrOutputParser()
)

# ----------- VALIDATOR (Snowflake) -----------
template_validation = ChatPromptTemplate.from_messages([
    ("system", """
You are a highly capable and precise Snowflake SQL query validator and fixer.

STRICT OUTPUT CONTRACT
- Return ONLY a single corrected Snowflake SQL query as plain text. No prose, no explanations, no markdown/code fences.
- If the provided query is fully correct, return it UNCHANGED (still SQL only).
- If a CTE alias is referenced (e.g., FROM base), ensure the CTE is fully declared with WITH <name> AS (...) and parentheses balanced. Fix dangling parentheses or undefined aliases.
- If the provided query has issues, return a revised SQL query that is syntactically valid and logically correct.

SCHEMA & INPUT COMPLIANCE
- Prefer the tables and columns listed in "Relevant Tables and Columns".
- If the query uses a **standard join key or column** that is **obviously required** to connect the provided tables (see Known Join Keys) but was not listed, KEEP it (do not remove), as long as it only serves to correctly join the listed tables.
- Do NOT introduce unrelated tables/columns outside the provided context.

KNOWN JOIN KEYS (for reference during validation):
- orders.order_id ↔ order_items.order_id ↔ order_payments.order_id ↔ order_reviews.order_id
- orders.customer_id ↔ customer.customer_id
- order_items.product_id ↔ products.product_id
- products.product_category_name ↔ category_translation.product_category_name
- order_items.seller_id ↔ sellers.seller_id

- If a location field is referenced on the wrong table (e.g., orders.city), replace it with customer.customer_city / customer.customer_state AND add the necessary join orders.customer_id = customer.customer_id, provided these columns are available in the inputs or are standard join keys needed to connect the provided tables.
- Apply "Applicable filters" exactly as given (e.g., "between 2017-01-01 and 2017-01-31", ">= 5", "delivered"). Do not add or remove filters.

COLUMN & ALIAS POLICY
- Ensure required identifiers (join keys), grouping keys, and metric-related columns are present and used correctly.
- Keep SELECT minimal and relevant; you do not need to include every selected/available column, only what’s necessary for correctness.
- Do not use SQL reserved words (e.g., 'or', 'and', 'as') as table or column aliases.

AGGREGATION & DISTINCT
- When counting logical entities that may repeat across rows (e.g., multiple items per order in order_items), prefer COUNT(DISTINCT <entity_id>) to match the intended entity-level count.
- For seller-level order counts derived from order_items, the query must use COUNT(DISTINCT order_id) grouped by seller_id; if not, rewrite it accordingly.

ROBUSTNESS
- For grouped results or counts with filters, use subqueries/CTEs where needed to avoid conflicts between GROUP BY, HAVING, and aggregates.
- For “average delivery time”, default to DATEDIFF('day', order_purchase_timestamp, order_delivered_customer_date) with NULL filtering unless the user asked for hours.

Return ONLY the final SQL statement.
"""),
    ("human", '''
**User Question:**
{query}

**Relevant Tables and Columns:**
{columns}

**Applicable Filters:**
{filters}

**SQL Query to Validate:**
{sql_query}
''')
])

chain_query_validator = (
    RunnableMap({
        "columns": lambda x: x["columns"],
        "query": lambda x: x["query"],
        "filters": lambda x: x["filters"],
        "sql_query": lambda x: x["sql_query"],
    })
    | template_validation
    | llm
    | StrOutputParser()
)

# ========================================
# Customer Agent graph 
# ========================================
# Loads the knowledgebase (per-table descriptions + columns) produced by
# build_knowledgebase.py. Includes robust fallbacks to locate the pickle.
_KB_PATH = get_knowledgebase_path()
try:
    with open(_KB_PATH, "rb") as f:
        loaded_dict = pickle.load(f)
except FileNotFoundError:
    # Fallback 1: local CWD; Fallback 2: alongside this file
    candidates = [
        os.path.join(os.getcwd(), "knowledgebase.pkl"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledgebase.pkl"),
    ]
    for cand in candidates:
        try:
            with open(cand, "rb") as f:
                loaded_dict = pickle.load(f)
                break
        except FileNotFoundError:
            continue
    else:
        # Surface a clear error including attempted paths
        raise FileNotFoundError(
            f"knowledgebase.pkl not found at { _KB_PATH } or fallbacks { candidates }"
        )

# --- Canonicalization of table names (prevents KeyError on "customers" vs "customer") ---
# Map common variants to the exact KB keys.
_CANON_MAP = {
    "customer": "customer",
    "customers": "customer",
    "seller": "sellers",
    "sellers": "sellers",
    "order": "orders",
    "orders": "orders",
    "order_item": "order_items",
    "order_items": "order_items",
    "orderitems": "order_items",
    "order_payment": "order_payments",
    "order_payments": "order_payments",
    "payment": "order_payments",
    "payments": "order_payments",
    "order_review": "order_reviews",
    "order_reviews": "order_reviews",
    "review": "order_reviews",
    "reviews": "order_reviews",
    "product": "products",
    "products": "products",
    "category_translation": "category_translation",
    "category_translations": "category_translation",
    "categorytranslation": "category_translation",
    "categorytranslations": "category_translation",
}

def _canonicalize_table_name(name: str) -> str | None:
    """
    Normalize an LLM-proposed table name to a KB key.
    Strategy: lowercase + underscores → direct map → exact KB key →
              singular/plural toggles → fuzzy (difflib/rapidfuzz).
    """
    if not name:
        return None
    s = str(name).strip().lower().replace(" ", "_")
    # direct dictionary map
    if s in _CANON_MAP:
        return _CANON_MAP[s]
    # exact KB key as given
    if name in loaded_dict:
        return name
    if s in loaded_dict:
        return s
    # simple singular/plural toggles
    if s.endswith("s") and s[:-1] in loaded_dict:
        return s[:-1]
    if (s + "s") in loaded_dict:
        return s + "s"
    # fuzzy match (rapidfuzz if available, else difflib)
    kb_keys = list(loaded_dict.keys())
    try:
        from rapidfuzz import process, fuzz
        got = process.extractOne(s, kb_keys, scorer=fuzz.token_set_ratio)
        if got and got[1] >= 80:
            return got[0]
    except Exception:
        pass
    try:
        import difflib
        matches = difflib.get_close_matches(s, kb_keys, n=1, cutoff=0.8)
        if matches:
            return matches[0]
    except Exception:
        pass
    return None

def _normalize_subqs_to_known_tables(subqs: List[List[str]]) -> List[List[str]]:
    """
    Replace LLM table name variants with canonical KB keys and drop unknowns.
    """
    out: List[List[str]] = []
    for e in subqs:
        if not isinstance(e, list) or len(e) < 2:
            continue
        subq, tbl = e[0], e[1]
        canon = _canonicalize_table_name(tbl)
        if canon:
            out.append([subq, canon])
    return out

# Router groups -> concrete tables to include for subquestion/column selection
AGENT_TABLES = {
    "customer": ["customer", "sellers"],
    "orders": ["order_items", "order_payments", "order_reviews", "orders"],
    "product": ["products", "category_translation"],
}


class OverallState(TypedDict):
    """State passed through the LangGraph for subquestion & column extraction."""
    user_query: str
    table_lst: List[str]
    # NOTE: these inner lists are like [subquestion, table_name]
    table_extract: Annotated[list[str], add]
    # NOTE: downstream contains rows like ["name of table:<t>", "<col>", "<why>"]
    column_extract: Annotated[list[str], add]


def _agent_subquestion(q: str, v: str) -> str:
    """Run the subquestion chain over the provided table description dict string.

    q: user query
    v: stringified mapping of table -> [description, columns]
    """
    return chain_subquestion.invoke({"tables": v, "user_query": q}).replace("\n", "")


def _solve_subquestion(q: str, lst: List[str]) -> str:
    """Build a minimal table -> description dict for the selected list of tables
    and ask the LLM to split the question into subquestions and assign tables."""
    pairs = {t: loaded_dict[t][0] for t in lst}
    return _agent_subquestion(q, str(pairs))


def _sq_node(state: OverallState):
    """LangGraph node: compute subquestions mapped to tables and normalize."""
    q = state["user_query"]
    lst = state["table_lst"]
    raw = _solve_subquestion(q, lst) or "[]"
    parsed = parse_nested_list(raw)
    norm = normalize_subquestions(parsed)
    norm = _normalize_subqs_to_known_tables(norm)  # <-- canonicalize LLM table names
    return {"table_extract": norm}


def _agent_column_selection(mq: str, q: str, c: str) -> str:
    """Run column selection chain and extract the first top-level JSON array.
    Uses a regex to capture [[...],[...],...] blocks if extra text slips in."""
    resp = chain_column_extractor.invoke({
        "columns": c, "query": q, "main_question": mq
    }).replace("\n", "")
    m = re.search(r"\[\s*\[.*?\]\s*(,\s*\[.*?\]\s*)*\]", resp, re.DOTALL)
    return m.group(0) if m else "[]"


def _solve_column_selection(main_q: str, list_sub: list[list[str]]) -> list[list[str]]:
    """For each [subquestion, table] select the most relevant columns using the
    knowledgebase for that table, then assemble rows of the form:
    ["name of table:<table>", "<column>", "<reason>"]
    """
    final_col: list[list[str]] = []
    for tab in list_sub:
        if not tab:
            continue
        table_name = tab[-1]
        canon = _canonicalize_table_name(table_name)
        if not canon or canon not in loaded_dict:
            # Skip unknown/unmapped tables instead of raising KeyError
            continue
        question = " | ".join(tab[:-1]) or ""
        columns = loaded_dict[canon][1]
        out_column = _agent_column_selection(main_q, question, str(columns))
        trans_col = parse_nested_list(out_column)
        for col_selec in trans_col:
            if not isinstance(col_selec, list) or len(col_selec) < 2:
                continue
            final_col.append([f"name of table:{canon}", *col_selec])
    return final_col


def _column_node(state: OverallState):
    """LangGraph node: run column selection over subquestions."""
    subq = state["table_extract"]
    mq = state["user_query"]
    o = _solve_column_selection(mq, subq)
    return {"column_extract": o}


# --- Build a tiny 2-node graph: subquestions -> column selection ---
_builder = StateGraph(OverallState)
_builder.add_node("subquestion", _sq_node)
_builder.add_node("column_e", _column_node)
_builder.add_edge(START, "subquestion")
_builder.add_edge("subquestion", "column_e")
_builder.add_edge("column_e", END)
# Compiled callable graph
graph_final = _builder.compile()
