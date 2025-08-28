# =============================================================================
# nlq_to_viz_workflow.py — End-to-end pipeline to:
#  1) Route question to agent groups → concrete tables
#  2) Generate subquestions + select columns per table
#  3) Extract filters from NL
#  4) Generate SQL, validate/fix it
#  5) Execute and produce BI visualization code & result
# Returns a rich FinalState consumed by the Streamlit UI.
# =============================================================================

from __future__ import annotations
from typing import TypedDict, List, Any, Tuple
import ast, json
import pandas as pd

from agents import (
    agent_router,
    graph_final as customer_graph,
    AGENT_TABLES,
    chain_filter_extractor,
    chain_query_extractor,
    chain_query_validator,     # <--  import validator
)
from utils import parse_nested_list, fuzzy_match_filters, extract_sql
from sql_viz_workflow import run_workflow as run_sql_viz  # validates SQL, executes, BI, viz gen/validate


class FinalState(TypedDict):
    question: str
    sql: str
    # expose subquestions → table mapping for UI transparency
    subquestions: list
    columns_selected: list
    filters_raw: Any
    filters_matched: Any
    result_debug_sql: str
    error_msg_debug_sql: str
    result_debug_python_code_data_visualization: str
    error_msg_debug_python_code_data_visualization: str
    df: pd.DataFrame
    visualization_request: str
    python_code_data_visualization: str
    python_code_store_variables_dict: dict


def _pick_tables_for_question(question: str) -> List[str]:
    """Route to agent groups and expand to concrete tables.

    Falls back to 'orders' group if router output can't be parsed or is empty.
    Duplicates are removed while preserving order.
    """
    raw = agent_router(question)  # e.g., "['customer','orders']"
    try:
        agents = ast.literal_eval(raw)
        if not isinstance(agents, list):
            agents = []
    except Exception:
        agents = []
    tables: List[str] = []
    for a in agents:
        tables.extend(AGENT_TABLES.get(a, []))
    if not tables:
        tables = AGENT_TABLES.get("orders", [])
    seen = set()
    return [t for t in tables if not (t in seen or seen.add(t))]


def _subquestions_and_columns(question: str, tables: List[str]) -> Tuple[list, list]:
    """Invoke the LangGraph that produces BOTH subquestions and column selections."""
    st = customer_graph.invoke({"user_query": question, "table_lst": tables})
    subqs = st.get("table_extract", []) or []
    cols = st.get("column_extract", []) or []
    return subqs, cols


# --- de-dupe selected column rows across agents ---
def _dedupe_columns(selected: list[list]) -> list[list]:
    """Remove exact duplicate rows while preserving order (idempotent)."""
    seen, out = set(), []
    for row in selected or []:
        key = tuple(row) if isinstance(row, list) else tuple([row])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _filters(question: str, columns_selected: list):
    """Call filter extractor and then attempt to match categorical predicates to
    real DB values (using fuzzy matching), returning (raw, matched)."""
    raw = chain_filter_extractor.invoke({
        "query": question,
        "columns": str(columns_selected)
    }).strip()
    as_list = parse_nested_list(raw)
    if as_list:
        matched = fuzzy_match_filters(as_list)
        return raw, matched
    return raw, raw


def _generate_sql(question: str, columns_selected: list, filters_any) -> str:
    """Generate SQL from context; returns a single SELECT statement as text."""
    filters_str = json.dumps(filters_any) if isinstance(filters_any, (list, dict)) else str(filters_any)
    raw_sql = chain_query_extractor.invoke({
        "query": question,
        "columns": str(columns_selected),
        "filters": filters_str
    }).strip()
    # Be tolerant if the model wraps in ```sql fences; extract the actual SQL.
    return extract_sql(raw_sql)


# --- validator step before execution ---
def _validate_sql(question: str, columns_selected: list, filters_any, sql_text: str) -> str:
    """Run SQL through the validator/fixer before execution."""
    filters_str = json.dumps(filters_any) if isinstance(filters_any, (list, dict)) else str(filters_any)
    sql_valid = chain_query_validator.invoke({
        "query": question,
        "columns": str(columns_selected),
        "filters": filters_str,
        "sql_query": sql_text
    }).strip()
    # normalize any accidental markdown/code fencing from the validator
    sql_valid = extract_sql(sql_valid) if sql_valid else sql_text
    return sql_valid or sql_text


def run(question: str, *, max_retries: int = 3) -> FinalState:
    """End-to-end run producing SQL, DataFrame, and viz artifacts.

    max_retries governs how many times SQL and viz code are auto-fixed.
    """
    tables = _pick_tables_for_question(question)

    # Get both subquestions and column selections from the agents graph
    subquestions_raw, columns_selected_raw = _subquestions_and_columns(question, tables)

    # De-dupe before downstream usage
    columns_selected = _dedupe_columns(columns_selected_raw)

    # Filters over deduped columns
    filters_raw, filters_matched = _filters(question, columns_selected)

    # Generate then validate SQL (pre-execution)
    sql_raw = _generate_sql(question, columns_selected, filters_matched)
    sql = _validate_sql(question, columns_selected, filters_matched, sql_raw)

    # Execute + BI/Viz
    state = run_sql_viz(
        question=question,
        sql=sql,
        columns=str(columns_selected),
        filters=json.dumps(filters_matched) if not isinstance(filters_matched, str) else filters_matched,
        max_retries=max_retries
    )

    combined: FinalState = {
        "question": question,
        "sql": state["sql"],  # sql after validation/possible fixer inside run_sql_viz
        "subquestions": subquestions_raw,
        "columns_selected": columns_selected,
        "filters_raw": filters_raw,
        "filters_matched": filters_matched,
        "df": state.get("df", pd.DataFrame()),
        "visualization_request": state.get("visualization_request", ""),
        "python_code_data_visualization": state.get("python_code_data_visualization", ""),
        "python_code_store_variables_dict": state.get("python_code_store_variables_dict", {}),
        "result_debug_sql": state.get("result_debug_sql", ""),
        "error_msg_debug_sql": state.get("error_msg_debug_sql", ""),
        "result_debug_python_code_data_visualization": state.get("result_debug_python_code_data_visualization",""),
        "error_msg_debug_python_code_data_visualization": state.get("error_msg_debug_python_code_data_visualization",""),
    }
    return combined
