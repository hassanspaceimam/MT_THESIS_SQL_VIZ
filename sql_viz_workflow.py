"""
sql_viz_workflow.py

End-to-end “SQL → execute → BI hint → Plotly code → validate/run” pipeline.

This module wires a small LangGraph graph that:
1) Validates the generated SQL (and auto-fixes if the DB errors),
2) Executes the query and stores the resulting DataFrame,
3) Asks a BI expert prompt to recommend the right visualization,
4) Generates Plotly (or table/text) Python code,
5) Executes and, if needed, silently fixes the code until it runs.

Notes:
- The graph is intentionally linear. Each node reads/writes a shared `AgentState`.
- SQL safety: only SELECTs are allowed and large results are capped with a LIMIT.
- Plotly code must produce exactly one of: `fig`, `df_viz`, or `string_viz_result`.
"""

from __future__ import annotations
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy import text
import pandas as pd
import re
import traceback

from config import get_llm, get_engine
from utils import extract_code_block, extract_sql

# Initialize shared singletons once per process
engine = get_engine()
llm = get_llm()

# ------------------ Inlined prompts ------------------
# BI expert prompt – converts a question + df structure into a concise viz recommendation.
system_prompt_agent_bi_expert_node = """
Role:
You are a Business Intelligence (BI) expert specializing in data visualization. You will receive a user question, the SQL query used, and a Pandas DataFrame sample/structure. Your task is to determine the most effective way to present the data.

Guidelines:
- Analyze the user question and DataFrame to determine the best visualization method (chart or table).
- If the result contains a single value, suggest displaying it as a simple print statement with a label.
- Maintain the exact column names as they appear in the query.
- Be concise and explicit about which columns map to each axis or table.

Inputs
User Question:
{question}

SQL Query:
{query}

Data Structure & Types:
{df_structure}

Sample Data:
{df_sample}

Output Format
Provide a concise answer describing the best visualization method. Follow these guidelines:
- Specify whether a chart (e.g., bar chart, line chart, scatter plot, etc.) or table is more appropriate.
- Mention the columns to be used for each axis (if applicable).
- Use query column names for consistency.
"""

# Plotly code generator prompt – must output a *single* fenced python block.
system_prompt_agent_python_code_data_visualization_generator_node = """
You are an expert Python data visualization assistant specializing in Plotly and Python visualization. You will receive a Pandas DataFrame (as the variable df) and a detailed visualization request.

Your task is to analyze the DataFrame and the requested visualization and generate Python code using Plotly. Follow these rules:

CRITICAL RULES (read carefully):
- Use **only** the variable **df** for the dataset. **Do not** reference state or any other variables.
- **Do not** call fig.show(); the caller will render the figure.
- Do not load external files or connect to external services. No file I/O.
- The code must define **one and only one** of the following outputs:
  1) fig  – a Plotly figure object when a chart is appropriate, or
  2) df_viz – a pandas DataFrame to render as a table when a table is best, or
  3) string_viz_result – a short string when the result is a single value or message.
- If df is empty or has no rows, set string_viz_result explaining that there is no data to visualize.
- Keep axis labels and titles clear; use the query's column names as provided.
- Return **only** the code inside a fenced block: 
python ...


General expectations:
- Choose the most appropriate chart type based on the data and request.
- Label axes and titles properly.
- Prefer readable layouts (legends, marker visibility, number formatting when helpful).

Input DataFrame Summary:
Structure & Data Types:
{df_structure}
Sample Data:
{df_sample}

Request Visualization:
{visualization_request}

Output:
Analyze the DataFrame and request and provide the complete Python code to generate the requested visualization using Plotly. Remember: only df, no fig.show(), and produce exactly one of fig, df_viz, or string_viz_result.
"""

# Silent validator/fixer for the produced python visualization code.
system_prompt_agent_python_code_data_visualization_validator_node = """
**Role:** You are a Python expert in data visualization focused on *silently* fixing errors.

**Inputs:**
1. Python code:
python
[USER'S PLOTLY CODE]

2. Error:
[ERROR]

**Rules:**
- Output **only** the corrected Python code (no explanations, no markdown).
- The corrected code must:
  - Use **only** the variable **df** as the dataset (do not reference state or other variables).
  - **Not** call fig.show() (the caller renders the figure).
  - Produce exactly one of: fig (Plotly figure), df_viz (DataFrame), or string_viz_result (string).
  - Avoid file I/O and external network access.

**Examples Output:**
python
import plotly.graph_objects as go
fig = go.Figure(data=[go.Bar(y=[2, 3, 1])])

python
string_viz_result = "Number of cities: " + str(df['num_cities'].iloc[0])

python
df_viz = df

---
**Your turn:**
python
{python_code_data_visualization}

Error:
{error_msg_debug}
"""


# ------------------ Types & helpers ------------------
class AgentState(TypedDict):
    """Shared mutable state passed through the graph."""
    question: str
    sql: str
    columns: str                # Serialized columns that informed the SQL
    filters: str                # Serialized filters that informed the SQL
    num_retries_debug_sql: int
    max_num_retries_debug: int
    result_debug_sql: str       # "Pass" / "Not Pass" for SQL execution
    error_msg_debug_sql: str    # Last SQL error text (truncated)
    df: pd.DataFrame            # Result of the SQL query
    visualization_request: str  # BI expert recommendation (plain text)
    python_code_data_visualization: str  # Raw generated python code (string)
    num_retries_debug_python_code_data_visualization: int
    result_debug_python_code_data_visualization: str
    error_msg_debug_python_code_data_visualization: str
    python_code_store_variables_dict: dict  # Exec env after running the viz code


def _only_select(sql: str) -> None:
    """Reject non-SELECT/CTE statements up front (defense-in-depth)."""
    if not re.match(r"(?is)^\s*(select|with)\b", sql or ""):
        raise ValueError("Only SELECT/CTE statements are allowed.")

def _wrap_with_limit(sql: str, limit: int = 2000) -> str:
    """
    Add a LIMIT (or FETCH) without wrapping the SELECT.
    - If the query already ends with LIMIT/FETCH, leave it alone.
    - For CTEs or plain SELECTs, simply append LIMIT to the end.
    """
    s = (sql or "").strip().rstrip(";")
    if re.search(r"(?is)\blimit\s+\d+\s*$", s):
        return s
    if re.search(r"(?is)\bfetch\s+first\s+\d+\s+rows\s+only\s*$", s):
        return s
    return f"{s} LIMIT {limit}"

def _explain_safe(sql: str) -> None:
    """
    Try to EXPLAIN the query to surface obvious syntax/plan issues early.
    Failures are tolerated — EXPLAIN is best-effort and non-blocking.
    """
    try:
        with engine.begin() as conn:
            conn.execute(text("EXPLAIN USING TEXT " + sql))
    except Exception:
        pass  # Ignore explain errors; the real execution will surface issues.

# Prompt that attempts to fix a broken SQL using the known context (question/columns/filters).
_sql_fixer_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a precise Snowflake SQL query fixer.

STRICT OUTPUT:
- Return ONLY a single corrected Snowflake SQL SELECT statement. No prose, no markdown.

CONSTRAINTS:
- Use ONLY tables/columns implied by the provided "Relevant context" if present.
- Apply filters exactly when provided (do not invent or omit).
- If city/state is referenced for customers, it comes from customer.customer_city / customer.customer_state via orders.customer_id = customer.customer_id.
- Use Snowflake idioms: DATE_TRUNC('month', ...), TO_VARCHAR(ts, 'YYYY-MM'), LISTAGG, DATEDIFF('day', start, end), DATEADD(unit, n, ts).
- Avoid reserved words as aliases. Balance parentheses. If using CTEs, ensure full WITH clauses.
"""),
    ("human", """
User question:
{question}

Relevant context (optional; may be empty):
Columns:
{columns}

Filters:
{filters}

Current SQL to fix:
{sql}

Database error message:
{error}
""")
])
_sql_fixer_chain = _sql_fixer_prompt | llm | StrOutputParser()

# ------------------ Graph nodes ------------------
def sql_validate_and_execute_node(state: AgentState) -> AgentState:
    """
    Attempt to run the provided SQL:
      - Enforce SELECT-only
      - Add a LIMIT if missing (append; do not wrap)
      - Optionally EXPLAIN
      - Execute and capture resulting DataFrame
    On DB error:
      - Prompt the SQL fixer with context and retry, up to max retries.
    """
    sql_in = (state.get("sql") or "").strip()
    if not sql_in:
        raise ValueError("No SQL provided to the validator. Pass sql=... or generate one before this step.")

    for attempt in range(state["num_retries_debug_sql"], state["max_num_retries_debug"] + 1):
        try:
            _only_select(sql_in)
            limited_sql = _wrap_with_limit(sql_in, limit=2000)
            _explain_safe(limited_sql)

            # Use a fresh engine connection for thread/process safety.
            df = pd.read_sql(text(limited_sql), con=get_engine())
            state["df"] = df
            state["result_debug_sql"] = "Pass"
            state["error_msg_debug_sql"] = ""
            state["sql"] = sql_in  # preserve original (unwrapped) SQL
            return state

        except Exception as e:
            # Capture error info and attempt an automatic model-based fix.
            state["num_retries_debug_sql"] = attempt + 1
            state["result_debug_sql"] = "Not Pass"
            tb = traceback.format_exc(limit=1)
            err_short = (str(e) + " | " + tb)[:600]
            state["error_msg_debug_sql"] = err_short

            # Ask LLM to fix the SQL in place, then loop again.
            sql_in = extract_sql(_sql_fixer_chain.invoke({
                "question": state["question"],
                "columns": state.get("columns", ""),
                "filters": state.get("filters", ""),
                "sql": sql_in,
                "error": err_short
            }).strip())
    return state  # Return last failure if all retries exhausted.

def bi_expert_node(state: AgentState) -> AgentState:
    """
    Produce a concise “what to plot” recommendation based on the DataFrame and question.
    This is deliberately plain text so the next node can turn it into concrete code.
    """
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt_agent_bi_expert_node)])
    chain = prompt | llm | StrOutputParser()
    df = state.get("df", pd.DataFrame())
    response = chain.invoke({
        "question": state["question"],
        "query": state["sql"],
        "df_structure": df.dtypes if not df.empty else "EMPTY",
        "df_sample": df.head(5) if not df.empty else "EMPTY"
    }).strip()
    state["visualization_request"] = response
    return state

def viz_code_generator_node(state: AgentState) -> AgentState:
    """
    Turn the BI recommendation + df summary into Plotly (or table/text) Python code.
    The subsequent validator will actually run (and fix) that code if needed.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_agent_python_code_data_visualization_generator_node)
    ])
    chain = prompt | llm | StrOutputParser()
    df = state.get("df", pd.DataFrame())
    response = chain.invoke({
        "visualization_request": state["visualization_request"],
        "df_structure": df.dtypes if not df.empty else "EMPTY",
        "df_sample": df.head(5) if not df.empty else "EMPTY"
    })
    # Extract fenced python only; keep raw code string for execution later.
    state["python_code_data_visualization"] = extract_code_block(response, "python").strip()
    return state

def viz_code_validator_node(state: AgentState) -> AgentState:
    """
    Execute the generated code in a controlled namespace:
      - Inject only df/pd/px/go/state (state carries df as well for legacy code).
      - Strip `fig.show()` calls if present.
      - If execution fails, call the silent fixer and retry up to max retries.
    On success, store the execution globals so the Streamlit app can render.
    """
    code = state.get("python_code_data_visualization", "").strip()
    if not code:
        state["result_debug_python_code_data_visualization"] = "Not Pass"
        state["error_msg_debug_python_code_data_visualization"] = "Empty python visualization code."
        return state

    for attempt in range(state["num_retries_debug_python_code_data_visualization"], state["max_num_retries_debug"] + 1):
        try:
            import pandas as pd
            import plotly.express as px
            import plotly.graph_objects as go
            import re

            df = state.get("df")
            if df is None:
                df = pd.DataFrame()

            # Make code robust to accidental references to `state.get('df')`.
            code_to_run = re.sub(r"state\.get\(\s*['\"]df['\"]\s*\)", "df", code)
            # Ensure non-interactive execution environment.
            code_to_run = re.sub(r"fig\.show\(\)\s*;?", "", code_to_run)

            exec_globals: Dict[str, Any] = {"df": df, "pd": pd, "px": px, "go": go, "state": {"df": df}}
            exec(code_to_run, exec_globals)

            # Persist outputs for UI consumption.
            state["python_code_store_variables_dict"] = exec_globals
            state["result_debug_python_code_data_visualization"] = "Pass"
            state["error_msg_debug_python_code_data_visualization"] = ""
            state["python_code_data_visualization"] = code_to_run
            return state

        except Exception as e:
            # Capture error and round-trip through the fixer prompt.
            import traceback
            state["num_retries_debug_python_code_data_visualization"] = attempt + 1
            state["result_debug_python_code_data_visualization"] = "Not Pass"
            err_short = (str(e) + " | " + traceback.format_exc(limit=1))[:800]
            state["error_msg_debug_python_code_data_visualization"] = err_short

            fix_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt_agent_python_code_data_visualization_validator_node),
                ("human", "python\n{python_code_data_visualization}\n\nError:\n{error_msg_debug}")
            ])
            chain = fix_prompt | llm | StrOutputParser()
            fixed = chain.invoke({
                "python_code_data_visualization": code,
                "error_msg_debug": err_short
            })
            from utils import extract_code_block as _extract
            code = _extract(fixed, "python").strip()
    return state  # Return last failure if all retries exhausted.

# ------------------ Graph wiring ------------------
graph = StateGraph(AgentState)
graph.add_node("sql_validate_and_execute", sql_validate_and_execute_node)
graph.add_node("bi_expert", bi_expert_node)
graph.add_node("viz_code_generator", viz_code_generator_node)
graph.add_node("viz_code_validator", viz_code_validator_node)

graph.add_edge(START, "sql_validate_and_execute")
graph.add_edge("sql_validate_and_execute", "bi_expert")
graph.add_edge("bi_expert", "viz_code_generator")
graph.add_edge("viz_code_generator", "viz_code_validator")
graph.add_edge("viz_code_validator", END)

app = graph.compile()

def run_workflow(
    question: str,
    sql: str,
    *,
    columns: str = "",
    filters: str = "",
    max_retries: int = 3
) -> AgentState:
    """
    Entry point for callers. Builds the initial state and runs the compiled graph.

    Args:
        question: Original NLQ used for traceability and SQL repair.
        sql: The SQL to validate/execute (already generated & optionally pre-validated).
        columns: Text description of “relevant columns” that informed the SQL.
        filters: Text/JSON of applied filters for traceability and SQL fixes.
        max_retries: Budget for (a) SQL fix attempts and (b) viz code fix attempts.

    Returns:
        Final AgentState including df, BI recommendation, viz code, and any errors.
    """
    initial: AgentState = {
        "question": question,
        "sql": sql,
        "columns": columns or "",
        "filters": filters or "",
        "num_retries_debug_sql": 0,
        "max_num_retries_debug": int(max_retries),
        "result_debug_sql": "",
        "error_msg_debug_sql": "",
        "df": pd.DataFrame(),
        "visualization_request": "",
        "python_code_data_visualization": "",
        "num_retries_debug_python_code_data_visualization": 0,
        "result_debug_python_code_data_visualization": "",
        "error_msg_debug_python_code_data_visualization": "",
        "python_code_store_variables_dict": {},
    }
    return app.invoke(initial)
