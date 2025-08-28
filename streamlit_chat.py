"""
streamlit_chat.py

Minimal Streamlit front-end for the NLQ → SQL → Viz pipeline.
- Captures a natural language question.
- Runs the full workflow (SQL gen/validation/exec + BI + viz code).
- Renders the result figure/table/text and exposes convenient downloads.
"""

# streamlit_chat.py
import json
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from nlq_to_viz_workflow import run as run_full

# Basic page metadata/layout.
st.set_page_config(page_title="SQL/BI Agent", layout="wide")

# --- Make code blocks (e.g., Generated SQL) much taller & wrap long lines ---
st.markdown(
    """
<style>
/* Wrapper of st.code() */
div[data-testid="stCodeBlock"]{
    min-height: 45vh !important;      /* tall by default */
    max-height: 80vh !important;      /* allow very tall before scrolling */
    overflow: auto !important;         /* scroll when longer */
}
/* The <pre> inside the code block */
div[data-testid="stCodeBlock"] pre,
div.stCodeBlock pre {
    white-space: pre-wrap !important;  /* wrap long lines */
    word-break: break-word !important; /* break very long tokens */
    line-height: 1.35 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("📊 SQL And Visualization Generator")
st.markdown("Type a question in English. I’ll generate the SQL, run it, and show the best visualization.")

# --- Inputs ---
# Use a controlled text input so Streamlit re-runs predictably.
question = st.text_input(
    "Your question",
    key="question",
    placeholder="e.g., What is the monthly trend of total sales?",
)

# Advanced options are tucked away to keep the main UI clean.
with st.expander("Advanced (optional)"):
    max_retries = st.number_input(
        "Max retries (SQL & Viz)",  # Applies to both SQL repair and viz code repair.
        min_value=0,
        max_value=6,
        value=3,
        step=1,
        key="max_retries",
    )

# --- Session state for last results ---
# Keep prior results stable across Streamlit's re-runs (e.g., when clicking download).
if "last_state" not in st.session_state:
    st.session_state.last_state = None

# --- Run action: compute once, store in session ---
if st.button("Run", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking, generating SQL, validating, and visualizing…"):
            # The heavy lifting happens in backend workflow modules.
            st.session_state.last_state = run_full(question, max_retries=max_retries)

# Always render from session (so reruns e.g., downloads don't clear the UI)
state = st.session_state.last_state

if state is None:
    st.info("Enter a question and click **Run** to see results.")
else:
    # Two-pane layout: left shows metadata/debug; right shows viz.
    c1, c2 = st.columns([0.45, 0.55])

    with c1:
        st.subheader("Generated SQL")
        st.code(state["sql"], language="sql")

        sql_text = state.get("sql", "") or ""

        # Optional: a large, scrollable viewer for extra-long SQL
        with st.expander("🔎 View full SQL (large)"):
            st.text_area(
                label="SQL",
                value=sql_text,
                height=650,
                label_visibility="collapsed",
            )

        col_sql_1, col_sql_2 = st.columns(2)

        with col_sql_1:
            # One-click save of the produced SQL for offline use.
            st.download_button(
                "Download SQL",
                data=sql_text.encode("utf-8"),
                file_name="query.sql",
                mime="text/sql",
                use_container_width=True,
                key="download_sql_btn",
            )

        with col_sql_2:
            # Safe copy-to-clipboard using an injected, sandboxed script.
            # JSON-encode the SQL to avoid breaking out of the string with quotes/newlines.
            escaped_sql = json.dumps(sql_text)
            components.html(
                f"""
                <div style="display:flex;gap:8px;align-items:center;">
                  <button
                    id="copy-sql-btn"
                    style="width:100%;padding:0.5rem 0.75rem;border:1px solid #ddd;border-radius:6px;cursor:pointer;background:#f6f6f6;"
                  >Copy SQL</button>
                </div>
                <script>
                  const SQL = {escaped_sql};
                  const btn = document.getElementById('copy-sql-btn');
                  btn.addEventListener('click', async () => {{
                    try {{
                      await navigator.clipboard.writeText(SQL);
                      const old = btn.innerText;
                      btn.innerText = 'Copied!';
                      setTimeout(() => btn.innerText = old, 1200);
                    }} catch (err) {{
                      console.error(err);
                    }}
                  }});
                </script>
                """,
                height=80,
            )

        # Agent reasoning expander (subquestions + deduped columns)
        with st.expander("🧠 Agent reasoning (subquestions & selected columns)"):
            st.caption("Subquestions → table mapping")
            st.write(state.get("subquestions", []))
            st.caption("Selected columns (deduped)")
            st.write(state.get("columns_selected", []))

        st.caption("Filters (raw → matched)")
        st.code(str(state.get("filters_raw", "")))
        st.code(str(state.get("filters_matched", "")))

        st.subheader("BI Expert Recommendation")
        st.write(state.get("visualization_request", ""))

        st.subheader("Generated Python (Plotly)")
        viz_code_str = state.get("python_code_data_visualization", "")
        st.code(viz_code_str, language="python")

        # Download the visualization code as .py
        if viz_code_str.strip():
            st.download_button(
                "Download viz code (.py)",
                data=viz_code_str.encode("utf-8"),
                file_name="visualization.py",
                mime="text/x-python",
                use_container_width=True,
                key="download_viz_code_btn",
            )

        st.subheader("SQL Validation")
        status = state.get("result_debug_sql", "")
        st.markdown(f"**Status:** {status or 'Unknown'}")

        # Friendly summary + raw error toggle
        err_text = state.get("error_msg_debug_sql", "")
        if status == "Not Pass":
            st.warning(
                "I couldn't execute the SQL successfully even after automatic fixes. "
                "Try rephrasing your question, narrowing the scope, or copy the SQL to tweak it manually."
            )
            if err_text:
                with st.expander("Show raw error details"):
                    st.code(err_text)
        elif err_text:
            with st.expander("Show raw error details"):
                st.code(err_text)

    with c2:
        st.subheader("Result")
        # The workflow executes the code and returns all variables in this dict.
        d = state.get("python_code_store_variables_dict", {}) or {}
        fig = d.get("fig")
        df_viz = d.get("df_viz")
        text_v = d.get("string_viz_result")

        # Render in priority order: figure > table > text.
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

        elif isinstance(df_viz, pd.DataFrame):
            st.dataframe(df_viz, use_container_width=True)
        elif text_v:
            st.markdown(text_v)
        else:
            st.info("No figure/table/text produced by the visualization code.")

        # Provide a CSV download: prefer the viz table (if produced), else raw SQL df.
        download_df = None
        if isinstance(df_viz, pd.DataFrame) and not df_viz.empty:
            download_df = df_viz
        elif isinstance(state.get("df"), pd.DataFrame) and not state["df"].empty:
            download_df = state["df"]

        if download_df is not None:
            csv_bytes = download_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results (CSV)",
                data=csv_bytes,
                file_name="results.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_results_btn",
            )
