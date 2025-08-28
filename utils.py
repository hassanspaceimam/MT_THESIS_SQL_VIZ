# utils.py

"""
utils.py

Small utilities for:
- Robustly parsing model outputs (lists, code blocks, SQL snippets),
- Extracting fenced code,
- Fuzzy-matching categorical filter values against actual DB values.

Important:
- Functions here are side-effect free except fuzzy matchers, which read the DB.
- Snowflake dialect: identifiers are case-insensitive when unquoted; we therefore
  use **unquoted, sanitized** names for safety (no backticks or brackets).
"""

from __future__ import annotations
import ast
import json
import re
import unicodedata
from typing import List
from functools import lru_cache

import pandas as pd
from sqlalchemy import text

from config import get_engine

# -------------- Parsing helpers --------------
def parse_nested_list(text_in: str) -> list:
    """Parse model output into a Python list; tries JSON, then literal_eval, then bracket extraction."""
    if not text_in:
        return []
    s = str(text_in).strip()
    # Try JSON
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, list) else []
    except Exception:
        pass
    # Try Python literal
    try:
        obj = ast.literal_eval(s)
        return obj if isinstance(obj, list) else []
    except Exception:
        pass
    # Fallback: first top-level [ [ ... ], ... ] pattern
    m = re.search(r"\[\s*\[.*?\]\s*(,\s*\[.*?\]\s*)*\]", s, re.DOTALL)
    if m:
        try:
            obj = ast.literal_eval(m.group(0))
            return obj if isinstance(obj, list) else []
        except Exception:
            return []
    return []

def normalize_subquestions(entries: list) -> List[List[str]]:
    """Ensure each entry is exactly [subquestion, table]."""
    norm: List[List[str]] = []
    for e in entries:
        if isinstance(e, list) and len(e) >= 2:
            subq = str(e[0]).strip()
            table = str(e[1]).strip()
            if subq and table:
                norm.append([subq, table])
    return norm

def extract_sql(text_in: str) -> str:
    """
    Extract SQL from a ```sql ...``` fenced block, else preserve full CTEs that
    start with WITH, else from first SELECT onward, else raw stripped.
    """
    if not text_in:
        return ""
    s = str(text_in).strip()

    # Prefer fenced ```sql ... ```
    m = re.search(r"```(?:\s*sql)?\s*(.*?)```", s, flags=re.I | re.S)
    if m:
        return m.group(1).strip()

    # If the statement begins with a CTE, keep it intact.
    if re.match(r"(?is)^\s*with\b", s):
        return s

    # Else grab from the first SELECT onward.
    m = re.search(r"(?is)\bselect\b.*", s)
    if m:
        return m.group(0).strip()

    return s

# -------------- Code block extraction --------------
def extract_code_block(content: str, language: str) -> str:
    """
    Extract code from a fenced block: ```<language> ... ```
    If not found, return the first fenced block or content without backticks.
    """
    if content is None:
        return ""
    s = str(content)
    # ```language ... ```
    pattern_lang = rf"```(?:\s*{re.escape(language)})\s*(.*?)```"
    m = re.search(pattern_lang, s, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # First fenced block
    m = re.search(r"```(.*?)```", s, re.DOTALL)
    if m:
        return m.group(1).strip()
    return s.replace("```", "").strip()

# -------------- Fuzzy filter matcher --------------
_engine = get_engine()

def _quote_ident(name: str) -> str | None:
    """
    Minimal identifier sanitizer for Snowflake.
    - Allow only letters, numbers, and underscores.
    - Return an **unquoted** identifier (Snowflake resolves to UPPER case).
    - Return None if invalid.
    """
    if not isinstance(name, str):
        return None
    if not re.match(r"^[A-Za-z0-9_]+$", name):
        return None
    return name  # unquoted (case-insensitive in Snowflake)

@lru_cache(maxsize=512)
def _get_values(table_name: str, column_name: str):
    qt = _quote_ident(table_name)
    qc = _quote_ident(column_name)
    if not qt or not qc:
        # If names look unsafe/invalid, avoid executing SQL; let fuzzy matcher pass-through.
        return []
    q = text(f"SELECT DISTINCT {qc} AS v FROM {qt}")
    df = pd.read_sql(q, con=_engine)
    return df["v"].dropna().astype(str).tolist()

def _best_fuzzy_match(input_value: str, choices):
    # replacement for rapidfuzz token_set_ratio using simple heuristic if RF not installed
    try:
        from rapidfuzz import process, fuzz
        match, score, _ = process.extractOne(input_value, choices, scorer=fuzz.token_set_ratio)
        return match, score
    except Exception:
        # Very light fallback: exact or casefold match, else return original
        s = str(input_value).casefold()
        for c in choices:
            if s == str(c).casefold():
                return c, 100
        return input_value, 0

def _flatten_filters_structure(filters):
    """
    Accept either:
      ["yes", ["t","c","v"], ["t","c","v"], ...]
    or the nested variant:
      ["yes", [ ["t","c","v"], ["t","c","v"], ... ]]
    and normalize to the flat version.
    """
    if (
        isinstance(filters, list)
        and filters
        and filters[0] == "yes"
        and len(filters) == 2
        and isinstance(filters[1], list)
        and filters[1]
        and all(isinstance(x, list) for x in filters[1])
    ):
        return ["yes", *filters[1]]
    return filters

# ---- Helpers for robust Unicode-aware fuzzy matching ----
def _ascii_fold(s: str) -> str:
    """Remove accents/diacritics but keep letters/numbers/spaces."""
    if not isinstance(s, str):
        s = str(s)
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

def _has_letters_any_unicode(s: str) -> bool:
    """True if the string contains any alphabetic character (Unicode-aware)."""
    return any(ch.isalpha() for ch in str(s))

def _has_operators_or_dates(s: str) -> bool:
    """Heuristic for numeric/date/range predicates we shouldn't fuzzy-match."""
    s = str(s)
    return bool(re.search(r"\bbetween\b|<=|>=|<|>|before|after|\d{4}-\d{2}-\d{2}", s, re.I))

def _split_values_list(s: str):
    """Split a predicate like 'credit card, boleto' into ['credit card', 'boleto']."""
    # split on comma/semicolon and also textual ' or ' / ' and '
    parts = re.split(r"(?:\s+or\s+|\s+and\s+|[;,])", str(s), flags=re.I)
    return [p.strip() for p in parts if p and p.strip()]

def _word_initials(s: str) -> str:
    """Initials from words in a string, Unicode-aware: 'São Paulo' -> 'SP'."""
    words, cur = [], ""
    for ch in str(s):
        if ch.isalpha():
            cur += ch
        else:
            if cur:
                words.append(cur)
                cur = ""
    if cur:
        words.append(cur)
    return ''.join(w[0] for w in words if w).upper()

def _normalize_token(s: str) -> str:
    """
    Normalize for matching: lowercase, ASCII-fold, collapse spaces/hyphens to underscores,
    and strip non-alphanum/underscore.
    """
    s = _ascii_fold(str(s)).lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^\w]", "", s)
    return s

def _maybe_redirect_city_to_state(table: str, column: str, predicate: str):
    """
    If the LLM chose a *_city column but the DB stores states as abbreviations,
    attempt to redirect to the corresponding *_state column using initials.
    Example: 'São Paulo' -> 'SP' on customer.customer_state.
    """
    if table in ("customer", "sellers") and column.endswith("_city"):
        state_col = column[:-5] + "_state"  # replace suffix _city -> _state
        state_choices = _get_values(table, state_col)
        if state_choices:
            abbrev_set = {c for c in state_choices if isinstance(c, str) and c.isupper() and 1 <= len(c) <= 3}
            if abbrev_set:
                initials = _word_initials(predicate)
                if initials:
                    return state_col, initials
    return None, None

def fuzzy_match_filters(filters):
    """
    For categorical equality-like predicates (no operators), fuzzy-match each value
    to the column's distinct set. Handles Unicode (e.g., 'São Paulo') and
    comma/semicolon lists ('credit card, boleto'). If choices look like abbreviations
    (2–3 uppercase chars), will also try initials ('São Paulo' -> 'SP').

    Additionally, if a city column was selected for customers/sellers but the state
    column contains abbreviations, this function will *redirect* the filter to the
    *_state column when initials match the stored abbreviations.

    Input:
      ["yes", ["table","column","predicate"], ...]  or
      ["yes", [ ["table","column","predicate"], ... ]]
    Output (same shape, normalized to flat):
      ["yes", ["table","column","matched_predicate"], ...]
      where matched_predicate preserves multi-values order, comma-separated.
    """
    if not isinstance(filters, list) or not filters or filters[0] == "no":
        return filters

    filters = _flatten_filters_structure(filters)
    out = ["yes"]

    for t in filters[1:]:
        if not isinstance(t, list) or len(t) < 3:
            continue
        table, column, predicate = t[0], t[1], str(t[2]).strip()

        # If it's clearly numeric/date/range, pass through unchanged
        if _has_operators_or_dates(predicate):
            out.append([table, column, predicate])
            continue

        # Only attempt fuzzy when it looks textual in any language
        if not _has_letters_any_unicode(predicate):
            out.append([table, column, predicate])
            continue

        # Redirect city -> state when appropriate (customer/sellers)
        new_col, new_pred = _maybe_redirect_city_to_state(table, column, predicate)
        if new_col and new_pred:
            column = new_col
            predicate = new_pred

        # Fetch choices for (possibly redirected) column
        choices = _get_values(table, column)
        if not choices:
            out.append([table, column, predicate])
            continue

        # Prepare alternative choice spaces
        choices_norm = [_normalize_token(c) for c in choices]
        choices_fold = [_ascii_fold(c) for c in choices]

        # Support multiple comma/semicolon/'or'/'and' separated values
        values = _split_values_list(predicate) or [predicate]
        matched_values = []

        # Precompute if choices look like abbreviations (e.g., 'SP', 'RJ')
        abbrev_set = {c for c in choices if isinstance(c, str) and c.isupper() and 1 <= len(c) <= 3}

        for v in values:
            # 1) Direct fuzzy on originals
            best, score = _best_fuzzy_match(v, choices)

            # 2) If weak, try ASCII-folded space (handles accents)
            if score < 60:
                v_fold = _ascii_fold(v)
                fold_best, score2 = _best_fuzzy_match(v_fold, choices_fold)
                if score2 > score:
                    try:
                        idx = choices_fold.index(fold_best)
                        best = choices[idx]
                        score = score2
                    except ValueError:
                        pass

            # 3) If still weak, try normalized tokens (spaces->underscores etc.)
            if score < 60:
                v_norm = _normalize_token(v)
                norm_best, score3 = _best_fuzzy_match(v_norm, choices_norm)
                if score3 > score:
                    try:
                        idx = choices_norm.index(norm_best)
                        best = choices[idx]
                        score = score3
                    except ValueError:
                        pass

            # 4) If still weak and the column stores abbreviations, try initials
            if score < 60 and abbrev_set:
                initials = _word_initials(v)
                if initials in abbrev_set:
                    best = initials
                    score = 100

            matched_values.append(best)

        # Re-join multi-values with comma + space to match prompt examples
        out.append([table, column, ", ".join(matched_values)])

    return out
