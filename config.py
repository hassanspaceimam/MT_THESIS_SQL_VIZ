# =============================================================================
# config.py — Azure OpenAI + Snowflake SQLAlchemy engine (Snowflake-only)
# =============================================================================

import os
from functools import lru_cache
from sqlalchemy import create_engine, text
from langchain_openai import AzureChatOpenAI

# --- Load .env ---
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.getenv("ENV_FILE", ".env"), override=False)
except Exception:
    pass

# --- Azure OpenAI ---
AZURE_ENDPOINT    = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip()
AZURE_DEPLOYMENT  = (os.getenv("AZURE_OPENAI_DEPLOYMENT") or "").strip()
AZURE_API_VERSION = (os.getenv("AZURE_OPENAI_API_VERSION") or "").strip()
AZURE_API_KEY     = (os.getenv("AZURE_OPENAI_API_KEY") or "").strip()

# --- Snowflake (required) ---
SF_ACCOUNT   = (os.getenv("SF_ACCOUNT")   or "").strip()
SF_USER      = (os.getenv("SF_USER")      or "").strip()
SF_PASSWORD  = (os.getenv("SF_PASSWORD")  or "").strip()
SF_WAREHOUSE = (os.getenv("SF_WAREHOUSE") or "").strip()
SF_DATABASE  = (os.getenv("SF_DATABASE")  or "").strip()
SF_SCHEMA    = (os.getenv("SF_SCHEMA")    or "").strip()
SF_ROLE      = (os.getenv("SF_ROLE")      or "").strip()  # optional

# --- Knowledgebase path ---
DEFAULT_KB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledgebase.pkl")
KNOWLEDGEBASE_PATH = os.getenv("KNOWLEDGEBASE_PATH", DEFAULT_KB)


def _normalize_azure_endpoint(url: str) -> str:
    """AzureChatOpenAI needs the resource base URL only."""
    if not url:
        return url
    base = url.split("?", 1)[0]
    if "/openai/" in base:
        base = base.split("/openai/", 1)[0]
    if not base.endswith("/"):
        base += "/"
    return base


@lru_cache(maxsize=1)
def get_llm() -> AzureChatOpenAI:
    endpoint = _normalize_azure_endpoint(AZURE_ENDPOINT)
    missing = [k for k, v in {
        "AZURE_OPENAI_ENDPOINT": endpoint,
        "AZURE_OPENAI_DEPLOYMENT": AZURE_DEPLOYMENT,
        "AZURE_OPENAI_API_VERSION": AZURE_API_VERSION,
        "AZURE_OPENAI_API_KEY": AZURE_API_KEY,
    }.items() if not v]
    if missing:
        raise ValueError(f"Missing Azure OpenAI env vars: {', '.join(missing)}")

    # Send reasoning controls via extra_body to avoid model_kwargs/explicit-kwarg warnings.
    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        azure_deployment=AZURE_DEPLOYMENT,   # e.g., "o4-mini"
        api_version=AZURE_API_VERSION,       # e.g., "2025-04-01-preview"
        api_key=AZURE_API_KEY,
        extra_body={
            "reasoning_effort": "medium",     # low | medium | high  (gpt-5 also supports "minimal")
            "max_completion_tokens": 5000,    # adjust for your thesis as needed
        },
    )


@lru_cache(maxsize=1)
def get_engine():
    # Build a Snowflake SQLAlchemy URL manually (avoids relying on DATABASE_URL).
    from urllib.parse import quote_plus

    required = {
        "SF_ACCOUNT": SF_ACCOUNT, "SF_USER": SF_USER, "SF_PASSWORD": SF_PASSWORD,
        "SF_WAREHOUSE": SF_WAREHOUSE, "SF_DATABASE": SF_DATABASE, "SF_SCHEMA": SF_SCHEMA
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(f"Missing Snowflake env vars: {', '.join(missing)}")

    user = quote_plus(SF_USER)
    pwd  = quote_plus(SF_PASSWORD)
    acct = SF_ACCOUNT
    db   = SF_DATABASE
    sch  = SF_SCHEMA

    q = f"warehouse={SF_WAREHOUSE}"
    if SF_ROLE:
        q += f"&role={SF_ROLE}"

    url = f"snowflake://{user}:{pwd}@{acct}/{db}/{sch}?{q}"
    engine = create_engine(url)

    # Session tweaks for consistent timestamp behavior & auto formats.
    with engine.begin() as conn:
        conn.execute(text("ALTER SESSION SET TIMESTAMP_TYPE_MAPPING = 'TIMESTAMP_NTZ'"))
        conn.execute(text("ALTER SESSION SET DATE_INPUT_FORMAT = 'AUTO'"))
        conn.execute(text("ALTER SESSION SET TIMESTAMP_INPUT_FORMAT = 'AUTO'"))

    return engine


@lru_cache(maxsize=1)
def get_knowledgebase_path() -> str:
    return KNOWLEDGEBASE_PATH
