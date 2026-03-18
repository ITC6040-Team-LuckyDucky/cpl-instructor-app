import os
import sqlite3


def is_local_dev():
    """Returns True when SQL_CONNECTION_STRING is absent (local dev mode)."""
    return not os.getenv("SQL_CONNECTION_STRING")


def get_db_connection():
    """
    Returns an open database connection.
    - Azure SQL (pyodbc) when SQL_CONNECTION_STRING is set.
    - SQLite (local_dev.db) otherwise.

    Both expose the same .cursor() / .commit() / .close() interface,
    so callers don't need to know which backend is in use.
    """
    if is_local_dev():
        conn = sqlite3.connect("local_dev.db", timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row  # lets you access columns by name
        _ensure_sqlite_schema(conn)
        return conn

    import pyodbc
    conn_str = os.getenv("SQL_CONNECTION_STRING")
    return pyodbc.connect(conn_str, timeout=10)


# ---------------------------------------------------------------------------
# SQLite schema bootstrap
# Creates tables on first connect so local dev works with zero setup.
# Mirrors the Azure SQL schema from /setup-db (minus SQL Server-specific syntax).
# ---------------------------------------------------------------------------
def _ensure_sqlite_schema(conn):
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_label TEXT
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );

        CREATE TABLE IF NOT EXISTS uploads (
            upload_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            filename TEXT,
            blob_url TEXT,
            content_type TEXT,
            size INTEGER,
            uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            extracted_text TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );

        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            summary_text TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_version TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );
    """)

    conn.commit()
