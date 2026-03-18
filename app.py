import os
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory
from openai import AzureOpenAI

from db import get_db_connection, is_local_dev
from storage import upload_file, ensure_storage_ready


# Explicit template folder for Azure App Service reliability
app = Flask(__name__, template_folder="templates")


# ===============================
# Auto DB Initialization
# Runs table creation once on the first request so the app is ready
# without needing a manual /setup-db call.
# ===============================
_db_initialized = False

@app.before_request
def ensure_db_initialized():
    global _db_initialized
    if _db_initialized:
        return
    _db_initialized = True  # Set early so a parallel request doesn't double-run
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        if is_local_dev():
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
                CREATE TABLE IF NOT EXISTS interview_state (
                    session_id TEXT PRIMARY KEY,
                    current_stage TEXT NOT NULL DEFAULT 'welcome',
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
            """)
        else:
            for stmt in [
                """IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='sessions' AND xtype='U')
                   CREATE TABLE sessions (
                       session_id NVARCHAR(50) PRIMARY KEY,
                       created_at DATETIME DEFAULT GETDATE(),
                       user_label NVARCHAR(100)
                   )""",
                """IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='messages' AND xtype='U')
                   CREATE TABLE messages (
                       id INT IDENTITY(1,1) PRIMARY KEY,
                       session_id NVARCHAR(50) NOT NULL,
                       role NVARCHAR(20) NOT NULL,
                       content NVARCHAR(MAX),
                       timestamp DATETIME DEFAULT GETDATE(),
                       FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                   )""",
                """IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='uploads' AND xtype='U')
                   CREATE TABLE uploads (
                       upload_id NVARCHAR(50) PRIMARY KEY,
                       session_id NVARCHAR(50) NOT NULL,
                       filename NVARCHAR(255),
                       blob_url NVARCHAR(500),
                       content_type NVARCHAR(100),
                       size INT,
                       uploaded_at DATETIME DEFAULT GETDATE(),
                       extracted_text NVARCHAR(MAX),
                       FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                   )""",
                """IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='summaries' AND xtype='U')
                   CREATE TABLE summaries (
                       id INT IDENTITY(1,1) PRIMARY KEY,
                       session_id NVARCHAR(50) NOT NULL,
                       summary_text NVARCHAR(MAX),
                       created_at DATETIME DEFAULT GETDATE(),
                       model_version NVARCHAR(50),
                       FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                   )""",
                """IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='interview_state' AND xtype='U')
                   CREATE TABLE interview_state (
                       session_id NVARCHAR(50) PRIMARY KEY,
                       current_stage NVARCHAR(50) NOT NULL DEFAULT 'welcome',
                       updated_at DATETIME DEFAULT GETDATE(),
                       FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                   )""",
            ]:
                cursor.execute(stmt)

        conn.commit()
        conn.close()
    except Exception:
        app.logger.warning("Auto DB initialization failed — tables may not exist yet", exc_info=True)


# ===============================
# Interview Stage Definitions
# The bot follows this fixed sequence; the app (not the LLM) decides when to advance.
# ===============================
INTERVIEW_STAGES = [
    "welcome",           # Stage 1: Greet and get student's name
    "course_id",         # Stage 2: Identify course/competency area
    "experience",        # Stage 3: Background — where, how long, role
    "skills_reflection", # Stage 4: Skills, knowledge, examples
    "evidence",          # Stage 5: Evidence and document upload
    "summary",           # Stage 6: Summarize and confirm (final stage)
]


def get_current_stage(session_id):
    """Returns the current interview stage for the given session, or 'welcome' if not found."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT current_stage FROM interview_state WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else "welcome"
    except Exception:
        app.logger.exception("Failed to get current stage")
        return "welcome"


def advance_stage(session_id, current_stage):
    """
    Moves the session to the next stage in INTERVIEW_STAGES.
    Does nothing if already at the final stage.
    Returns the new stage name.
    """
    if current_stage not in INTERVIEW_STAGES:
        return current_stage
    idx = INTERVIEW_STAGES.index(current_stage)
    if idx >= len(INTERVIEW_STAGES) - 1:
        return current_stage  # Already at final stage
    next_stage = INTERVIEW_STAGES[idx + 1]
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE interview_state SET current_stage = ?, updated_at = CURRENT_TIMESTAMP "
            "WHERE session_id = ?",
            (next_stage, session_id),
        )
        conn.commit()
        conn.close()
    except Exception:
        app.logger.exception("Failed to advance stage")
    return next_stage


def should_advance(current_stage, user_message, assistant_response):
    """
    Simple heuristic check — returns True if the conversation has collected
    enough information to move past the current stage.
    The LLM does NOT make this decision; the app does.
    """
    msg = user_message.lower()
    words = msg.split()

    if current_stage == "welcome":
        # Advance when the user has given a substantive reply (likely includes their name)
        return len(words) >= 2

    if current_stage == "course_id":
        # Advance when the user mentions a course, subject, or programme
        course_keywords = [
            "course", "class", "subject", "program", "programme", "module",
            "degree", "certificate", "diploma", "credit", "unit",
        ]
        return any(kw in msg for kw in course_keywords) or len(words) >= 4

    if current_stage == "experience":
        # Advance when the user describes a work role or duration
        experience_keywords = [
            "worked", "work", "job", "role", "position", "years", "months",
            "manager", "engineer", "developer", "nurse", "teacher", "director",
            "employed", "company", "organization", "team", "project",
        ]
        return any(kw in msg for kw in experience_keywords)

    if current_stage == "skills_reflection":
        # Advance when the user gives a concrete example or description
        reflection_keywords = [
            "example", "instance", "specifically", "when i", "i did",
            "i used", "i learned", "i managed", "i built", "i created",
            "i led", "responsible", "skill", "knowledge", "ability",
        ]
        return any(kw in msg for kw in reflection_keywords) or len(words) >= 15

    if current_stage == "evidence":
        # Advance when the user indicates they have provided or have no more evidence
        evidence_keywords = [
            "uploaded", "attached", "no more", "that's all", "done",
            "no evidence", "nothing else", "finished", "complete",
        ]
        return any(kw in msg for kw in evidence_keywords)

    # "summary" is the final stage — never advance
    return False


# ===============================
# Azure OpenAI Client Factory
# ===============================
def get_client():
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    if not endpoint:
        return None, "Missing AZURE_OPENAI_ENDPOINT"
    if not api_key:
        return None, "Missing AZURE_OPENAI_API_KEY"

    try:
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        return client, None
    except Exception as e:
        return None, f"Client initialization failed: {type(e).__name__}"


# ===============================
# Static File Route (bulletproof)
# ===============================
@app.get("/static/<path:filename>")
def static_files(filename):
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    return send_from_directory(static_dir, filename)


# ===============================
# Basic Pages
# ===============================
@app.get("/")
def home():
    return render_template("index.html")


@app.get("/chat")
def chat_page():
    return render_template("chat.html")


@app.get("/admin")
def admin_page():
    status = {
        "AZURE_OPENAI_ENDPOINT": " set" if os.getenv("AZURE_OPENAI_ENDPOINT") else "missing",
        "AZURE_OPENAI_API_KEY": " set" if os.getenv("AZURE_OPENAI_API_KEY") else "missing",
        "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION") or "(default: 2024-12-01-preview)",
        "AZURE_OPENAI_DEPLOYMENT": " set" if os.getenv("AZURE_OPENAI_DEPLOYMENT") else "missing",
        # Show whether connection strings are present (never expose the actual values)
        "SQL_CONNECTION_STRING": "set" if os.getenv("SQL_CONNECTION_STRING") else "missing",
        "AZURE_STORAGE_CONNECTION_STRING": "set" if os.getenv("AZURE_STORAGE_CONNECTION_STRING") else "missing",
    }
    return render_template("admin.html", status=status)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


# ===============================
# DEBUG SUPERPOWER ROUTE
# Shows SDK versions for troubleshooting
# ===============================
@app.get("/versions")
def versions():
    try:
        import openai
        import httpx
        return jsonify({
            "openai_version": getattr(openai, "__version__", "unknown"),
            "httpx_version": getattr(httpx, "__version__", "unknown"),
            "python_version": os.sys.version,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===============================
# ✅ DB CHECK ROUTE
# Verifies Web App can connect to Azure SQL
# ===============================
@app.get("/dbcheck")
def dbcheck():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        row = cursor.fetchone()
        conn.close()

        return jsonify({
            "status": "DB Connected",
            "result": int(row[0]),
            "mode": "local (SQLite)" if is_local_dev() else "Azure SQL",
        })
    except Exception as e:
        app.logger.exception("DB connection check failed")
        return jsonify({
            "error": f"DB check failed: {type(e).__name__}",
            "details": str(e),
        }), 500


# ===============================
# 🛠️ DB SETUP ROUTE
# Creates database tables if they don't exist
# ===============================
@app.get("/setup-db")
def setup_db():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        if is_local_dev():
            # SQLite syntax
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

                CREATE TABLE IF NOT EXISTS interview_state (
                    session_id TEXT PRIMARY KEY,
                    current_stage TEXT NOT NULL DEFAULT 'welcome',
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
            """)
        else:
            # Azure SQL syntax
            cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='sessions' AND xtype='U')
                CREATE TABLE sessions (
                    session_id NVARCHAR(50) PRIMARY KEY,
                    created_at DATETIME DEFAULT GETDATE(),
                    user_label NVARCHAR(100)
                )
            """)

            cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='messages' AND xtype='U')
                CREATE TABLE messages (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    session_id NVARCHAR(50) NOT NULL,
                    role NVARCHAR(20) NOT NULL,
                    content NVARCHAR(MAX),
                    timestamp DATETIME DEFAULT GETDATE(),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='uploads' AND xtype='U')
                CREATE TABLE uploads (
                    upload_id NVARCHAR(50) PRIMARY KEY,
                    session_id NVARCHAR(50) NOT NULL,
                    filename NVARCHAR(255),
                    blob_url NVARCHAR(500),
                    content_type NVARCHAR(100),
                    size INT,
                    uploaded_at DATETIME DEFAULT GETDATE(),
                    extracted_text NVARCHAR(MAX),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='summaries' AND xtype='U')
                CREATE TABLE summaries (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    session_id NVARCHAR(50) NOT NULL,
                    summary_text NVARCHAR(MAX),
                    created_at DATETIME DEFAULT GETDATE(),
                    model_version NVARCHAR(50),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='interview_state' AND xtype='U')
                CREATE TABLE interview_state (
                    session_id NVARCHAR(50) PRIMARY KEY,
                    current_stage NVARCHAR(50) NOT NULL DEFAULT 'welcome',
                    updated_at DATETIME DEFAULT GETDATE(),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

        conn.commit()
        conn.close()

        return jsonify({
            "status": "success",
            "message": "Database tables created successfully",
            "mode": "local (SQLite)" if is_local_dev() else "Azure SQL",
            "tables": ["sessions", "messages", "uploads", "summaries", "interview_state"],
        })

    except Exception as e:
        app.logger.exception("Database setup failed")
        return jsonify({
            "error": f"Database setup failed: {type(e).__name__}",
            "details": str(e),
        }), 500


# ===============================
# DB MIGRATION ROUTE
# Adds extracted_text column to existing uploads table
# ===============================
@app.get("/migrate-uploads")
def migrate_uploads():
    if is_local_dev():
        # SQLite schema already includes extracted_text; nothing to do
        return jsonify({
            "status": "success",
            "message": "Local dev: extracted_text already present in SQLite schema",
            "mode": "local (SQLite)",
        })

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID('uploads') AND name = 'extracted_text')
                ALTER TABLE uploads ADD extracted_text NVARCHAR(MAX)
        """)

        conn.commit()
        conn.close()

        return jsonify({"status": "success", "message": "Migration complete: extracted_text column ensured on uploads table"})

    except Exception as e:
        app.logger.exception("Migration failed")
        return jsonify({
            "error": f"Migration failed: {type(e).__name__}",
            "details": str(e),
        }), 500


# ===============================
# File Upload Endpoint
# ===============================
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def extract_text(file_bytes, ext):
    if ext == ".txt":
        return file_bytes.decode("utf-8", errors="replace")

    if ext == ".pdf":
        import io
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if ext == ".docx":
        import io
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)

    return None


@app.post("/api/upload")
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    filename = f.filename or ""
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"File type '{ext}' not allowed. Use .pdf, .txt, or .docx"}), 400

    file_bytes = f.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        return jsonify({"error": "File exceeds 10 MB limit"}), 400

    upload_id = str(uuid.uuid4())

    # Store file via storage helper (Azure Blob or local folder)
    try:
        ensure_storage_ready()
        blob_url = upload_file(file_bytes, filename, upload_id)
    except Exception as e:
        app.logger.exception("File storage failed")
        return jsonify({"error": f"File storage failed: {type(e).__name__}", "details": str(e)}), 500

    # Extract text — failure is non-fatal
    extracted_text = None
    try:
        extracted_text = extract_text(file_bytes, ext)
    except Exception:
        app.logger.exception("Text extraction failed; storing NULL")

    # Read session_id from the form data (set by the frontend after /api/session)
    session_id = (request.form.get("session_id") or "").strip()
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    # Save record to DB
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO uploads
                (upload_id, session_id, filename, blob_url, content_type, size, uploaded_at, extracted_text)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
            """,
            (upload_id,
            session_id,
            filename,
            blob_url,
            f.content_type or "",
            len(file_bytes),
            extracted_text),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        app.logger.exception("DB insert failed")
        return jsonify({"error": f"DB insert failed: {type(e).__name__}", "details": str(e)}), 500

    return jsonify({
        "status": "success",
        "upload_id": upload_id,
        "filename": filename,
        "size": len(file_bytes),
        "extracted_text_length": len(extracted_text) if extracted_text else 0,
    })


# ===============================
# List Uploads Endpoint
# ===============================
@app.get("/api/uploads")
def api_list_uploads():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT upload_id, filename, content_type, size, uploaded_at, blob_url "
            "FROM uploads ORDER BY uploaded_at DESC"
        )
        rows = cursor.fetchall()
        conn.close()

        uploads = [
            {
                "upload_id": row[0],
                "filename":  row[1],
                "content_type": row[2],
                "size": row[3],
                "uploaded_at": str(row[4]),
                "blob_url": row[5],
            }
            for row in rows
        ]
        return jsonify({"uploads": uploads})

    except Exception as e:
        app.logger.exception("Failed to list uploads")
        return jsonify({"error": f"Failed to list uploads: {type(e).__name__}", "details": str(e)}), 500


# ===============================
# Session Endpoint
# Creates a new session row and returns its ID
# ===============================
@app.post("/api/session")
def api_session():
    try:
        session_id = str(uuid.uuid4())

        conn = get_db_connection()
        cursor = conn.cursor()
        # Insert new session row; created_at defaults to now in both SQLite and Azure SQL
        cursor.execute(
            "INSERT INTO sessions (session_id) VALUES (?)",
            (session_id,),
        )
        # Seed the interview state at the first stage
        cursor.execute(
            "INSERT INTO interview_state (session_id, current_stage) VALUES (?, ?)",
            (session_id, INTERVIEW_STAGES[0]),
        )
        conn.commit()
        conn.close()

        return jsonify({"session_id": session_id, "stage": INTERVIEW_STAGES[0]})

    except Exception as e:
        app.logger.exception("Failed to create session")
        return jsonify({"error": f"Failed to create session: {type(e).__name__}", "details": str(e)}), 500


# ===============================
# Chat API Endpoint
# Requires session_id; persists user + assistant messages to DB
# ===============================
@app.post("/api/chat")
def api_chat():
    try:
        data = request.get_json(silent=True) or {}
        user_message = (data.get("message") or "").strip()
        session_id = (data.get("session_id") or "").strip()

        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        if not session_id:
            return jsonify({"error": "session_id is required"}), 400

        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        if not deployment:
            return jsonify({"error": "Missing AZURE_OPENAI_DEPLOYMENT"}), 500

        client, err = get_client()
        if err:
            return jsonify({"error": err}), 500

        # Fetch conversation history for this session (most recent 20 messages)
        history = []
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, content FROM messages "
                "WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,),
            )
            rows = cursor.fetchall()
            conn.close()
            # Keep only the last 20 to stay within token limits
            history = [{"role": row[0], "content": row[1]} for row in rows[-20:]]
        except Exception:
            # History is best-effort — proceed with empty history if DB fails
            app.logger.exception("Failed to load message history")

        # Read current stage so we can inform the LLM which part of the interview we're in
        current_stage = get_current_stage(session_id)

        # Build the full messages array: system prompt + history + current user message
        messages = [
            {"role": "system", "content": (
                "You are a CPL (Credit for Prior Learning) advisor assistant. Your job is to interview "
                "students about their professional experience and help them articulate their skills for "
                "academic credit evaluation. Be friendly, ask clarifying follow-up questions, and help "
                "them identify relevant evidence of their learning. Start by asking what course or "
                "competency area they want to receive credit for.\n\n"
                f"Current interview stage: {current_stage}"
            )},
            *history,
            {"role": "user", "content": user_message},
        ]

        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=0.3,
        )

        answer = (response.choices[0].message.content or "").strip()

        # Save both turns to the messages table
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, "user", user_message),
            )
            cursor.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, "assistant", answer),
            )
            conn.commit()
            conn.close()
        except Exception:
            # Log but don't fail the chat response — message saving is best-effort
            app.logger.exception("Failed to save messages to DB")

        # Check whether the app should advance to the next interview stage
        # (current_stage was already fetched above before the OpenAI call)
        if should_advance(current_stage, user_message, answer):
            current_stage = advance_stage(session_id, current_stage)

        return jsonify({"answer": answer, "stage": current_stage})

    except Exception as e:
        app.logger.exception("Azure OpenAI call failed")
        return jsonify({
            "error": f"Azure OpenAI call failed: {type(e).__name__}"
        }), 500


# ===============================
# Local Dev Entry Point
# ===============================
if __name__ == "__main__":
    if is_local_dev():
        print("Running in LOCAL DEV mode (SQLite + local file storage)")
    else:
        print("Running in AZURE mode")
    app.run(host="0.0.0.0", port=8000)
