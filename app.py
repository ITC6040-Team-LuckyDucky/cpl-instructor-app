import os
import re
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
                CREATE TABLE IF NOT EXISTS collected_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    field_value TEXT,
                    stage TEXT,
                    collected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
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
                """IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='collected_data' AND xtype='U')
                   CREATE TABLE collected_data (
                       id INT IDENTITY(1,1) PRIMARY KEY,
                       session_id NVARCHAR(50) NOT NULL,
                       field_name NVARCHAR(100) NOT NULL,
                       field_value NVARCHAR(MAX),
                       stage NVARCHAR(50),
                       collected_at DATETIME DEFAULT GETDATE(),
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

# Fields to extract from the conversation at each stage.
# The extraction LLM call uses these to know what JSON keys to return.
STAGE_FIELDS = {
    "welcome":           ["student_name"],
    "course_id":         ["course_name", "competency_area"],
    "experience":        ["experience_description", "employer_or_source", "role", "duration"],
    "skills_reflection": ["skills_learned", "concrete_examples"],
    "evidence":          ["evidence_description", "uploaded_documents"],
    "summary":           [],  # Nothing new to extract at summary stage
}


def get_system_prompt(stage, collected_data=None):
    """
    Returns a stage-specific system prompt that constrains the bot to the
    current interview topic. collected_data is a dict with keys like
    'name', 'course', 'experience_summary' populated by prior stages.
    """
    d = collected_data or {}
    name = d.get("student_name") or "the student"
    course = d.get("course_name") or d.get("competency_area") or "their chosen course"
    experience_summary = d.get("experience_description") or "their prior experience"

    STYLE = (
        "Keep your response under 3 sentences. "
        "Ask only ONE question. "
        "Do not use bullet points, numbered lists, or markdown formatting. "
        "Speak naturally like a friendly interviewer."
    )

    prompts = {
        "welcome": (
            "You are a CPL (Credit for Prior Learning) interview assistant. "
            "Follow this exact sequence, one question per response, and ALWAYS follow up — never stop at a greeting. "
            "Step 1: Ask for the student's name. "
            "Step 2: After they give their name, say a brief greeting and ask: 'Are you a current Northeastern University student?' (if yes, ask for their NUID). "
            "Step 3: After they answer, ask: 'What is your major or intended major?' "
            "Step 4: After they answer, ask exactly this: 'How can I help you today? A: I'd like to waive a course based on my prior experience. B: I'd like to check if my experience qualifies me for a specific course. C: I have other questions about CPL.' "
            "You must always move to the next step immediately. Never stop after a greeting. "
            f"{STYLE}"
        ),
        "course_id": (
            f"You are a CPL interview assistant. The student's name is {name}. "
            "Ask them which course or competency area they want to receive credit for. "
            "Do not ask anything else yet. "
            f"{STYLE}"
        ),
        "experience": (
            f"You are a CPL interview assistant helping {name} seek credit for {course}. "
            "Ask ONE question about their most relevant hands-on experience related to that course — "
            "focus on what they actually did, not general background. "
            "Do not ask about skills or evidence yet. "
            f"{STYLE}"
        ),
        "skills_reflection": (
            f"You are a CPL interview assistant helping {name} seek credit for {course}. "
            f"They described their experience as: {experience_summary}. "
            "This is the most critical stage — you need to determine whether the student already "
            "has the knowledge and skills that this course teaches. "
            "Ask for ONE specific example that proves they already know the course material. "
            "Do not ask about evidence or move to other topics. "
            f"{STYLE}"
        ),
        "evidence": (
            f"You are a CPL interview assistant helping {name} seek credit for {course}. "
            "Ask if they have any documents that prove their experience, such as certificates, "
            "work samples, or letters. They can upload files using the button in the chat. "
            f"{STYLE}"
        ),
        "summary": (
            f"You are a CPL interview assistant. The interview with {name} is complete. "
            "Briefly confirm what they shared: the course they want credit for, their key experience, "
            "and the strongest example they gave. Keep it to 2 or 3 sentences — no lists, no headers. "
            f"{STYLE}"
        ),
    }

    return prompts.get(stage, prompts["welcome"])


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
        # Advance only when the student picks an option (A, B, or C) from the final menu
        return bool(re.search(r'\b(option\s*)?[abc]\b', msg) or
                    re.search(r'\b(waive|waiver|qualify|qualifies|other questions?)\b', msg))

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


def get_collected_data(session_id):
    """
    Returns all collected structured data for a session as a flat dict
    {field_name: field_value}. When a field has been collected more than
    once (e.g. the student corrected their name) the most-recent value wins.
    """
    result = {}
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Order ASC so later rows overwrite earlier ones in the dict
        cursor.execute(
            "SELECT field_name, field_value FROM collected_data "
            "WHERE session_id = ? ORDER BY collected_at ASC",
            (session_id,),
        )
        for row in cursor.fetchall():
            result[row[0]] = row[1]
        conn.close()
    except Exception:
        app.logger.exception("Failed to load collected data")
    return result


def extract_fields(session_id, stage, user_message, assistant_response):
    """
    Makes a SEPARATE Azure OpenAI call to extract structured fields from the
    latest conversation exchange and saves them to the collected_data table.

    Skipped gracefully if:
      - The stage has no fields to extract
      - Azure OpenAI env vars are missing (e.g. local dev without Azure)
      - The LLM returns unparseable JSON
    """
    import json

    fields = STAGE_FIELDS.get(stage, [])
    if not fields:
        return  # Nothing to extract at this stage

    # Skip if Azure OpenAI is not configured (safe for local dev)
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if not deployment:
        return

    client, err = get_client()
    if err:
        app.logger.warning(f"extract_fields: skipping — {err}")
        return

    fields_str = ", ".join(f'"{f}"' for f in fields)
    extraction_prompt = (
        f"Given the following conversation exchange, extract these fields as JSON: [{fields_str}]. "
        "Return ONLY valid JSON with exactly those keys. "
        'If a field is not mentioned or cannot be determined, use null. '
        "Do not include any explanation or extra text — ONLY the JSON object.\n\n"
        f"User said: {user_message}\n"
        f"Assistant said: {assistant_response}"
    )

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0,
        )
        raw = (response.choices[0].message.content or "").strip()
        extracted = json.loads(raw)

        # Save each non-null field to the DB
        conn = get_db_connection()
        cursor = conn.cursor()
        for field_name, field_value in extracted.items():
            if field_value is not None and field_name in fields:
                cursor.execute(
                    "INSERT INTO collected_data "
                    "(session_id, field_name, field_value, stage, collected_at) "
                    "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
                    (session_id, field_name, str(field_value), stage),
                )
        conn.commit()
        conn.close()

    except Exception:
        # Extraction is best-effort — never crash the chat response
        app.logger.exception("extract_fields failed; continuing without extracted data")


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

                CREATE TABLE IF NOT EXISTS collected_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    field_value TEXT,
                    stage TEXT,
                    collected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
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

            cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='collected_data' AND xtype='U')
                CREATE TABLE collected_data (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    session_id NVARCHAR(50) NOT NULL,
                    field_name NVARCHAR(100) NOT NULL,
                    field_value NVARCHAR(MAX),
                    stage NVARCHAR(50),
                    collected_at DATETIME DEFAULT GETDATE(),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

        conn.commit()
        conn.close()

        return jsonify({
            "status": "success",
            "message": "Database tables created successfully",
            "mode": "local (SQLite)" if is_local_dev() else "Azure SQL",
            "tables": ["sessions", "messages", "uploads", "summaries", "interview_state", "collected_data"],
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
        session_id = (request.args.get("session_id") or "").strip()
        if session_id:
            cursor.execute(
                "SELECT upload_id, filename, content_type, size, uploaded_at, blob_url "
                "FROM uploads WHERE session_id = ? ORDER BY uploaded_at DESC",
                (session_id,),
            )
        else:
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
# Session Data Endpoint
# Returns all collected structured data for a session (useful for debugging
# and for populating the summary stage)
# ===============================
@app.get("/api/session/<session_id>/data")
def api_session_data(session_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT field_name, field_value, stage, collected_at "
            "FROM collected_data WHERE session_id = ? ORDER BY collected_at ASC",
            (session_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        fields = [
            {
                "field_name": row[0],
                "field_value": row[1],
                "stage": row[2],
                "collected_at": str(row[3]),
            }
            for row in rows
        ]
        # Also return as a flat dict for convenience
        flat = {row[0]: row[1] for row in rows}
        return jsonify({"session_id": session_id, "fields": fields, "summary": flat})

    except Exception as e:
        app.logger.exception("Failed to retrieve session data")
        return jsonify({"error": f"Failed to retrieve session data: {type(e).__name__}", "details": str(e)}), 500


# ===============================
# Summary Endpoints
# POST generates a new summary via Azure OpenAI and saves it.
# GET retrieves the most-recently saved summary.
# ===============================
@app.post("/api/session/<session_id>/summary")
def api_generate_summary(session_id):
    try:
        # Require at least some collected data before generating
        collected = get_collected_data(session_id)
        if not collected:
            return jsonify({"error": "No data collected for this session yet"}), 400

        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        if not deployment:
            return jsonify({"error": "Missing AZURE_OPENAI_DEPLOYMENT"}), 500

        client, err = get_client()
        if err:
            return jsonify({"error": err}), 500

        # Fetch conversation history for context (most recent 40 messages)
        history_text = ""
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
            history_text = "\n".join(
                f"{row[0].upper()}: {row[1]}" for row in rows[-40:]
            )
        except Exception:
            app.logger.exception("Failed to load history for summary — continuing without it")

        # Format the collected structured data for the prompt
        collected_text = "\n".join(
            f"- {k}: {v}" for k, v in collected.items() if v
        )

        summary_prompt = (
            "You are a CPL (Credit for Prior Learning) portfolio writer. "
            "Based on the structured data and conversation transcript below, "
            "generate a comprehensive CPL portfolio summary for the student. "
            "The summary must include:\n"
            "  1. Student name\n"
            "  2. Course or competency area they are seeking credit for\n"
            "  3. Relevant professional or life experience (employer, role, duration)\n"
            "  4. Skills and knowledge demonstrated, with concrete examples\n"
            "  5. Evidence provided (documents uploaded or described)\n"
            "  6. A recommendation statement assessing the strength of their CPL claim\n\n"
            "Write in a professional, third-person tone suitable for submission to an academic "
            "review committee. Be thorough but concise.\n\n"
            f"STRUCTURED DATA COLLECTED:\n{collected_text}\n\n"
            f"CONVERSATION TRANSCRIPT:\n{history_text}"
        )

        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.4,
        )

        summary_text = (response.choices[0].message.content or "").strip()

        # Save to the summaries table
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO summaries (session_id, summary_text, model_version) VALUES (?, ?, ?)",
            (session_id, summary_text, deployment),
        )
        conn.commit()
        conn.close()

        return jsonify({
            "status": "success",
            "session_id": session_id,
            "summary": summary_text,
            "model_version": deployment,
        })

    except Exception as e:
        app.logger.exception("Summary generation failed")
        return jsonify({"error": f"Summary generation failed: {type(e).__name__}", "details": str(e)}), 500


@app.get("/api/session/<session_id>/summary")
def api_get_summary(session_id):
    """Returns the most recently generated summary for the session."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT summary_text, model_version, created_at FROM summaries "
            "WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return jsonify({"error": "No summary found for this session"}), 404

        return jsonify({
            "session_id": session_id,
            "summary": row[0],
            "model_version": row[1],
            "created_at": str(row[2]),
        })

    except Exception as e:
        app.logger.exception("Failed to retrieve summary")
        return jsonify({"error": f"Failed to retrieve summary: {type(e).__name__}", "details": str(e)}), 500


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
# Greeting Endpoint
# Returns a canned welcome message and saves it to the messages table
# so it appears in conversation history. Only fires at the "welcome" stage.
# ===============================
GREETING_TEXT = (
    "Hi! I'm your CPL Interview Assistant. I'll guide you through a structured "
    "interview to help document your prior learning for academic credit. "
    "Let's start — what's your name?"
)

@app.get("/api/chat/greeting")
def api_chat_greeting():
    session_id = (request.args.get("session_id") or "").strip()
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    stage = get_current_stage(session_id)
    if stage != "welcome":
        return jsonify({"greeting": ""})

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, "assistant", GREETING_TEXT),
        )
        conn.commit()
        conn.close()
    except Exception:
        app.logger.exception("Failed to save greeting message")

    return jsonify({"greeting": GREETING_TEXT})


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

        # Load structured data collected in prior turns to inform the system prompt
        collected = get_collected_data(session_id)

        # Build the full messages array: stage-specific system prompt + history + current turn
        system_prompt = get_system_prompt(current_stage, collected_data=collected)
        messages = [
            {"role": "system", "content": system_prompt},
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

        # Extract structured fields from this exchange (separate LLM call, best-effort)
        extract_fields(session_id, current_stage, user_message, answer)

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
