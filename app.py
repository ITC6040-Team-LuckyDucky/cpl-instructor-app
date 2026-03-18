import os
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory
from openai import AzureOpenAI

from db import get_db_connection, is_local_dev
from storage import upload_file, ensure_storage_ready


# Explicit template folder for Azure App Service reliability
app = Flask(__name__, template_folder="templates")


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

        conn.commit()
        conn.close()

        return jsonify({
            "status": "success",
            "message": "Database tables created successfully",
            "mode": "local (SQLite)" if is_local_dev() else "Azure SQL",
            "tables": ["sessions", "messages", "uploads", "summaries"],
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

    # Save record to DB
    try:
        # TODO (Phase 3): replace "default" with the real session_id sent by the frontend
        session_id = "default"

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
# Chat API Endpoint
# ===============================
@app.post("/api/chat")
def api_chat():
    try:
        data = request.get_json(silent=True) or {}
        user_message = (data.get("message") or "").strip()

        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        if not deployment:
            return jsonify({"error": "Missing AZURE_OPENAI_DEPLOYMENT"}), 500

        client, err = get_client()
        if err:
            return jsonify({"error": err}), 500

        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "You are a CPL (Credit for Prior Learning) advisor assistant. Your job is to interview students about their professional experience and help them articulate their skills for academic credit evaluation. Be friendly, ask clarifying follow-up questions, and help them identify relevant evidence of their learning. Start by asking what course or competency area they want to receive credit for."},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
        )

        answer = (response.choices[0].message.content or "").strip()

        return jsonify({"answer": answer})

    except Exception as e:
        # Log full traceback in Azure Log Stream
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
