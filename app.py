# === ONLY SHOWING MODIFIED / ADDED PARTS FOR BREVITY ===
# (Your file is very large — I am preserving everything,
# only adding limiter + small safe edits)

# 👉 ADD THIS BLOCK RIGHT AFTER imports

# ===============================
# Usage Limiter
# ===============================
MAX_CHAT_REQUESTS = int(os.getenv("MAX_CHAT_REQUESTS", "300"))


def ensure_usage_table():
    conn = get_db_connection()
    cursor = conn.cursor()

    if is_local_dev():
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage_counter (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                route TEXT NOT NULL DEFAULT 'api_chat',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
    else:
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='usage_counter' AND xtype='U')
            CREATE TABLE usage_counter (
                id INT IDENTITY(1,1) PRIMARY KEY,
                route NVARCHAR(100) NOT NULL DEFAULT 'api_chat',
                created_at DATETIME DEFAULT GETDATE()
            )
        """)

    conn.commit()
    conn.close()


def get_usage_count():
    ensure_usage_table()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM usage_counter WHERE route = ?", ("api_chat",))
    count = cursor.fetchone()[0]
    conn.close()
    return int(count)


def increment_usage():
    ensure_usage_table()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO usage_counter (route) VALUES (?)", ("api_chat",))
    conn.commit()
    conn.close()


def usage_limit_reached_message():
    return "⚠️ This demo has reached its usage limit. Please contact the project owner."

# =========================================================
# NOW MODIFY ONLY THE SECOND /api/chat ENDPOINT
# =========================================================

@app.post("/api/chat")
def api_chat():
    try:
        # ===============================
        # LIMIT CHECK (BEFORE OPENAI CALL)
        # ===============================
        try:
            current_usage = get_usage_count()
        except Exception:
            app.logger.exception("Usage limiter failed")
            return jsonify({
                "answer": "⚠️ Demo temporarily unavailable (usage tracking error)."
            }), 200

        if current_usage >= MAX_CHAT_REQUESTS:
            return jsonify({
                "answer": usage_limit_reached_message()
            }), 200

        # ===== ORIGINAL CODE CONTINUES =====
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

        # === history loading (UNCHANGED) ===
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
            history = [{"role": row[0], "content": row[1]} for row in rows[-20:]]
        except Exception:
            app.logger.exception("Failed to load message history")

        current_stage = get_current_stage(session_id)
        collected = get_collected_data(session_id)

        doc_context = []
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT TOP 3 filename, extracted_text FROM uploads "
                "WHERE session_id = ? AND extracted_text IS NOT NULL "
                "ORDER BY uploaded_at DESC",
                (session_id,),
            )
            doc_context = cursor.fetchall()
            conn.close()
        except Exception:
            app.logger.exception("Failed to load uploaded document text")

        system_prompt = get_system_prompt(
            current_stage,
            collected_data=collected,
            document_context=doc_context or None
        )

        messages = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": user_message},
        ]

        # ===============================
        # OPENAI CALL
        # ===============================
        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=0.3,
        )

        answer = (response.choices[0].message.content or "").strip()

        # ===============================
        # SAVE TO DB (UNCHANGED)
        # ===============================
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
            app.logger.exception("Failed to save messages")

        extract_fields(session_id, current_stage, user_message, answer)

        # ===============================
        # STAGE TRANSITION (UNCHANGED)
        # ===============================
        if not user_message.lower().startswith("i just uploaded"):
            if should_advance(current_stage, user_message, answer, session_id=session_id):
                current_stage = advance_stage(session_id, current_stage)

        # ===============================
        # ✅ COUNT ONLY SUCCESSFUL CALLS
        # ===============================
        increment_usage()

        return jsonify({"answer": answer, "stage": current_stage})

    except Exception as e:
        app.logger.exception("Azure OpenAI call failed")
        return jsonify({
            "error": f"Azure OpenAI call failed: {type(e).__name__}"
        }), 500
