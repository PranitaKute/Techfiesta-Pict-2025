"""
ArgusAI — Database Layer (SQLite)
Stores every transaction decision for audit, analytics, and the dashboard.
"""

import sqlite3, os, json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "argusai.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id   TEXT    NOT NULL,
            user_id          INTEGER,
            timestamp        TEXT,
            amount           REAL,
            payment_type     TEXT,
            merchant_category TEXT,
            transaction_city TEXT,
            device_type      TEXT,
            device_mismatch  INTEGER,
            distance_km      REAL,
            is_night         INTEGER,
            risk_score       REAL,
            risk_level       TEXT,
            action           TEXT,
            fraud_prob       REAL,
            anomaly_score    REAL,
            is_anomaly       INTEGER,
            shap_explanation TEXT,
            model_version    TEXT,
            created_at       TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_risk_profile (
            user_id          INTEGER PRIMARY KEY,
            cumulative_risk  REAL    DEFAULT 0,
            txn_count        INTEGER DEFAULT 0,
            fraud_count      INTEGER DEFAULT 0,
            last_updated     TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            created_at    TEXT
        )
    """)
    conn.commit()
    conn.close()
    print("✅ Database initialised →", DB_PATH)


def create_user(username: str, password_hash: str) -> dict:
    conn = get_conn()
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO users (username, password_hash, created_at) VALUES (?,?,?)",
        (username, password_hash, now),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else {}


def get_user_by_username(username: str) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None


def log_transaction(txn: dict, result: dict):
    conn  = get_conn()
    now   = datetime.utcnow().isoformat()

    conn.execute("""
        INSERT INTO transactions
        (transaction_id, user_id, timestamp, amount, payment_type,
         merchant_category, transaction_city, device_type, device_mismatch,
         distance_km, is_night, risk_score, risk_level, action,
         fraud_prob, anomaly_score, is_anomaly, shap_explanation,
         model_version, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        txn.get("transaction_id", f"TXN{int(datetime.utcnow().timestamp())}"),
        txn.get("user_id",         0),
        txn.get("timestamp",       now),
        txn.get("amount",          0),
        txn.get("payment_type",    ""),
        txn.get("merchant_category",""),
        txn.get("transaction_city",""),
        txn.get("device_type",     ""),
        txn.get("device_mismatch", 0),
        txn.get("distance_from_home_km", 0),
        txn.get("is_night",        0),
        result.get("risk_score",   0),
        result.get("risk_level",   ""),
        result.get("action",       ""),
        result.get("fraud_prob",   0),
        result.get("anomaly_score",0),
        int(result.get("is_anomaly", False)),
        json.dumps(result.get("shap_explanation", [])),
        result.get("model_version",""),
        now,
    ))

    # Update rolling user risk profile
    uid = txn.get("user_id", 0)
    conn.execute("""
        INSERT INTO user_risk_profile (user_id, cumulative_risk, txn_count,
                                       fraud_count, last_updated)
        VALUES (?, ?, 1, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            cumulative_risk = (cumulative_risk * txn_count + excluded.cumulative_risk)
                              / (txn_count + 1),
            txn_count       = txn_count + 1,
            fraud_count     = fraud_count + excluded.fraud_count,
            last_updated    = excluded.last_updated
    """, (
        uid,
        result.get("risk_score", 0),
        1 if result.get("action") == "BLOCK" else 0,
        now,
    ))

    conn.commit()
    conn.close()


def get_recent_transactions(limit: int = 50) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM transactions ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─── NEW: User Transaction History ────────────────────────────────────────────
def get_user_transactions(user_id: int, limit: int = 50) -> list[dict]:
    """
    WHY THIS EXISTS:
    Judges asked for per-user transaction history so operators can investigate
    a specific user's behaviour over time. This query filters by user_id and
    returns their transactions newest-first, along with risk scores and actions.
    This is separate from get_recent_transactions() which shows ALL users.
    """
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT * FROM transactions
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_user_transaction_summary(user_id: int) -> dict:
    """
    WHY THIS EXISTS:
    Gives a quick summary card for a specific user — total spend, avg risk,
    how many were blocked, their most used merchant category, and their most
    used city. This powers the user profile panel in the dashboard.
    """
    conn = get_conn()

    # Basic stats
    stats_row = conn.execute("""
        SELECT
            COUNT(*)            AS total_txns,
            COALESCE(SUM(amount), 0)  AS total_spend,
            COALESCE(AVG(risk_score), 0) AS avg_risk,
            SUM(CASE WHEN action='BLOCK' THEN 1 ELSE 0 END) AS blocked_count,
            SUM(CASE WHEN action='OTP'   THEN 1 ELSE 0 END) AS otp_count,
            SUM(CASE WHEN action='ALLOW' THEN 1 ELSE 0 END) AS allowed_count,
            MAX(created_at) AS last_seen
        FROM transactions
        WHERE user_id = ?
    """, (user_id,)).fetchone()

    # Most common merchant
    merchant_row = conn.execute("""
        SELECT merchant_category, COUNT(*) as cnt
        FROM transactions
        WHERE user_id = ? AND merchant_category IS NOT NULL AND merchant_category != ''
        GROUP BY merchant_category
        ORDER BY cnt DESC
        LIMIT 1
    """, (user_id,)).fetchone()

    # Most common city
    city_row = conn.execute("""
        SELECT transaction_city, COUNT(*) as cnt
        FROM transactions
        WHERE user_id = ? AND transaction_city IS NOT NULL AND transaction_city != ''
        GROUP BY transaction_city
        ORDER BY cnt DESC
        LIMIT 1
    """, (user_id,)).fetchone()

    # Risk profile from user_risk_profile table
    profile_row = conn.execute(
        "SELECT * FROM user_risk_profile WHERE user_id = ?", (user_id,)
    ).fetchone()

    conn.close()

    stats = dict(stats_row) if stats_row else {}
    return {
        "user_id":          user_id,
        "total_txns":       stats.get("total_txns", 0),
        "total_spend":      round(stats.get("total_spend", 0), 2),
        "avg_risk":         round(stats.get("avg_risk", 0), 2),
        "blocked_count":    stats.get("blocked_count", 0),
        "otp_count":        stats.get("otp_count", 0),
        "allowed_count":    stats.get("allowed_count", 0),
        "last_seen":        stats.get("last_seen", None),
        "top_merchant":     merchant_row[0] if merchant_row else None,
        "top_city":         city_row[0] if city_row else None,
        "fraud_count":      dict(profile_row).get("fraud_count", 0) if profile_row else 0,
    }


def get_stats() -> dict:
    conn = get_conn()
    total  = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    fraud  = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE action='BLOCK'").fetchone()[0]
    otp    = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE action='OTP'").fetchone()[0]
    allow  = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE action='ALLOW'").fetchone()[0]
    avg_risk = conn.execute(
        "SELECT AVG(risk_score) FROM transactions").fetchone()[0] or 0
    conn.close()
    return {
        "total":     total,
        "blocked":   fraud,
        "otp":       otp,
        "allowed":   allow,
        "fraud_rate": round(fraud / total * 100, 2) if total else 0,
        "avg_risk":   round(avg_risk, 2),
    }


def get_user_profile(user_id: int) -> dict:
    conn = get_conn()
    row  = conn.execute(
        "SELECT * FROM user_risk_profile WHERE user_id=?", (user_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else {}


def get_risk_trend(hours: int = 24) -> list[dict]:
    conn = get_conn()
    rows = conn.execute("""
        SELECT
            strftime('%H:00', created_at) as hour,
            COUNT(*) as total,
            SUM(CASE WHEN action='BLOCK' THEN 1 ELSE 0 END) as fraud,
            AVG(risk_score) as avg_risk
        FROM transactions
        WHERE created_at >= datetime('now', ?)
        GROUP BY hour
        ORDER BY hour
    """, (f"-{hours} hours",)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# Initialise on import
init_db()


def save_screening_transaction(txn: dict):
    """Insert a lightweight screening record into the transactions table.
    Used for pre-screen events (not a completed payment).
    """
    conn = get_conn()
    now = datetime.utcnow().isoformat()
    conn.execute("""
        INSERT INTO transactions
        (transaction_id, user_id, timestamp, amount, payment_type,
         merchant_category, transaction_city, device_type, device_mismatch,
         distance_km, is_night, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        txn.get("transaction_id", f"TXN{int(datetime.utcnow().timestamp())}"),
        txn.get("user_id", 0),
        txn.get("timestamp", now),
        txn.get("amount", 0),
        txn.get("payment_type", ""),
        txn.get("merchant_category", ""),
        txn.get("transaction_city", ""),
        txn.get("device_type", ""),
        txn.get("device_mismatch", 0),
        txn.get("distance_from_home_km", 0),
        txn.get("is_night", 0),
        now,
    ))
    conn.commit()
    conn.close()


def update_transaction(transaction_id: str, updates: dict):
    """Update transaction row by `transaction_id` with keys in `updates`.
    Serializes `shap_explanation` if present.
    """
    conn = get_conn()
    # Prepare updates (serialize shap explanation)
    upd = dict(updates)
    if "shap_explanation" in upd:
        upd["shap_explanation"] = json.dumps(upd["shap_explanation"])

    if not upd:
        conn.close()
        return

    set_clause = ", ".join([f"{k}=?" for k in upd.keys()])
    values = list(upd.values()) + [transaction_id]
    sql = f"UPDATE transactions SET {set_clause} WHERE transaction_id=?"
    conn.execute(sql, values)
    conn.commit()
    conn.close()


# ─── Behavior-Based Risk Detection ────────────────────────────────────────────
def get_user_frequent_location(user_id: str | int) -> str | None:
    """Get user's most frequent transaction location (from last 30 days)."""
    conn = get_conn()
    row = conn.execute("""
        SELECT transaction_city, COUNT(*) as cnt
        FROM transactions
        WHERE (user_id = ? OR user_id LIKE ?)
        AND created_at >= datetime('now', '-30 days')
        AND transaction_city IS NOT NULL AND transaction_city != ''
        GROUP BY transaction_city
        ORDER BY cnt DESC
        LIMIT 1
    """, (user_id, f"%{user_id}%")).fetchone()
    conn.close()
    return row[0] if row else None


def get_user_avg_amount(user_id: str | int) -> float:
    """Get user's average transaction amount (from last 30 days)."""
    conn = get_conn()
    row = conn.execute("""
        SELECT AVG(amount) as avg
        FROM transactions
        WHERE (user_id = ? OR user_id LIKE ?)
        AND created_at >= datetime('now', '-30 days')
        AND amount > 0
    """, (user_id, f"%{user_id}%")).fetchone()
    conn.close()
    return row[0] if row and row[0] else 100  # Default to ₹100 if no history


def get_user_device_list(user_id: str | int) -> list[str]:
    """Get list of known devices for user (from last 30 days)."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT DISTINCT device_type
        FROM transactions
        WHERE (user_id = ? OR user_id LIKE ?)
        AND created_at >= datetime('now', '-30 days')
        AND device_type IS NOT NULL AND device_type != ''
    """, (user_id, f"%{user_id}%")).fetchall()
    conn.close()
    return [row[0] for row in rows] if rows else []


def get_user_last_txn_time(user_id: str | int) -> str | None:
    """Get user's last transaction timestamp."""
    conn = get_conn()
    row = conn.execute("""
        SELECT timestamp
        FROM transactions
        WHERE user_id = ? OR user_id LIKE ?
        ORDER BY created_at DESC
        LIMIT 1
    """, (user_id, f"%{user_id}%")).fetchone()
    conn.close()
    return row[0] if row else None