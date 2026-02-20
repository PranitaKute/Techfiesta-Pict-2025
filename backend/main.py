"""
ArgusAI — FastAPI Backend
Endpoints:
  POST /api/transaction               → analyze single transaction
  POST /api/transaction/fraud         → inject a fraud transaction (demo)
  GET  /api/transactions              → recent transaction history
  GET  /api/stats                     → system statistics
  GET  /api/user/{id}                 → user risk profile
  POST /api/otp/verify                → verify OTP
  POST /api/razorpay/create-order     → create Razorpay order
  POST /api/razorpay/verify-and-score → verify payment + fraud score
  WS   /ws/stream                     → live transaction WebSocket feed
"""

import sys, os, asyncio, json, hmac, hashlib, random
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env BEFORE importing anything else
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import razorpay

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml.predict                  import predict_transaction
from backend.database            import (log_transaction, get_recent_transactions,
                                          get_stats, get_user_profile, get_risk_trend)
from backend.alert               import send_otp_alert, send_block_alert, verify_otp
from backend.transaction_stream  import generate_live_transaction

# ─── Razorpay Client ──────────────────────────────────────────────────────────
razorpay_client = razorpay.Client(auth=(
    os.getenv("RAZORPAY_KEY_ID"),
    os.getenv("RAZORPAY_KEY_SECRET"),
))

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "ArgusAI Fraud Detection API",
    description = "Real-time AI-powered fraud detection & risk management",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ─── WebSocket Manager ────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager        = ConnectionManager()
_streaming     = False
_stream_task   = None


# ─── Core transaction processor ───────────────────────────────────────────────
async def process_and_broadcast(txn: dict):
    result = predict_transaction(txn)
    log_transaction(txn, result)

    # Send alerts
    if result["action"] == "BLOCK":
        asyncio.create_task(send_block_alert(txn, result))
    elif result["action"] == "OTP":
        otp_data = await send_otp_alert(txn, result)
        result["otp_sent"] = otp_data.get("sent", False)
        result["otp"]      = otp_data.get("otp", "")

    payload = {**txn, **result, "processed_at": datetime.utcnow().isoformat()}
    await manager.broadcast(payload)
    return result


# ─── Background stream ────────────────────────────────────────────────────────
async def _auto_stream(interval: float):
    global _streaming
    while _streaming:
        txn = generate_live_transaction()
        await process_and_broadcast(txn)
        await asyncio.sleep(interval)


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────
class TransactionInput(BaseModel):
    transaction_id:        Optional[str]   = None
    user_id:               Optional[int]   = 1
    timestamp:             Optional[str]   = None
    amount:                float
    payment_type:          str
    merchant_category:     str
    transaction_city:      str
    distance_from_home_km: float           = 0.0
    device_type:           str             = "Mobile"
    device_mismatch:       int             = 0
    card_age_days:         int             = 365
    transaction_hour:      int             = 12
    transaction_day:       int             = 0
    is_weekend:            int             = 0
    is_night:              int             = 0
    daily_txn_count:       int             = 1
    avg_amount_7d:         float           = 1000.0
    amount_vs_avg_ratio:   float           = 1.0

class OTPVerify(BaseModel):
    transaction_id: str
    otp:            str

class StreamControl(BaseModel):
    action:   str    = "start"   # "start" | "stop"
    interval: float  = 3.0

class RazorpayOrderRequest(BaseModel):
    amount:            float
    merchant_category: str = "Shopping"
    transaction_city:  str = "Mumbai"
    device_type:       str = "Mobile"

class RazorpayVerifyRequest(BaseModel):
    razorpay_order_id:   str
    razorpay_payment_id: str
    razorpay_signature:  str
    amount:              float
    merchant_category:   str = "Shopping"
    transaction_city:    str = "Mumbai"
    device_type:         str = "Mobile"


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service": "ArgusAI Fraud Detection",
        "version": "1.0.0",
        "status":  "operational",
        "docs":    "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/api/transaction")
async def analyze_transaction(txn_input: TransactionInput):
    """Analyze a single transaction and return risk assessment."""
    txn = txn_input.dict()
    if not txn.get("transaction_id"):
        txn["transaction_id"] = f"TXN{int(datetime.utcnow().timestamp())}"
    if not txn.get("timestamp"):
        txn["timestamp"] = datetime.utcnow().isoformat()

    result = await process_and_broadcast(txn)
    return {"transaction": txn, "result": result}


@app.post("/api/transaction/fraud")
async def inject_fraud_transaction():
    """Inject a simulated fraud transaction (for live demo)."""
    txn = generate_live_transaction(force_fraud=True)
    result = await process_and_broadcast(txn)
    return {"transaction": txn, "result": result, "demo": True}


@app.post("/api/transaction/simulate")
async def simulate_transaction():
    """Generate and analyze one random transaction."""
    txn = generate_live_transaction()
    result = await process_and_broadcast(txn)
    return {"transaction": txn, "result": result}


@app.post("/api/transaction/user-initiated")
async def user_initiated_transaction(txn_input: TransactionInput):
    """User-initiated transaction (from demo form)."""
    txn = txn_input.dict()

    if not txn.get("transaction_id"):
        txn["transaction_id"] = f"TXN{random.randint(100000, 999999)}"
    if not txn.get("timestamp"):
        txn["timestamp"] = datetime.utcnow().isoformat()
    if not txn.get("user_id"):
        txn["user_id"] = random.randint(1000, 9999)
    if not txn.get("distance_from_home_km"):
        txn["distance_from_home_km"] = random.randint(0, 50)
    if txn.get("device_mismatch") is None:
        txn["device_mismatch"] = random.randint(0, 1)
    if not txn.get("card_age_days"):
        txn["card_age_days"] = random.randint(30, 1825)
    if txn.get("transaction_hour") is None:
        txn["transaction_hour"] = random.randint(0, 23)
    if txn.get("transaction_day") is None:
        txn["transaction_day"] = random.randint(0, 6)
    if txn.get("is_weekend") is None:
        txn["is_weekend"] = 1 if txn.get("transaction_day", 0) >= 5 else 0
    if txn.get("is_night") is None:
        hour = txn.get("transaction_hour", 12)
        txn["is_night"] = 1 if hour in range(20, 24) or hour in range(0, 6) else 0
    if txn.get("daily_txn_count") is None:
        txn["daily_txn_count"] = random.randint(1, 10)
    if not txn.get("avg_amount_7d"):
        txn["avg_amount_7d"] = txn.get("amount", 1000)
    if not txn.get("amount_vs_avg_ratio"):
        txn["amount_vs_avg_ratio"] = round(txn.get("amount", 1000) / max(txn.get("avg_amount_7d", 1), 1), 2)

    result = await process_and_broadcast(txn)
    return {"transaction": txn, "result": result, "source": "user"}


@app.get("/api/transactions")
async def recent_transactions(limit: int = 50):
    return {"transactions": get_recent_transactions(limit)}


@app.get("/api/stats")
async def system_stats():
    stats = get_stats()
    trend = get_risk_trend(hours=24)
    return {"stats": stats, "trend": trend}


@app.get("/api/user/{user_id}")
async def user_profile(user_id: int):
    profile = get_user_profile(user_id)
    return {"user_id": user_id, "profile": profile}


@app.post("/api/otp/verify")
async def verify_otp_endpoint(body: OTPVerify):
    ok = verify_otp(body.transaction_id, body.otp)
    return {
        "verified": ok,
        "message":  "Transaction approved ✅" if ok
                    else "Invalid OTP ❌ Transaction blocked",
    }


@app.post("/api/stream/control")
async def control_stream(body: StreamControl):
    global _streaming, _stream_task
    if body.action == "start" and not _streaming:
        _streaming   = True
        _stream_task = asyncio.create_task(_auto_stream(body.interval))
        return {"status": "stream started", "interval": body.interval}
    elif body.action == "stop":
        _streaming = False
        if _stream_task:
            _stream_task.cancel()
        return {"status": "stream stopped"}
    return {"status": "no change"}


# ─── Razorpay Routes ──────────────────────────────────────────────────────────
@app.post("/api/razorpay/create-order")
async def razorpay_create_order(body: RazorpayOrderRequest):
    """Creates a Razorpay order. Frontend uses order_id to open Checkout popup."""
    try:
        order = razorpay_client.order.create({
            "amount":          int(body.amount * 100),  # paise
            "currency":        "INR",
            "payment_capture": 1,
            "notes": {
                "merchant_category": body.merchant_category,
                "transaction_city":  body.transaction_city,
                "device_type":       body.device_type,
            }
        })
        return {
            "order_id": order["id"],
            "amount":   body.amount,
            "currency": "INR",
            "key_id":   os.getenv("RAZORPAY_KEY_ID"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Razorpay order creation failed: {e}")


@app.post("/api/razorpay/verify-and-score")
async def razorpay_verify_and_score(body: RazorpayVerifyRequest):
    """Verifies Razorpay signature, fetches payment, enriches + fraud-scores it."""

    # ── 1. Signature verification ─────────────────────────────────────────────
    secret   = os.getenv("RAZORPAY_KEY_SECRET", "")
    expected = hmac.new(
        secret.encode(),
        f"{body.razorpay_order_id}|{body.razorpay_payment_id}".encode(),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected, body.razorpay_signature):
        raise HTTPException(status_code=400, detail="Invalid payment signature — possible tampering detected")

    # ── 2. Fetch payment details from Razorpay ────────────────────────────────
    try:
        payment = razorpay_client.payment.fetch(body.razorpay_payment_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not fetch payment from Razorpay: {e}")

    # ── 3. Enrich with fraud features ─────────────────────────────────────────
    created_at  = datetime.fromtimestamp(payment["created_at"])
    txn_hour    = created_at.hour
    txn_day     = created_at.weekday()
    is_weekend  = 1 if txn_day >= 5 else 0
    is_night    = 1 if txn_hour < 6 or txn_hour >= 22 else 0
    amount      = payment["amount"] / 100

    method_map = {
        "card": "Credit Card", "upi": "UPI",
        "netbanking": "Net Banking", "wallet": "Wallet", "emi": "EMI",
    }
    payment_type = method_map.get(payment.get("method", ""), "UPI")

    distance_from_home = round(random.uniform(0.5, 35.0), 2)
    avg_amount_7d      = round(random.uniform(500, 15000), 2)
    amount_vs_avg      = round(amount / max(avg_amount_7d, 1), 4)
    daily_txn_count    = random.randint(1, 12)
    device_mismatch    = random.choices([0, 1], weights=[85, 15])[0]
    card_age_days      = random.randint(30, 2000)

    txn = {
        "transaction_id":        f"RPY-{body.razorpay_payment_id[-8:].upper()}",
        "user_id":               random.randint(1000, 9999),
        "timestamp":             created_at.isoformat(),
        "amount":                amount,
        "payment_type":          payment_type,
        "merchant_category":     body.merchant_category,
        "transaction_city":      body.transaction_city,
        "distance_from_home_km": distance_from_home,
        "device_type":           body.device_type,
        "device_mismatch":       device_mismatch,
        "card_age_days":         card_age_days,
        "transaction_hour":      txn_hour,
        "transaction_day":       txn_day,
        "is_weekend":            is_weekend,
        "is_night":              is_night,
        "daily_txn_count":       daily_txn_count,
        "avg_amount_7d":         avg_amount_7d,
        "amount_vs_avg_ratio":   amount_vs_avg,
    }

    # ── 4. Run through existing fraud pipeline ────────────────────────────────
    result = await process_and_broadcast(txn)

    return {
        "transaction": txn,
        "result":      result,
        "razorpay": {
            "payment_id": body.razorpay_payment_id,
            "order_id":   body.razorpay_order_id,
            "method":     payment.get("method"),
            "status":     payment.get("status"),
        },
        "source": "razorpay",
    }


# ─── WebSocket endpoint ───────────────────────────────────────────────────────
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            msg  = json.loads(data)
            if msg.get("action") == "ping":
                await websocket.send_json({"action": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ─── Startup / Shutdown ───────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    print("🚀 ArgusAI API starting up...")
    global _streaming, _stream_task
    _streaming   = True
    _stream_task = asyncio.create_task(_auto_stream(3.0))
    print("✅ Live transaction stream started (every 3s)")


@app.on_event("shutdown")
async def shutdown():
    global _streaming
    _streaming = False
    print("👋 ArgusAI shutting down")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)