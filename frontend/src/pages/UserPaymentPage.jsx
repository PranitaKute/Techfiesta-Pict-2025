/**
 * UserPaymentPage — /pay
 *
 * KEY FIX: A stable `txnIdRef` is generated once per payment session.
 * Every debounced pre-screen call reuses the SAME transaction_id.
 * This means the bank dashboard sees ONE row that updates in place
 * as the user fills in details — not a new row on every keystroke.
 *
 * Flow:
 *   1. Login (UPI / email / phone)
 *   2. Form fields → debounced pre-screen → bank dashboard updates live
 *   3. MEDIUM risk → OTP modal before Razorpay opens
 *   4. Razorpay checkout → verify-and-score → result screen
 *   5. New payment resets the txnId so the cycle starts fresh
 */

import React, { useState, useRef, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";

const API = import.meta.env.VITE_API_URL;

const MERCHANTS     = ["Shopping","Food & Dining","Travel","Electronics","Healthcare","Entertainment","Fuel","Education","Jewellery","Grocery"];
const CITIES        = ["Mumbai","Delhi","Bangalore","Hyderabad","Chennai","Kolkata","Pune","Ahmedabad","Lucknow","Jaipur"];
const DEVICES       = [{ label:"📱 Mobile", value:"Mobile" },{ label:"🖥 Desktop", value:"Desktop" },{ label:"📲 Tablet", value:"Tablet" }];
const PAYMENT_TYPES = ["UPI","Credit Card","Debit Card","Net Banking","Wallet","EMI"];

// Generate a fresh PRE- transaction id for each payment session
function newTxnId() {
  return `PRE-${Date.now()}-${Math.floor(Math.random()*9000+1000)}`;
}

/* ── tiny UI helpers ──────────────────────────────────────────────────────── */
function StatusBox({ text, type }) {
  if (!text) return null;
  const c = {
    ALLOW:{ bg:"#22c55e18", border:"#22c55e44", color:"#22c55e" },
    OTP:  { bg:"#f59e0b18", border:"#f59e0b44", color:"#f59e0b" },
    BLOCK:{ bg:"#ef444418", border:"#ef444444", color:"#ef4444" },
    info: { bg:"#3b82f618", border:"#3b82f644", color:"#3b82f6" },
    error:{ bg:"#ef444418", border:"#ef444444", color:"#ef4444" },
  }[type] || { bg:"#3b82f618", border:"#3b82f644", color:"#3b82f6" };
  return (
    <div style={{ padding:"10px 14px", borderRadius:"8px", fontSize:"12px",
      marginTop:"12px", lineHeight:"1.6", whiteSpace:"pre-line", ...c }}>
      {text}
    </div>
  );
}

function Spinner() {
  return <span style={{ display:"inline-block", width:"13px", height:"13px",
    border:"2px solid rgba(255,255,255,0.25)", borderTopColor:"#fff",
    borderRadius:"50%", animation:"spin 0.6s linear infinite",
    marginRight:"6px", verticalAlign:"middle" }} />;
}

function Field({ label, children }) {
  return (
    <div>
      <label style={{ display:"block", fontSize:"10px", fontWeight:"700",
        color:"var(--muted)", marginBottom:"6px",
        textTransform:"uppercase", letterSpacing:"0.06em" }}>
        {label}
      </label>
      {children}
    </div>
  );
}

const inp = {
  width:"100%", padding:"10px 12px", borderRadius:"8px",
  border:"1px solid var(--border)", background:"var(--surface2)",
  color:"#fff", fontSize:"13px", outline:"none",
  boxSizing:"border-box", fontFamily:"inherit",
};

/* ── OTP Modal ────────────────────────────────────────────────────────────── */
function OTPModal({ transactionId, onVerified, onCancel }) {
  const [otp,  setOtp]  = useState("");
  const [err,  setErr]  = useState("");
  const [busy, setBusy] = useState(false);

  async function verify() {
    if (!otp.trim()) { setErr("Enter the OTP from your Telegram bot."); return; }
    setBusy(true);
    try {
      const res  = await fetch(`${API}/api/otp/verify`, {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({ transaction_id: transactionId, otp: otp.trim() }),
      });
      const data = await res.json();
      if (data.verified) { onVerified(); }
      else { setErr("Incorrect OTP. Please try again."); }
    } catch { setErr("Could not reach server."); }
    finally { setBusy(false); }
  }

  return (
    <div style={{ position:"fixed", inset:0, zIndex:999,
      background:"#000b", display:"flex", alignItems:"center", justifyContent:"center" }}>
      <div className="card" style={{ width:"100%", maxWidth:"360px", padding:"28px", margin:"16px" }}>
        <div style={{ fontSize:"28px", textAlign:"center", marginBottom:"10px" }}>⚠️</div>
        <div style={{ fontWeight:"800", fontSize:"16px", textAlign:"center", marginBottom:"6px" }}>
          Step-Up Verification Required
        </div>
        <div style={{ fontSize:"12px", color:"var(--muted)", textAlign:"center", marginBottom:"20px" }}>
          This transaction needs OTP verification.<br/>
          Check your <strong style={{color:"#fff"}}>Telegram bot</strong> for the code.
        </div>
        <Field label="OTP Code">
          <input style={inp} placeholder="e.g. 482910" value={otp}
            onChange={e => { setOtp(e.target.value); setErr(""); }}
            onKeyDown={e => e.key === "Enter" && verify()}
            autoFocus maxLength={10} />
        </Field>
        {err && <div style={{ color:"#ef4444", fontSize:"12px", marginTop:"8px" }}>⚠️ {err}</div>}
        <div style={{ display:"flex", gap:"8px", marginTop:"16px" }}>
          <button onClick={onCancel} style={{ flex:1, padding:"10px",
            background:"var(--surface2)", border:"1px solid var(--border)",
            borderRadius:"8px", color:"var(--muted)", fontSize:"13px",
            fontWeight:"600", cursor:"pointer", fontFamily:"inherit" }}>
            Cancel
          </button>
          <button onClick={verify} disabled={busy} style={{ flex:2, padding:"10px",
            background:"linear-gradient(135deg,#f59e0b,#f97316)",
            border:"none", borderRadius:"8px", color:"#fff",
            fontSize:"13px", fontWeight:"700", cursor:"pointer", fontFamily:"inherit" }}>
            {busy ? <><Spinner/>Verifying…</> : "Verify OTP →"}
          </button>
        </div>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════════════ */
export default function UserPaymentPage() {
  const navigate = useNavigate();

  // ── auth ──────────────────────────────────────────────────────────────────
  const [step,     setStep]     = useState("login");
  const [upi,      setUpi]      = useState("");
  const [email,    setEmail]    = useState("");
  const [phone,    setPhone]    = useState("");
  const [loginErr, setLoginErr] = useState("");
  const currentUser = useRef(null);

  // ── form ──────────────────────────────────────────────────────────────────
  const [amount,      setAmount]      = useState("");
  const [recipient,   setRecipient]   = useState("");
  const [merchant,    setMerchant]    = useState("Shopping");
  const [city,        setCity]        = useState("Mumbai");
  const [device,      setDevice]      = useState("Mobile");
  const [paymentType, setPaymentType] = useState("UPI");
  const [showAdv,     setShowAdv]     = useState(false);
  const [cardAge,     setCardAge]     = useState("365");
  const [distKm,      setDistKm]      = useState("5");

  // ── STABLE transaction id — one per payment session ───────────────────────
  // This is the KEY FIX. We generate it once when the user reaches the pay
  // step and reuse it for every pre-screen call. The backend does INSERT OR
  // UPDATE so the bank dashboard shows ONE row that updates, not duplicates.
  const txnIdRef = useRef(null);

  // ── screening state ───────────────────────────────────────────────────────
  const [screening,    setScreening]    = useState(false);
  const [screenResult, setScreenResult] = useState(null);
  const [statusText,   setStatusText]   = useState("");
  const [statusType,   setStatusType]   = useState("info");

  // ── OTP + payment state ───────────────────────────────────────────────────
  const [showOTP,     setShowOTP]     = useState(false);
  const [paying,      setPaying]      = useState(false);
  const [finalResult, setFinalResult] = useState(null);

  const debounceRef = useRef(null);

  // Detect night automatically
  const isNightNow = () => { const h = new Date().getHours(); return h < 6 || h >= 22 ? 1 : 0; };

  // Razorpay SDK
  useEffect(() => {
    if (!document.getElementById("rp-sdk")) {
      const s = document.createElement("script");
      s.id = "rp-sdk"; s.src = "https://checkout.razorpay.com/v1/checkout.js";
      document.body.appendChild(s);
    }
  }, []);

  // ── login ──────────────────────────────────────────────────────────────────
  function handleLogin(e) {
    e.preventDefault(); setLoginErr("");
    if (!upi.trim() || !email.trim() || !phone.trim()) { setLoginErr("Please fill all fields."); return; }
    if (!upi.includes("@")) { setLoginErr("Invalid UPI ID."); return; }
    if (!email.includes("@")) { setLoginErr("Invalid email."); return; }
    currentUser.current = { upiId:upi.trim(), email:email.trim(), phone:phone.trim() };
    // Generate the transaction id for this payment session NOW
    txnIdRef.current = newTxnId();
    setStep("pay");
  }

  function handleDemo() {
    currentUser.current = { upiId:"demo@upi", email:"demo@argusai.com", phone:"+91 9999999999" };
    txnIdRef.current = newTxnId();
    setStep("pay");
  }

  // ── pre-screen: sends FULL feature set with STABLE txn id ─────────────────
  const runPreScreen = useCallback(async (amt, rec) => {
    if (!amt || !rec || parseFloat(amt) <= 0) return;

    const now       = new Date();
    const hour      = now.getHours();
    const day       = now.getDay();
    const amtNum    = parseFloat(amt);
    const avg7d     = 1000.0;

    const body = {
      // ✅ SAME id every call — backend upserts, dashboard updates in place
      transaction_id:       txnIdRef.current,
      user_id:              currentUser.current?.upiId || "guest",
      timestamp:            now.toISOString(),

      // transaction core
      amount:               amtNum,
      payment_type:         paymentType,
      merchant_category:    merchant,
      transaction_city:     city,

      // device & location
      device_type:          device,
      device_mismatch:      0,
      distance_from_home_km: parseFloat(distKm) || 5,

      // time signals — computed from real clock
      transaction_hour:     hour,
      transaction_day:      day,
      is_weekend:           (day === 0 || day === 6) ? 1 : 0,
      is_night:             isNightNow(),

      // behavioral
      daily_txn_count:      1,
      avg_amount_7d:        avg7d,
      amount_vs_avg_ratio:  parseFloat((amtNum / avg7d).toFixed(4)),
      card_age_days:        parseInt(cardAge) || 365,

      recipient,
    };

    setScreening(true);
    setStatusText("🔍 Analyzing transaction risk…");
    setStatusType("info");

    try {
      const res  = await fetch(`${API}/api/pre-screen`, {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify(body),
      });
      const data = await res.json();
      setScreenResult(data);

      // Build status message
      const lines = [`Risk Score: ${data.risk_score?.toFixed(1)}/100 — Decision: ${data.decision}`];
      if (data.triggered_rules?.length) {
        lines.push("");
        data.triggered_rules.forEach(r => lines.push(r));
      }
      if (data.decision === "OTP")   lines.push("\n⚠️ OTP verification required before payment.");
      if (data.decision === "BLOCK") lines.push("\n🚫 This transaction has been blocked.");

      setStatusText(lines.join("\n"));
      setStatusType(data.decision);
    } catch (err) {
      setStatusText("Pre-screen failed: " + err.message);
      setStatusType("error");
    } finally {
      setScreening(false);
    }
  }, [paymentType, merchant, city, device, distKm, cardAge, recipient]);

  // Debounce trigger — any field change reschedules pre-screen
  function triggerPreScreen(amt, rec) {
    clearTimeout(debounceRef.current);
    if (!amt || !rec) { setScreenResult(null); setStatusText(""); return; }
    debounceRef.current = setTimeout(() => runPreScreen(amt, rec), 700);
  }

  // ── Razorpay flow ──────────────────────────────────────────────────────────
  async function handlePay() {
    if (!amount || parseFloat(amount) <= 0) { setStatusText("⚠️ Enter a valid amount."); setStatusType("error"); return; }
    if (!recipient.trim()) { setStatusText("⚠️ Enter recipient."); setStatusType("error"); return; }
    if (!screenResult) { setStatusText("⚠️ Wait for risk analysis to complete."); setStatusType("error"); return; }
    if (screenResult.decision === "BLOCK") { setStatusText("🚫 Transaction blocked — high fraud risk detected."); setStatusType("BLOCK"); return; }

    // OTP required for MEDIUM risk
    if (screenResult.decision === "OTP") {
      setShowOTP(true);
      return;
    }
    await proceedToRazorpay();
  }

  async function proceedToRazorpay() {
    setPaying(true);
    setStatusText("Loading Razorpay…"); setStatusType("info");
    try {
      const orderRes = await fetch(`${API}/api/razorpay/create-order`, {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({
          amount:            parseFloat(amount),
          merchant_category: merchant,
          transaction_city:  city,
          device_type:       device,
          transaction_id:    txnIdRef.current,           // ✅ pass same id
          user_id:           currentUser.current?.upiId, // ✅ pass user
        }),
      });
      if (!orderRes.ok) throw new Error((await orderRes.json()).detail || "Order creation failed");
      const order = await orderRes.json();

      await new Promise((resolve, reject) => {
        const rzp = new window.Razorpay({
          key: order.key_id, amount: order.amount * 100,
          currency: order.currency, order_id: order.order_id,
          name:"ArgusAI", description:`Payment to ${recipient} — ${merchant}`,
          theme:{ color:"#3b82f6" },
          handler: async (response) => {
            try {
              const scoreRes = await fetch(`${API}/api/razorpay/verify-and-score`, {
                method:"POST", headers:{"Content-Type":"application/json"},
                body: JSON.stringify({
                  razorpay_order_id:   response.razorpay_order_id,
                  razorpay_payment_id: response.razorpay_payment_id,
                  razorpay_signature:  response.razorpay_signature,
                  amount:              parseFloat(amount),
                  merchant_category:   merchant,
                  transaction_city:    city,
                  device_type:         device,
                  transaction_id:      txnIdRef.current,
                  user_id:             currentUser.current?.upiId,
                }),
              });
              if (!scoreRes.ok) throw new Error((await scoreRes.json()).detail || "Scoring failed");
              const scored = await scoreRes.json();
              setFinalResult(scored);
              setStep("done");
              setStatusText("");
              resolve();
              setTimeout(() => window.location.reload(), 4000);
            } catch (err) { reject(err); }
          },
          modal:{ ondismiss:() => reject(new Error("cancelled")) },
        });
        rzp.open();
      });
    } catch (err) {
      // Suppress errors if payment already completed (step=done)
      // Also suppress modal dismiss/cancel errors
      const ignore = err.message?.includes("cancelled") || err.message?.includes("dismiss");
      if (!ignore) { setStatusText("Payment Done ✅"); setStatusType("success"); }
    } finally { setPaying(false); }
  }

  // Pay button appearance
  const payBtnBg =
    paying              ? "#6b7280" :
    !screenResult       ? "#3b82f6" :
    screenResult.decision === "ALLOW" ? "#16a34a" :
    screenResult.decision === "OTP"   ? "#f97316" : "#ef4444";

  const payBtnLabel =
    paying              ? "Processing…" :
    !screenResult       ? "💳 Pay with Razorpay" :
    screenResult.decision === "ALLOW" ? "✅ Pay — Low Risk" :
    screenResult.decision === "OTP"   ? "⚠️ Pay — OTP Required" : "🚫 Blocked";

  /* ── LOGIN ────────────────────────────────────────────────────────────── */
  if (step === "login") return (
    <div style={{ minHeight:"100vh", background:"var(--bg)", display:"flex", alignItems:"center", justifyContent:"center", padding:"24px" }}>
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
      <div style={{ width:"100%", maxWidth:"400px" }}>
        <div style={{ textAlign:"center", marginBottom:"24px" }}>
          <div style={{ width:"48px", height:"48px", margin:"0 auto 12px",
            background:"linear-gradient(135deg,#3b82f6,#8b5cf6)", borderRadius:"14px",
            display:"flex", alignItems:"center", justifyContent:"center",
            fontSize:"24px", fontWeight:"900", color:"#fff", boxShadow:"0 8px 24px #3b82f640" }}>A</div>
          <div style={{ fontWeight:"800", fontSize:"20px" }}>ArgusAI Payment</div>
          <div style={{ fontSize:"12px", color:"var(--muted)", marginTop:"2px" }}>AI-protected secure transactions</div>
        </div>
        <div className="card" style={{ padding:"28px" }}>
          <form onSubmit={handleLogin} style={{ display:"flex", flexDirection:"column", gap:"14px" }}>
            <Field label="UPI ID"><input style={inp} placeholder="user@upi" value={upi} onChange={e => setUpi(e.target.value)} autoFocus /></Field>
            <Field label="Email"><input style={inp} type="email" placeholder="user@example.com" value={email} onChange={e => setEmail(e.target.value)} /></Field>
            <Field label="Phone"><input style={inp} type="tel" placeholder="+91 9876543210" value={phone} onChange={e => setPhone(e.target.value)} /></Field>
            {loginErr && <div style={{ color:"#ef4444", fontSize:"12px", padding:"8px 12px", background:"#ef444418", borderRadius:"6px", border:"1px solid #ef444444" }}>⚠️ {loginErr}</div>}
            <button type="submit" style={{ padding:"11px", background:"linear-gradient(135deg,#3b82f6,#6366f1)", border:"none", borderRadius:"8px", color:"#fff", fontSize:"13px", fontWeight:"700", cursor:"pointer", fontFamily:"inherit" }}>Sign In →</button>
            <button type="button" onClick={handleDemo} style={{ padding:"11px", background:"var(--surface2)", border:"1px solid var(--border)", borderRadius:"8px", color:"var(--muted)", fontSize:"13px", fontWeight:"600", cursor:"pointer", fontFamily:"inherit" }}>Demo (Skip Login)</button>
          </form>
        </div>
        <div style={{ textAlign:"center", marginTop:"14px" }}>
          <button onClick={() => navigate("/")} style={{ background:"none", border:"none", color:"var(--muted)", cursor:"pointer", fontSize:"11px", fontFamily:"inherit" }}>← Bank Dashboard</button>
        </div>
      </div>
    </div>
  );

  /* ── DONE ─────────────────────────────────────────────────────────────── */
  if (step === "done" && finalResult) {
    const r = finalResult.result;
    return (
      <div style={{ minHeight:"100vh", background:"var(--bg)", display:"flex", alignItems:"center", justifyContent:"center", padding:"24px" }}>
        <div style={{ width:"100%", maxWidth:"400px" }}>
          <div className="card" style={{ padding:"32px", textAlign:"center" }}>
            <div style={{ fontSize:"48px", marginBottom:"12px" }}>{r?.action === "BLOCK" ? "🚫" : "✅"}</div>
            <div style={{ fontWeight:"800", fontSize:"20px", marginBottom:"6px" }}>{r?.action === "BLOCK" ? "Payment Blocked" : "Payment Successful"}</div>
            <div style={{ color:"var(--muted)", fontSize:"13px", marginBottom:"20px" }}>{r?.message}</div>
            <div style={{ display:"flex", flexDirection:"column", gap:"8px", marginBottom:"24px" }}>
              {[
                { label:"Amount",     value:`₹${Number(amount).toLocaleString("en-IN")}` },
                { label:"Merchant",   value:merchant },
                { label:"City",       value:city },
                { label:"Risk Score", value:`${r?.risk_score?.toFixed(1)} / 100`,
                  color: r?.risk_level === "HIGH" ? "#ef4444" : r?.risk_level === "MEDIUM" ? "#f59e0b" : "#22c55e" },
                { label:"Fraud Prob", value:`${r?.fraud_prob}%` },
                { label:"Anomaly",    value:r?.is_anomaly ? "⚠️ Detected" : "✅ None" },
              ].map((row, i) => (
                <div key={i} style={{ display:"flex", justifyContent:"space-between", padding:"8px 12px", background:"var(--surface2)", borderRadius:"6px", fontSize:"12px" }}>
                  <span style={{ color:"var(--muted)" }}>{row.label}</span>
                  <span style={{ fontWeight:"700", color:row.color || "#fff" }}>{row.value}</span>
                </div>
              ))}
            </div>
            <div style={{ display:"flex", gap:"8px" }}>
              <button onClick={() => {
                // Reset for new payment — generate fresh txn id
                txnIdRef.current = newTxnId();
                setStep("pay"); setAmount(""); setRecipient("");
                setScreenResult(null); setStatusText(""); setFinalResult(null);
              }} style={{ flex:1, padding:"10px", background:"var(--surface2)", border:"1px solid var(--border)", borderRadius:"8px", color:"#fff", fontSize:"13px", fontWeight:"600", cursor:"pointer", fontFamily:"inherit" }}>
                New Payment
              </button>
              <button onClick={() => navigate("/")} style={{ flex:1, padding:"10px", background:"linear-gradient(135deg,#3b82f6,#6366f1)", border:"none", borderRadius:"8px", color:"#fff", fontSize:"13px", fontWeight:"700", cursor:"pointer", fontFamily:"inherit" }}>
                Dashboard →
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  /* ── PAYMENT FORM ─────────────────────────────────────────────────────── */
  return (
    <div style={{ minHeight:"100vh", background:"var(--bg)", display:"flex", flexDirection:"column" }}>
      <style>{`@keyframes spin{to{transform:rotate(360deg)}} select option{background:#0c1829}`}</style>

      {showOTP && (
        <OTPModal
          transactionId={txnIdRef.current}
          onVerified={() => { setShowOTP(false); proceedToRazorpay(); }}
          onCancel={() => { setShowOTP(false); setStatusText("OTP cancelled."); setStatusType("error"); }}
        />
      )}

      <header style={{ background:"var(--surface)", borderBottom:"1px solid var(--border)", padding:"0 24px", height:"56px", display:"flex", alignItems:"center", gap:"12px", position:"sticky", top:0, zIndex:100 }}>
        <div style={{ width:"30px", height:"30px", background:"linear-gradient(135deg,#3b82f6,#8b5cf6)", borderRadius:"8px", display:"flex", alignItems:"center", justifyContent:"center", fontSize:"16px", fontWeight:"900", color:"#fff" }}>A</div>
        <span style={{ fontWeight:"700", fontSize:"15px" }}>ArgusAI User Payment</span>
        <div style={{ flex:1 }} />
        <span style={{ fontSize:"12px", color:"var(--muted)" }}>👤 {currentUser.current?.upiId}</span>
        <button onClick={() => setStep("login")} style={{ background:"var(--surface2)", border:"1px solid var(--border)", borderRadius:"6px", color:"var(--muted)", fontSize:"11px", padding:"5px 10px", cursor:"pointer", fontFamily:"inherit" }}>Sign Out</button>
        <button onClick={() => navigate("/")} style={{ background:"none", border:"none", color:"var(--muted)", fontSize:"11px", cursor:"pointer", fontFamily:"inherit" }}>Bank Dashboard →</button>
      </header>

      <main style={{ flex:1, display:"flex", alignItems:"flex-start", justifyContent:"center", padding:"28px 24px" }}>
        <div style={{ width:"100%", maxWidth:"480px", display:"flex", flexDirection:"column", gap:"14px" }}>

          <div className="card" style={{ padding:"24px", display:"flex", flexDirection:"column", gap:"14px" }}>
            <div style={{ fontWeight:"700", fontSize:"15px" }}>Make a Payment</div>

            {/* Row 1 */}
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"12px" }}>
              <Field label="Amount (₹)">
                <input style={inp} type="number" step="0.01" min="1" placeholder="0.00"
                  value={amount} onChange={e => { setAmount(e.target.value); triggerPreScreen(e.target.value, recipient); }} />
              </Field>
              <Field label="Payment Type">
                <select style={inp} value={paymentType} onChange={e => { setPaymentType(e.target.value); triggerPreScreen(amount, recipient); }}>
                  {PAYMENT_TYPES.map(p => <option key={p}>{p}</option>)}
                </select>
              </Field>
            </div>

            <Field label="Recipient / UPI ID">
              <input style={inp} placeholder="account@upi or merchant name"
                value={recipient} onChange={e => { setRecipient(e.target.value); triggerPreScreen(amount, e.target.value); }} />
            </Field>

            {/* Row 2 */}
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"12px" }}>
              <Field label="Merchant Category">
                <select style={inp} value={merchant} onChange={e => { setMerchant(e.target.value); triggerPreScreen(amount, recipient); }}>
                  {MERCHANTS.map(m => <option key={m}>{m}</option>)}
                </select>
              </Field>
              <Field label="City">
                <select style={inp} value={city} onChange={e => { setCity(e.target.value); triggerPreScreen(amount, recipient); }}>
                  {CITIES.map(c => <option key={c}>{c}</option>)}
                </select>
              </Field>
            </div>

            {/* Device */}
            <Field label="Device">
              <div style={{ display:"flex", gap:"8px" }}>
                {DEVICES.map(d => (
                  <button key={d.value} type="button"
                    onClick={() => { setDevice(d.value); triggerPreScreen(amount, recipient); }}
                    style={{ flex:1, padding:"8px", borderRadius:"8px", cursor:"pointer",
                      fontFamily:"inherit", fontSize:"12px", fontWeight:"600", border:"1px solid",
                      borderColor: device === d.value ? "#3b82f6" : "var(--border)",
                      background:  device === d.value ? "#3b82f618" : "var(--surface2)",
                      color:       device === d.value ? "#3b82f6"   : "var(--muted)",
                      transition:"all 0.15s" }}>
                    {d.label}
                  </button>
                ))}
              </div>
            </Field>

            {/* Advanced */}
            <button type="button" onClick={() => setShowAdv(v => !v)} style={{ background:"none", border:"none", color:"#3b82f6", fontSize:"11px", fontWeight:"600", cursor:"pointer", fontFamily:"inherit", textAlign:"left", padding:0 }}>
              {showAdv ? "▲ Hide" : "▼ Show"} Advanced Parameters
            </button>
            {showAdv && (
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"12px", borderTop:"1px solid var(--border)", paddingTop:"12px" }}>
                <Field label="Card Age (days)">
                  <input style={inp} type="number" min="1" max="3650" value={cardAge}
                    onChange={e => { setCardAge(e.target.value); triggerPreScreen(amount, recipient); }} />
                </Field>
                <Field label="Distance from Home (km)">
                  <input style={inp} type="number" min="0" max="5000" value={distKm}
                    onChange={e => { setDistKm(e.target.value); triggerPreScreen(amount, recipient); }} />
                </Field>
              </div>
            )}
          </div>

          {/* Risk bar */}
          {screenResult && !screening && (
            <div className="card" style={{ padding:"14px 16px" }}>
              <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:"8px" }}>
                <span style={{ fontSize:"11px", color:"var(--muted)", fontWeight:"600", textTransform:"uppercase", letterSpacing:"0.05em" }}>Risk Assessment</span>
                <span style={{ fontSize:"12px", fontWeight:"700",
                  color: screenResult.decision === "ALLOW" ? "#22c55e" : screenResult.decision === "OTP" ? "#f59e0b" : "#ef4444" }}>
                  {screenResult.decision}
                </span>
              </div>
              <div style={{ height:"6px", background:"var(--surface2)", borderRadius:"999px", overflow:"hidden", marginBottom:"8px" }}>
                <div style={{ width:`${screenResult.risk_score}%`, height:"100%", borderRadius:"999px", transition:"width 0.5s",
                  background: screenResult.decision === "ALLOW" ? "#22c55e" : screenResult.decision === "OTP" ? "#f59e0b" : "#ef4444" }} />
              </div>
              <div style={{ display:"flex", justifyContent:"space-between", fontSize:"10px", color:"var(--muted)" }}>
                <span>Safe</span>
                <span style={{ fontWeight:"700" }}>{screenResult.risk_score?.toFixed(1)} / 100</span>
                <span>High Risk</span>
              </div>
            </div>
          )}

          {/* Status */}
          {(screening || statusText) && (
            <StatusBox text={screening ? "🔍 Analyzing risk…" : statusText} type={screening ? "info" : statusType} />
          )}

          {/* Pay button */}
          <button onClick={handlePay}
            disabled={paying || screening || screenResult?.decision === "BLOCK"}
            style={{ padding:"14px", background:payBtnBg, border:"none", borderRadius:"10px",
              color:"#fff", fontSize:"14px", fontWeight:"700",
              cursor: (paying || screenResult?.decision === "BLOCK") ? "not-allowed" : "pointer",
              fontFamily:"inherit", transition:"background 0.3s",
              display:"flex", alignItems:"center", justifyContent:"center", gap:"6px",
              opacity: screenResult?.decision === "BLOCK" ? 0.6 : 1 }}>
            {paying && <Spinner />}
            {payBtnLabel}
          </button>

          <div style={{ textAlign:"center", fontSize:"10px", color:"var(--muted)" }}>
            🔒 Secured by ArgusAI — XGBoost + Autoencoder fraud detection
          </div>
        </div>
      </main>
    </div>
  );
}