/**
 * RazorpayTab.jsx
 * Drop into: frontend/src/components/RazorpayTab.jsx
 *
 * Usage in App.jsx:
 *   import RazorpayTab from "./components/RazorpayTab";
 *   // Add a tab button + render <RazorpayTab /> in your tab switcher
 */

import { useState, useEffect, useRef } from "react";

const API = "http://localhost:8000";

// Razorpay Checkout.js is loaded dynamically once
function loadRazorpayScript() {
  return new Promise((resolve) => {
    if (document.getElementById("razorpay-checkout-js")) {
      resolve(true);
      return;
    }
    const script = document.createElement("script");
    script.id = "razorpay-checkout-js";
    script.src = "https://checkout.razorpay.com/v1/checkout.js";
    script.onload = () => resolve(true);
    script.onerror = () => resolve(false);
    document.body.appendChild(script);
  });
}

const CATEGORIES = [
  "Shopping", "Food & Dining", "Travel", "Electronics",
  "Healthcare", "Entertainment", "Fuel", "Education",
];
const CITIES = [
  "Mumbai", "Delhi", "Bangalore", "Hyderabad",
  "Chennai", "Kolkata", "Pune", "Ahmedabad",
];
const DEVICES = ["Mobile", "Desktop", "Tablet"];

const RISK_COLOR = {
  LOW:    { bg: "rgba(34,197,94,0.15)",  border: "#22c55e", text: "#4ade80" },
  MEDIUM: { bg: "rgba(234,179,8,0.15)",  border: "#eab308", text: "#facc15" },
  HIGH:   { bg: "rgba(249,115,22,0.15)", border: "#f97316", text: "#fb923c" },
  BLOCK:  { bg: "rgba(239,68,68,0.15)",  border: "#ef4444", text: "#f87171" },
};

const ACTION_BADGE = {
  ALLOW: { bg: "#166534", color: "#4ade80", label: "✓ ALLOWED"  },
  OTP:   { bg: "#713f12", color: "#fbbf24", label: "⚠ OTP SENT" },
  BLOCK: { bg: "#7f1d1d", color: "#f87171", label: "✕ BLOCKED"  },
};

export default function RazorpayTab() {
  const [form, setForm] = useState({
    amount: "",
    merchant_category: "Shopping",
    transaction_city: "Mumbai",
    device_type: "Mobile",
  });
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState("");
  const [payments, setPayments] = useState([]);
  const [selected, setSelected] = useState(null);
  const listRef = useRef(null);

  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  async function handlePay() {
    setError("");
    const amt = parseFloat(form.amount);
    if (!amt || amt <= 0) { setError("Enter a valid amount (₹ > 0)"); return; }
    if (amt < 1)          { setError("Minimum amount is ₹1"); return; }

    setLoading(true);
    try {
      // 1. Load Razorpay SDK
      const ok = await loadRazorpayScript();
      if (!ok) throw new Error("Could not load Razorpay Checkout. Check your internet.");

      // 2. Create order on your backend
      const orderRes = await fetch(`${API}/api/razorpay/create-order`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          amount:            amt,
          merchant_category: form.merchant_category,
          transaction_city:  form.transaction_city,
          device_type:       form.device_type,
        }),
      });
      if (!orderRes.ok) {
        const err = await orderRes.json();
        throw new Error(err.detail || "Order creation failed");
      }
      const order = await orderRes.json();

      // 3. Open Razorpay Checkout popup
      await new Promise((resolve, reject) => {
        const rzp = new window.Razorpay({
          key:         order.key_id,
          amount:      order.amount * 100,
          currency:    order.currency,
          order_id:    order.order_id,
          name:        "ArgusAI Demo",
          description: `${form.merchant_category} — ${form.transaction_city}`,
          theme:       { color: "#6366f1" },

          handler: async (response) => {
            // 4. Verify + fraud-score on your backend
            try {
              const scoreRes = await fetch(`${API}/api/razorpay/verify-and-score`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  razorpay_order_id:   response.razorpay_order_id,
                  razorpay_payment_id: response.razorpay_payment_id,
                  razorpay_signature:  response.razorpay_signature,
                  amount:              amt,
                  merchant_category:   form.merchant_category,
                  transaction_city:    form.transaction_city,
                  device_type:         form.device_type,
                }),
              });
              if (!scoreRes.ok) {
                const err = await scoreRes.json();
                throw new Error(err.detail || "Scoring failed");
              }
              const scored = await scoreRes.json();
              setPayments((prev) => [scored, ...prev]);
              setSelected(scored);
              setForm((f) => ({ ...f, amount: "" }));
              resolve();
            } catch (e) {
              reject(e);
            }
          },

          modal: {
            ondismiss: () => reject(new Error("Payment cancelled")),
          },
        });
        rzp.open();
      });

    } catch (e) {
      // "Payment cancelled" is not an error worth alarming the user
      if (!e.message?.includes("cancelled")) setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  // Auto-scroll list when new payment arrives
  useEffect(() => {
    if (listRef.current) listRef.current.scrollTop = 0;
  }, [payments.length]);

  const risk  = selected?.result?.risk_level  || "LOW";
  const action = selected?.result?.action      || "ALLOW";
  const rc    = RISK_COLOR[risk]  || RISK_COLOR.LOW;
  const ab    = ACTION_BADGE[action] || ACTION_BADGE.ALLOW;

  return (
    <div style={styles.root}>
      {/* ── LEFT PANEL: Payment Form ── */}
      <div style={styles.card}>
        <div style={styles.cardHeader}>
          <span style={styles.rzpLogo}>
            <svg width="22" height="22" viewBox="0 0 40 40" fill="none">
              <path d="M8 32L20 8l4 10-8 4 16 10H8z" fill="#528FF0"/>
            </svg>
          </span>
          <h2 style={styles.cardTitle}>Razorpay Test Payment</h2>
        </div>
        <p style={styles.cardSub}>
          Uses Razorpay test gateway → enriched with fraud features → scored by ArgusAI ML engine
        </p>

        <label style={styles.label}>Amount (₹)</label>
        <input
          style={styles.input}
          type="number"
          placeholder="e.g. 4999"
          min="1"
          value={form.amount}
          onChange={(e) => set("amount", e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handlePay()}
        />

        <label style={styles.label}>Merchant Category</label>
        <select style={styles.select} value={form.merchant_category}
          onChange={(e) => set("merchant_category", e.target.value)}>
          {CATEGORIES.map((c) => <option key={c}>{c}</option>)}
        </select>

        <label style={styles.label}>City</label>
        <select style={styles.select} value={form.transaction_city}
          onChange={(e) => set("transaction_city", e.target.value)}>
          {CITIES.map((c) => <option key={c}>{c}</option>)}
        </select>

        <label style={styles.label}>Device</label>
        <div style={styles.deviceRow}>
          {DEVICES.map((d) => (
            <button key={d}
              style={{
                ...styles.deviceBtn,
                ...(form.device_type === d ? styles.deviceBtnActive : {}),
              }}
              onClick={() => set("device_type", d)}>
              {d === "Mobile" ? "📱" : d === "Desktop" ? "🖥" : "📲"} {d}
            </button>
          ))}
        </div>

        {error && <div style={styles.error}>⚠ {error}</div>}

        <button style={{ ...styles.payBtn, opacity: loading ? 0.6 : 1 }}
          onClick={handlePay} disabled={loading}>
          {loading
            ? <span style={styles.spinner} />
            : <>
                <svg width="18" height="18" viewBox="0 0 40 40" fill="none" style={{ marginRight: 8 }}>
                  <path d="M8 32L20 8l4 10-8 4 16 10H8z" fill="white"/>
                </svg>
                Pay with Razorpay
              </>
          }
        </button>

        <p style={styles.hint}>
          Use card <code style={styles.code}>4111 1111 1111 1111</code> · Any future expiry · CVV <code style={styles.code}>123</code>
        </p>
      </div>

      {/* ── RIGHT PANEL: Result + History ── */}
      <div style={styles.right}>

        {/* Latest Result */}
        {selected && (
          <div style={{ ...styles.resultCard, borderColor: rc.border, background: rc.bg }}>
            <div style={styles.resultHeader}>
              <span style={styles.resultTitle}>Latest Fraud Score</span>
              <span style={{ ...styles.actionBadge, background: ab.bg, color: ab.color }}>
                {ab.label}
              </span>
            </div>

            <div style={styles.scoreRow}>
              <div style={styles.scoreBlock}>
                <div style={{ ...styles.scoreNum, color: rc.text }}>
                  {((selected.result?.fraud_probability || 0) * 100).toFixed(1)}%
                </div>
                <div style={styles.scoreLabel}>Fraud Probability</div>
              </div>
              <div style={styles.scoreBlock}>
                <div style={{ ...styles.scoreNum, color: rc.text }}>
                  {selected.result?.risk_score?.toFixed(2) ?? "—"}
                </div>
                <div style={styles.scoreLabel}>Risk Score</div>
              </div>
              <div style={styles.scoreBlock}>
                <div style={{ ...styles.scoreNum, color: rc.text }}>{risk}</div>
                <div style={styles.scoreLabel}>Risk Level</div>
              </div>
            </div>

            <div style={styles.metaGrid}>
              {[
                ["Transaction ID",  selected.transaction?.transaction_id],
                ["Amount",          `₹${selected.transaction?.amount?.toFixed(2)}`],
                ["Payment Method",  selected.razorpay?.method?.toUpperCase()],
                ["City",            selected.transaction?.transaction_city],
                ["Device",          selected.transaction?.device_type],
                ["Razorpay ID",     selected.razorpay?.payment_id],
              ].map(([k, v]) => (
                <div key={k} style={styles.metaItem}>
                  <span style={styles.metaKey}>{k}</span>
                  <span style={styles.metaVal}>{v ?? "—"}</span>
                </div>
              ))}
            </div>

            {selected.result?.top_features && (
              <div style={styles.features}>
                <div style={styles.featuresTitle}>Top Risk Factors</div>
                {selected.result.top_features.slice(0, 4).map((f, i) => (
                  <div key={i} style={styles.featRow}>
                    <span style={styles.featName}>{f.feature}</span>
                    <div style={styles.featBar}>
                      <div style={{
                        ...styles.featFill,
                        width: `${Math.min(Math.abs(f.shap_value || f.importance || 0) * 300, 100)}%`,
                        background: f.shap_value > 0 ? "#f87171" : "#4ade80",
                      }}/>
                    </div>
                    <span style={styles.featVal}>
                      {(f.shap_value ?? f.importance ?? 0).toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {!selected && (
          <div style={styles.emptyState}>
            <div style={styles.emptyIcon}>⚡</div>
            <div style={styles.emptyText}>Complete a test payment to see real-time fraud scoring</div>
          </div>
        )}

        {/* History */}
        {payments.length > 0 && (
          <div style={styles.history} ref={listRef}>
            <div style={styles.historyTitle}>Payment History ({payments.length})</div>
            {payments.map((p, i) => {
              const r  = p.result?.risk_level || "LOW";
              const rc2 = RISK_COLOR[r] || RISK_COLOR.LOW;
              const ab2 = ACTION_BADGE[p.result?.action] || ACTION_BADGE.ALLOW;
              return (
                <div key={i}
                  style={{
                    ...styles.histRow,
                    borderLeft: `3px solid ${rc2.border}`,
                    background: selected === p ? "rgba(99,102,241,0.1)" : "transparent",
                    cursor: "pointer",
                  }}
                  onClick={() => setSelected(p)}>
                  <div style={styles.histLeft}>
                    <span style={styles.histId}>{p.transaction?.transaction_id}</span>
                    <span style={styles.histMeta}>
                      ₹{p.transaction?.amount?.toFixed(0)} · {p.transaction?.transaction_city}
                    </span>
                  </div>
                  <div style={styles.histRight}>
                    <span style={{ ...styles.histBadge, background: ab2.bg, color: ab2.color }}>
                      {ab2.label}
                    </span>
                    <span style={{ color: rc2.text, fontSize: 12 }}>
                      {((p.result?.fraud_probability || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────
const styles = {
  root: {
    display: "flex", gap: 24, padding: "24px", minHeight: "100%",
    // fontFamily: "'DM Mono', 'Fira Code', monospace",
    color: "#e2e8f0",
    flexWrap: "wrap",
  },
  card: {
    background: "#0f172a", border: "1px solid #1e293b",
    borderRadius: 12, padding: 28, width: 340, flexShrink: 0,
    display: "flex", flexDirection: "column", gap: 4,
  },
  cardHeader: { display: "flex", alignItems: "center", gap: 10, marginBottom: 4 },
  rzpLogo:    { display: "flex", alignItems: "center" },
  cardTitle:  { fontSize: 17, fontWeight: 700, color: "#f1f5f9", margin: 0 },
  cardSub:    { fontSize: 12, color: "#64748b", marginBottom: 16, lineHeight: 1.6 },
  label:      { fontSize: 11, color: "#94a3b8", textTransform: "uppercase",
                letterSpacing: "0.08em", marginTop: 12, marginBottom: 4 },
  input: {
    background: "#1e293b", border: "1px solid #334155", borderRadius: 8,
    padding: "10px 14px", color: "#f1f5f9", fontSize: 15, outline: "none",
    width: "100%", boxSizing: "border-box",
    fontFamily: "inherit",
  },
  select: {
    background: "#1e293b", border: "1px solid #334155", borderRadius: 8,
    padding: "10px 14px", color: "#f1f5f9", fontSize: 14, outline: "none",
    width: "100%", boxSizing: "border-box", cursor: "pointer",
    fontFamily: "inherit",
  },
  deviceRow:      { display: "flex", gap: 8, marginTop: 4 },
  deviceBtn: {
    flex: 1, padding: "8px 4px", borderRadius: 8, border: "1px solid #334155",
    background: "#1e293b", color: "#94a3b8", cursor: "pointer", fontSize: 12,
    fontFamily: "inherit", transition: "all 0.15s",
  },
  deviceBtnActive: { borderColor: "#6366f1", color: "#a5b4fc", background: "rgba(99,102,241,0.15)" },
  error: {
    background: "rgba(239,68,68,0.1)", border: "1px solid #7f1d1d",
    borderRadius: 8, padding: "8px 12px", color: "#fca5a5", fontSize: 13,
    marginTop: 8,
  },
  payBtn: {
    marginTop: 20, padding: "13px 0", borderRadius: 10,
    background: "linear-gradient(135deg, #528FF0 0%, #3b67d4 100%)",
    border: "none", color: "white", fontSize: 15, fontWeight: 700,
    cursor: "pointer", display: "flex", alignItems: "center",
    justifyContent: "center", letterSpacing: "0.02em",
    fontFamily: "inherit", transition: "opacity 0.2s",
    boxShadow: "0 4px 20px rgba(82,143,240,0.35)",
  },
  spinner: {
    width: 18, height: 18, borderRadius: "50%",
    border: "2px solid rgba(255,255,255,0.3)",
    borderTopColor: "white",
    animation: "spin 0.7s linear infinite",
    display: "inline-block",
  },
  hint: { fontSize: 11, color: "#475569", textAlign: "center", marginTop: 10, lineHeight: 1.7 },
  code: { background: "#1e293b", padding: "1px 5px", borderRadius: 4, fontSize: 11 },

  // Right panel
  right: { flex: 1, minWidth: 300, display: "flex", flexDirection: "column", gap: 20 },

  resultCard: {
    border: "1px solid", borderRadius: 12, padding: 22,
    background: "rgba(34,197,94,0.08)",
  },
  resultHeader: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 },
  resultTitle:  { fontSize: 13, fontWeight: 700, color: "#94a3b8", textTransform: "uppercase", letterSpacing: "0.08em" },
  actionBadge:  { fontSize: 12, fontWeight: 700, padding: "4px 12px", borderRadius: 6 },

  scoreRow:   { display: "flex", gap: 20, marginBottom: 20 },
  scoreBlock: { flex: 1, textAlign: "center" },
  scoreNum:   { fontSize: 28, fontWeight: 800, lineHeight: 1.1 },
  scoreLabel: { fontSize: 11, color: "#64748b", marginTop: 4 },

  metaGrid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px 16px", marginBottom: 16 },
  metaItem: { display: "flex", flexDirection: "column", gap: 2 },
  metaKey:  { fontSize: 10, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.06em" },
  metaVal:  { fontSize: 13, color: "#cbd5e1", fontWeight: 500 },

  features:      { borderTop: "1px solid rgba(255,255,255,0.06)", paddingTop: 14 },
  featuresTitle: { fontSize: 11, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 10 },
  featRow:       { display: "flex", alignItems: "center", gap: 10, marginBottom: 7 },
  featName:      { fontSize: 12, color: "#94a3b8", width: 160, flexShrink: 0 },
  featBar:       { flex: 1, height: 4, background: "#1e293b", borderRadius: 2, overflow: "hidden" },
  featFill:      { height: "100%", borderRadius: 2, transition: "width 0.4s ease" },
  featVal:       { fontSize: 11, color: "#64748b", width: 44, textAlign: "right" },

  emptyState: {
    border: "1px dashed #1e293b", borderRadius: 12, padding: 48,
    textAlign: "center", display: "flex", flexDirection: "column",
    alignItems: "center", gap: 12,
  },
  emptyIcon: { fontSize: 36, opacity: 0.4 },
  emptyText: { color: "#475569", fontSize: 14 },

  history:      { background: "#0f172a", border: "1px solid #1e293b", borderRadius: 12, padding: 16, maxHeight: 320, overflowY: "auto" },
  historyTitle: { fontSize: 11, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 12 },
  histRow:      { display: "flex", justifyContent: "space-between", alignItems: "center",
                  padding: "10px 12px", borderRadius: 8, marginBottom: 4, paddingLeft: 12 },
  histLeft:     { display: "flex", flexDirection: "column", gap: 3 },
  histId:       { fontSize: 12, color: "#94a3b8", fontWeight: 600 },
  histMeta:     { fontSize: 11, color: "#475569" },
  histRight:    { display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 3 },
  histBadge:    { fontSize: 10, fontWeight: 700, padding: "2px 8px", borderRadius: 4 },
};

// Inject keyframe for spinner
if (typeof document !== "undefined") {
  const id = "rzp-spin-style";
  if (!document.getElementById(id)) {
    const s = document.createElement("style");
    s.id = id;
    s.textContent = "@keyframes spin { to { transform: rotate(360deg); } }";
    document.head.appendChild(s);
  }
}