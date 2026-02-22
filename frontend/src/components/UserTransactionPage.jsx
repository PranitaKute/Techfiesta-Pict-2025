import React, { useState, useEffect, useRef } from "react";

const API = import.meta.env.VITE_API_URL;

export default function UserTransactionPage() {
  const [formData, setFormData] = useState({
    amount: "",
    recipient: "",
    transaction_type: "transfer",
  });

  const [riskResult, setRiskResult] = useState(null); // { risk_score, decision }
  const [loading, setLoading] = useState(false);
  const debounceTimer = useRef(null);

  // Fires on every form change — debounced
  useEffect(() => {
    const { amount, recipient } = formData;
    if (!amount || !recipient) return;
    clearTimeout(debounceTimer.current);
    debounceTimer.current = setTimeout(async () => {
      setLoading(true);
      try {
        const res = await fetch(`${API}/api/pre-screen`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            amount: parseFloat(amount),
            recipient: recipient,
            payment_type: formData.transaction_type,
            timestamp: new Date().toISOString(),
          }),
        });
        const data = await res.json();
        setRiskResult(data);
      } catch (err) {
        console.error("Pre-screen failed", err);
      } finally {
        setLoading(false);
      }
    }, 700);
  }, [formData]);

  const handleChange = (e) => setFormData({ ...formData, [e.target.name]: e.target.value });

  const isBlocked = riskResult?.decision === "BLOCK";
  const needsOTP = riskResult?.decision === "OTP";

  return (
    <div style={{ maxWidth: 520, margin: "48px auto", padding: 24, border: "1px solid var(--border)", borderRadius: 12 }}>
      <h2 style={{ marginBottom: 12 }}>Make a Payment</h2>
      <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 12 }}>
        <div>
          <label>Amount (₹)</label>
          <input name="amount" value={formData.amount} onChange={handleChange} type="number" style={{ width: "100%", padding: 10 }} />
        </div>
        <div>
          <label>Recipient</label>
          <input name="recipient" value={formData.recipient} onChange={handleChange} style={{ width: "100%", padding: 10 }} />
        </div>
        <div>
          <label>Type</label>
          <select name="transaction_type" value={formData.transaction_type} onChange={handleChange} style={{ width: "100%", padding: 10 }}>
            <option value="transfer">Transfer</option>
            <option value="payment">Payment</option>
            <option value="withdrawal">Withdrawal</option>
            <option value="wallet">Wallet</option>
          </select>
        </div>

        {loading && <div style={{ color: "var(--muted)" }}>Analyzing transaction…</div>}

        {riskResult && !loading && (
          <div style={{ padding: 12, border: "1px solid var(--border)", borderRadius: 8 }}>
            <div style={{ fontWeight: 700 }}>{riskResult.decision}</div>
            <div style={{ fontSize: 12, color: "var(--muted)" }}>Risk Score: {riskResult.risk_score}</div>
          </div>
        )}

        <button disabled={!riskResult || isBlocked} style={{ padding: 12, background: isBlocked ? "#fca5a5" : needsOTP ? "#f97316" : "#16a34a", color: "white", border: "none", borderRadius: 8 }}>
          {isBlocked ? "Transaction Blocked" : needsOTP ? "Verify OTP & Pay" : "Pay Now"}
        </button>
      </div>
    </div>
  );
}
