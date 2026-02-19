import React, { useState } from "react";

function fmt(amount) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency", currency: "INR", minimumFractionDigits: 2,
  }).format(amount);
}

// Helper to parse and format SHAP data
function parseSHAPData(shap) {
  if (!shap) return [];
  
  try {
    // If it's a string, parse it
    let data = typeof shap === 'string' ? JSON.parse(shap) : shap;
    
    // Handle array format (list of SHAP explanation objects)
    if (Array.isArray(data)) {
      return data
        .filter(item => item && (item.shap_val !== undefined || item.value !== undefined))
        .map(item => ({
          label: item.label || item.feature || "Unknown",
          value: Number(item.shap_val ?? item.value ?? 0),
          impact: item.impact || "LOW",
        }));
    }
    
    // Handle object format
    if (typeof data === 'object') {
      return Object.entries(data)
        .map(([key, val]) => {
          if (typeof val === 'object' && val.shap_val !== undefined) {
            return { label: val.label || key, value: Number(val.shap_val), impact: val.impact || "LOW" };
          }
          return { label: key, value: Number(val) || 0, impact: "LOW" };
        })
        .filter(item => !isNaN(item.value));
    }
    
    return [];
  } catch (e) {
    console.error('Failed to parse SHAP data:', e, shap);
    return [];
  }
}

export default function TransactionDetail({ transaction, onBack }) {
  const [loadingAction, setLoadingAction] = useState(false);
  const [message, setMessage] = useState("");
  if (!transaction) {
    return (
      <div style={{ padding: 24 }} className="card">
        <div style={{ fontSize: 14, color: "var(--muted)" }}>Transaction not found.</div>
        <div style={{ marginTop: 12 }}>
          <button className="btn" onClick={onBack}>← Back</button>
        </div>
      </div>
    );
  }

  const fields = [
    ["Transaction ID", transaction.transaction_id],
    ["Amount", fmt(transaction.amount)],
    ["Timestamp", transaction.timestamp],
    ["User ID", transaction.user_id],
    ["Payment Type", transaction.payment_type],
    ["Merchant Category", transaction.merchant_category],
    ["City", transaction.transaction_city],
    ["Distance (km)", transaction.distance_from_home_km],
    ["Device", transaction.device_type],
    ["Device Mismatch", transaction.device_mismatch],
    ["Card Age (days)", transaction.card_age_days],
    ["Txn Hour", transaction.transaction_hour],
    ["Is Weekend", transaction.is_weekend],
    ["Is Night", transaction.is_night],
    ["Daily Txn Count", transaction.daily_txn_count],
    ["Avg Amount (7d)", transaction.avg_amount_7d],
    ["Amount vs Avg Ratio", transaction.amount_vs_avg_ratio],
    ["Risk Level", transaction.risk_level],
    ["Risk Score", transaction.risk_score],
    ["Action", transaction.action],
  ];

  return (
    <div style={{ padding: 20 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
        <button className="btn" onClick={onBack}>← Back</button>
        <div style={{ fontWeight: 800, fontSize: 16 }}>Transaction Detail</div>
        <div style={{ marginLeft: "auto", color: "var(--muted)", fontSize: 12 }}>
          {transaction.transaction_id}
        </div>
      </div>

      <div className="card" style={{ padding: 16 }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          {fields.map(([k,v]) => (
            <div key={k} style={{ background: "var(--surface2)", padding: 10, borderRadius: 8 }}>
              <div style={{ color: "var(--muted)", fontSize: 11 }}>{k}</div>
              <div style={{ fontWeight: 700, marginTop: 6 }}>{String(v ?? "—")}</div>
            </div>
          ))}
        </div>

        {/* SHAP explanation */}
        <div style={{ marginTop: 16 }}>
          <div style={{ fontWeight: 700, marginBottom: 8 }}>Model Explanation (SHAP)</div>
          {transaction.shap_explanation ? (() => {
            const shapData = parseSHAPData(transaction.shap_explanation);
            return shapData.length > 0 ? (
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                {shapData.map(({ label, value, impact }) => (
                  <div key={label} style={{
                    background: "var(--surface2)",
                    padding: "12px 14px",
                    borderRadius: "8px",
                    border: "1px solid var(--border)",
                    minWidth: "140px",
                  }}>
                    <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 6 }}>
                      {label}
                      {impact && (
                        <span style={{
                          marginLeft: "6px",
                          fontWeight: 700,
                          color: impact === "HIGH" ? "var(--red)" : impact === "MEDIUM" ? "var(--yellow)" : "var(--green)",
                        }}>
                          [{impact}]
                        </span>
                      )}
                    </div>
                    <div style={{
                      fontWeight: 700,
                      fontSize: 14,
                      color: value > 0 ? "var(--green)" : "var(--red)",
                      fontFamily: "JetBrains Mono, monospace",
                    }}>
                      {value > 0 ? "↑ " : "↓ "}{Math.abs(value).toFixed(4)}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ color: "var(--muted)", fontSize: 13, padding: "12px", background: "var(--surface2)", borderRadius: "8px" }}>
                No SHAP values found. Raw data: {JSON.stringify(transaction.shap_explanation)}
              </div>
            );
          })() : (
            <div style={{ color: "var(--muted)", fontSize: 13 }}>No explanation available.</div>
          )}
        </div>

        {/* Actions */}
        <div style={{ marginTop: 18, display: "flex", gap: 8, alignItems: 'center' }}>
          <button
            className="btn btn-success"
            disabled={loadingAction}
            onClick={() => handleAction('ALLOW')}
          >{loadingAction ? 'Working...' : 'Approve'}</button>
          <button
            className="btn btn-danger"
            disabled={loadingAction}
            onClick={() => handleAction('BLOCK')}
          >{loadingAction ? 'Working...' : 'Block'}</button>
          <button
            className="btn btn-ghost"
            disabled={loadingAction}
            onClick={() => handleAction('OTP')}
          >{loadingAction ? 'Working...' : 'Request OTP'}</button>
          {message && (
            <div style={{ marginLeft: 12, color: 'var(--muted)', fontWeight: 700 }}>{message}</div>
          )}
        </div>

        {/* Local handlers */}
      </div>
    </div>
  );
}
async function apiSetAction(transaction_id, action) {
  const res = await fetch("http://localhost:8000/api/transaction/action", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ transaction_id, action }),
  });
  return await res.json();
}

// Note: `handleAction` is defined inside the component so it closes over state.