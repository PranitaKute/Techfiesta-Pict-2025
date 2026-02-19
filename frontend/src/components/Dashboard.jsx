import React from "react";

const riskColor = {
  LOW:    "var(--green)",
  MEDIUM: "var(--yellow)",
  HIGH:   "var(--red)",
};

const actionBadge = {
  ALLOW: <span className="badge badge-green">✅ Allow</span>,
  OTP:   <span className="badge badge-yellow">⚠️ OTP</span>,
  BLOCK: <span className="badge badge-red">🚫 Block</span>,
};

function fmt(amount) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency", currency: "INR", minimumFractionDigits: 2,
  }).format(amount);
}

export default function Dashboard({ transactions = [], onSelect, selected }) {
  // Ensure transactions are sorted by processed_at (ISO) or timestamp (newest first)
  const toISO = (s) => (typeof s === "string" ? s.replace(" ", "T") : s);
  const sortedTransactions = [...transactions].sort((a, b) => {
    const timeA = new Date(toISO(a.processed_at || a.timestamp || 0)).getTime() || 0;
    const timeB = new Date(toISO(b.processed_at || b.timestamp || 0)).getTime() || 0;
    return timeB - timeA; // Newest first
  });

  return (
    <div className="card" style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: "10px",
        marginBottom: "14px",
        flexShrink: 0,
      }}>
        <span className="live-dot" />
        <span style={{ fontWeight: "700", fontSize: "14px" }}>Live Transaction Feed</span>
        <span style={{
          marginLeft: "auto",
          fontSize: "11px",
          color: "var(--muted)",
        }}>
          {sortedTransactions.length} transactions
        </span>
      </div>

      {/* Table */}
      <div style={{ overflowY: "auto", flex: 1 }}>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              {["ID","Amount","Type","City","Risk","Score","Action"].map(h => (
                <th key={h} style={{
                  padding: "8px 10px",
                  textAlign: "left",
                  fontSize: "11px",
                  color: "var(--muted)",
                  fontWeight: "600",
                  letterSpacing: "0.05em",
                  textTransform: "uppercase",
                  position: "sticky",
                  top: 0,
                  background: "var(--surface)",
                }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedTransactions.length === 0 && (
              <tr>
                <td colSpan={7} style={{
                  textAlign: "center",
                  padding: "40px",
                  color: "var(--muted)",
                }}>
                  <div style={{ fontSize: "32px", marginBottom: "8px" }}>⏳</div>
                  Waiting for transactions...
                </td>
              </tr>
            )}
            {sortedTransactions.map((txn, i) => {
              const isSelected = selected?.transaction_id === txn.transaction_id;
              return (
                <tr
                  key={txn.transaction_id || i}
                  onClick={() => onSelect?.(txn)}
                  className="animate-fade-in"
                  style={{
                    borderBottom: "1px solid var(--border)",
                    cursor: "pointer",
                    background: isSelected
                      ? "var(--surface2)"
                      : i % 2 === 0 ? "transparent" : "rgba(255,255,255,0.01)",
                    transition: "background 0.15s",
                    animationDelay: `${Math.min(i * 0.02, 0.3)}s`,
                  }}
                  onMouseEnter={e => e.currentTarget.style.background = "var(--surface2)"}
                  onMouseLeave={e => e.currentTarget.style.background =
                    isSelected ? "var(--surface2)" :
                    i % 2 === 0 ? "transparent" : "rgba(255,255,255,0.01)"}
                >
                  <td style={{ padding: "9px 10px" }}>
                    <span style={{
                      fontFamily: "JetBrains Mono, monospace",
                      fontSize: "11px",
                      color: "var(--muted)",
                    }}>
                      {txn.transaction_id}
                    </span>
                  </td>
                  <td style={{ padding: "9px 10px", fontWeight: "600" }}>
                    {fmt(txn.amount)}
                  </td>
                  <td style={{ padding: "9px 10px", color: "var(--muted)" }}>
                    {txn.payment_type}
                  </td>
                  <td style={{ padding: "9px 10px" }}>
                    {txn.transaction_city}
                  </td>
                  <td style={{ padding: "9px 10px" }}>
                    <span style={{
                      color: riskColor[txn.risk_level] || "var(--muted)",
                      fontWeight: "700",
                      fontSize: "12px",
                    }}>
                      ● {txn.risk_level}
                    </span>
                  </td>
                  <td style={{ padding: "9px 10px" }}>
                    <div style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "6px",
                    }}>
                      <div style={{
                        width: "50px",
                        height: "5px",
                        background: "var(--border)",
                        borderRadius: "3px",
                        overflow: "hidden",
                      }}>
                        <div style={{
                          height: "100%",
                          width: `${txn.risk_score}%`,
                          background: riskColor[txn.risk_level] || "var(--muted)",
                          borderRadius: "3px",
                        }} />
                      </div>
                      <span style={{
                        fontSize: "11px",
                        fontFamily: "JetBrains Mono, monospace",
                        color: riskColor[txn.risk_level],
                      }}>
                        {txn.risk_score}
                      </span>
                    </div>
                  </td>
                  <td style={{ padding: "9px 10px" }}>
                    {actionBadge[txn.action] || txn.action}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}