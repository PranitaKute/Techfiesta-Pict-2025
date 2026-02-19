import React, { useState } from "react";

export default function InjectFraudModal({ isOpen, onClose, onSubmit }) {
  const [riskLevel, setRiskLevel] = useState("MEDIUM");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    setLoading(true);
    try {
      await onSubmit(riskLevel);
      setRiskLevel("MEDIUM");
      onClose();
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div style={{
      position: "fixed",
      top: 0, left: 0, right: 0, bottom: 0,
      background: "rgba(0,0,0,0.6)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      zIndex: 9999,
    }}>
      <div style={{
        background: "var(--surface)",
        padding: "32px",
        borderRadius: "12px",
        boxShadow: "0 20px 60px rgba(0,0,0,0.4)",
        maxWidth: "400px",
        width: "90%",
        border: "1px solid var(--border)",
      }}>
        <h2 style={{ marginTop: 0, fontWeight: "700", fontSize: "18px" }}>
          🚨 Inject Fraud Demo
        </h2>
        
        <p style={{ color: "var(--muted)", fontSize: "13px", lineHeight: "1.6" }}>
          Select a risk level to inject a fraudulent transaction:
        </p>

        {/* Risk Level Selection */}
        <div style={{
          display: "flex",
          gap: "12px",
          marginBottom: "24px",
        }}>
          {[
            { level: "MEDIUM", label: "⚠️ MEDIUM", desc: "Triggers OTP", color: "var(--yellow)" },
            { level: "HIGH", label: "⛔ HIGH", desc: "Auto-blocks", color: "var(--red)" },
          ].map(option => (
            <button
              key={option.level}
              onClick={() => setRiskLevel(option.level)}
              style={{
                flex: 1,
                padding: "14px",
                border: riskLevel === option.level ? `2px solid ${option.color}` : "1px solid var(--border)",
                background: riskLevel === option.level ? `${option.color}22` : "var(--surface2)",
                borderRadius: "8px",
                color: riskLevel === option.level ? option.color : "var(--text)",
                fontWeight: riskLevel === option.level ? "700" : "500",
                fontSize: "12px",
                cursor: "pointer",
                transition: "all 0.2s ease",
              }}
            >
              <div style={{ fontSize: "14px", marginBottom: "4px" }}>{option.label}</div>
              <div style={{ fontSize: "11px", opacity: 0.7 }}>{option.desc}</div>
            </button>
          ))}
        </div>

        {/* Buttons */}
        <div style={{
          display: "flex",
          gap: "12px",
          justifyContent: "flex-end",
        }}>
          <button
            onClick={onClose}
            disabled={loading}
            style={{
              padding: "10px 20px",
              border: "1px solid var(--border)",
              background: "var(--surface2)",
              borderRadius: "6px",
              cursor: loading ? "not-allowed" : "pointer",
              fontSize: "13px",
              opacity: loading ? 0.5 : 1,
            }}
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={loading}
            style={{
              padding: "10px 20px",
              border: "none",
              background: riskLevel === "MEDIUM" ? "var(--yellow)" : "var(--red)",
              color: "#000",
              borderRadius: "6px",
              fontWeight: "700",
              cursor: loading ? "not-allowed" : "pointer",
              fontSize: "13px",
              opacity: loading ? 0.6 : 1,
            }}
          >
            {loading ? "Injecting..." : "Inject"}
          </button>
        </div>
      </div>
    </div>
  );
}
