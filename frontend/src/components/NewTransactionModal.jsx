import React, { useState } from "react";

const API = "http://localhost:8000";

export default function NewTransactionModal({ isOpen, onClose, cities }) {
  const [amount, setAmount] = useState("");
  const [paymentType, setPaymentType] = useState("UPI");
  const [city, setCity] = useState(cities[0] || "");
  const [deviceType, setDeviceType] = useState("Mobile");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const paymentTypes = ["UPI", "Card", "Net Banking", "Wallet"];
  const deviceTypes = ["Mobile", "Desktop", "Tablet"];

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (!amount || Number(amount) <= 0) {
      setError("Please enter a valid amount");
      return;
    }
    if (!city) {
      setError("Please select a city");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch(`${API}/api/transaction/user-initiated`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          amount: Number(amount),
          payment_type: paymentType,
          transaction_city: city,
          device_type: deviceType,
          merchant_category: "E-Commerce", // default
        }),
      });

      const data = await res.json();
      if (res.ok) {
        setAmount("");
        setPaymentType("UPI");
        setCity(cities[0] || "");
        setDeviceType("Mobile");
        onClose();
      } else {
        setError(data.error || "Transaction failed");
      }
    } catch (err) {
      setError("Network error: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div style={{
      position: "fixed",
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: "rgba(0, 0, 0, 0.6)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      zIndex: 1000,
    }}>
      <div style={{
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: "16px",
        padding: "32px",
        maxWidth: "420px",
        width: "90%",
        boxShadow: "0 20px 60px rgba(0, 0, 0, 0.3)",
      }}>
        {/* Header */}
        <div style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: "24px",
        }}>
          <div style={{ fontSize: "24px", fontWeight: "800" }}>
            💳 New Transaction
          </div>
          <button
            onClick={onClose}
            style={{
              background: "transparent",
              border: "none",
              color: "var(--text)",
              fontSize: "24px",
              cursor: "pointer",
              padding: "0",
            }}
          >
            ✕
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          {/* Amount */}
          <div>
            <label style={{ fontSize: "12px", fontWeight: "600", color: "var(--muted)" }}>
              Amount (₹)
            </label>
            <input
              type="number"
              placeholder="Enter amount"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              min="0"
              step="0.01"
              style={{
                width: "100%",
                background: "var(--surface2)",
                border: "1px solid var(--border)",
                borderRadius: "8px",
                padding: "12px",
                color: "var(--text)",
                fontSize: "14px",
                marginTop: "6px",
                outline: "none",
                boxSizing: "border-box",
              }}
            />
          </div>

          {/* Payment Type */}
          <div>
            <label style={{ fontSize: "12px", fontWeight: "600", color: "var(--muted)" }}>
              Payment Type
            </label>
            <select
              value={paymentType}
              onChange={(e) => setPaymentType(e.target.value)}
              style={{
                width: "100%",
                background: "var(--surface2)",
                border: "1px solid var(--border)",
                borderRadius: "8px",
                padding: "12px",
                color: "var(--text)",
                fontSize: "14px",
                marginTop: "6px",
                outline: "none",
                boxSizing: "border-box",
              }}
            >
              {paymentTypes.map(pt => (
                <option key={pt} value={pt}>{pt}</option>
              ))}
            </select>
          </div>

          {/* City */}
          <div>
            <label style={{ fontSize: "12px", fontWeight: "600", color: "var(--muted)" }}>
              City
            </label>
            <select
              value={city}
              onChange={(e) => setCity(e.target.value)}
              style={{
                width: "100%",
                background: "var(--surface2)",
                border: "1px solid var(--border)",
                borderRadius: "8px",
                padding: "12px",
                color: "var(--text)",
                fontSize: "14px",
                marginTop: "6px",
                outline: "none",
                boxSizing: "border-box",
              }}
            >
              {cities.map(c => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>

          {/* Device Type */}
          <div>
            <label style={{ fontSize: "12px", fontWeight: "600", color: "var(--muted)" }}>
              Device Type
            </label>
            <select
              value={deviceType}
              onChange={(e) => setDeviceType(e.target.value)}
              style={{
                width: "100%",
                background: "var(--surface2)",
                border: "1px solid var(--border)",
                borderRadius: "8px",
                padding: "12px",
                color: "var(--text)",
                fontSize: "14px",
                marginTop: "6px",
                outline: "none",
                boxSizing: "border-box",
              }}
            >
              {deviceTypes.map(dt => (
                <option key={dt} value={dt}>{dt}</option>
              ))}
            </select>
          </div>

          {/* Error */}
          {error && (
            <div style={{
              background: "rgba(239, 68, 68, 0.1)",
              border: "1px solid var(--red)",
              borderRadius: "8px",
              padding: "12px",
              color: "var(--red)",
              fontSize: "13px",
            }}>
              ❌ {error}
            </div>
          )}

          {/* Buttons */}
          <div style={{ display: "flex", gap: "12px", marginTop: "12px" }}>
            <button
              type="button"
              onClick={onClose}
              disabled={loading}
              className="btn btn-ghost"
              style={{ flex: 1, fontSize: "14px", padding: "12px" }}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="btn btn-primary"
              style={{ flex: 1, fontSize: "14px", padding: "12px" }}
            >
              {loading ? "Processing..." : "Submit Transaction"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
