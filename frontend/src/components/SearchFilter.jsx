import React, { useState, useCallback } from "react";

export default function SearchFilter({ transactions, filterCriteria, onFilter }) {
  const [search, setSearch] = useState("");
  const [minAmount, setMinAmount] = useState("");
  const [maxAmount, setMaxAmount] = useState("");
  const [city, setCity] = useState("");
  const [riskLevel, setRiskLevel] = useState("");
  const [action, setAction] = useState("");

  // Get unique values for dropdowns
  const cities = [...new Set(transactions.map(t => t.transaction_city).filter(Boolean))].sort();
  const riskLevels = ["LOW", "MEDIUM", "HIGH"];
  const actions = ["ALLOW", "BLOCK", "OTP"];

  // Update filters whenever any input changes (debounced by onChange)
  const updateFilter = useCallback((newSearch, newMinAmount, newMaxAmount, newCity, newRiskLevel, newAction) => {
    const criteria = { 
      search: newSearch, 
      minAmount: newMinAmount, 
      maxAmount: newMaxAmount, 
      city: newCity, 
      riskLevel: newRiskLevel, 
      action: newAction 
    };
    onFilter(criteria);
  }, [onFilter]);

  const handleSearch = () => {
    // Search button for manual trigger
    updateFilter(search, minAmount, maxAmount, city, riskLevel, action);
  };

  const handleReset = () => {
    setSearch("");
    setMinAmount("");
    setMaxAmount("");
    setCity("");
    setRiskLevel("");
    setAction("");
    updateFilter("", "", "", "", "", "");
  };

  const inputStyle = {
    background: "var(--surface2)",
    border: "1px solid var(--border)",
    borderRadius: "10px",
    padding: "14px 16px",
    color: "var(--text)",
    fontSize: "14px",
    fontFamily: "'Inter', sans-serif",
    outline: "none",
    transition: "all 0.2s ease",
    boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
  };

  const inputFocusStyle = {
    ...inputStyle,
    borderColor: "var(--blue)",
    boxShadow: "0 4px 12px rgba(59, 130, 246, 0.2)",
  };

  return (
    <div style={{
      background: "var(--surface)",
      border: "1px solid var(--border)",
      borderRadius: "16px",
      padding: "24px",
      marginBottom: "20px",
      boxShadow: "0 4px 20px rgba(0,0,0,0.15)",
    }}>
      {/* Header */}
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: "10px",
        marginBottom: "20px",
      }}>
        <div style={{ fontSize: "24px" }}>🔍</div>
        <div style={{
          fontWeight: "800",
          fontSize: "18px",
          color: "var(--text)",
        }}>
          Search & Filter Transactions
        </div>
        <div style={{
          marginLeft: "auto",
          fontSize: "12px",
          color: "var(--muted)",
          background: "var(--surface2)",
          padding: "6px 12px",
          borderRadius: "8px",
        }}>
          {transactions.length} total
        </div>
      </div>

      {/* Filter Grid */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
        gap: "16px",
        marginBottom: "20px",
      }}>
        {/* Transaction ID */}
        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          <label style={{ fontSize: "13px", fontWeight: "600", color: "var(--muted)" }}>
            Transaction ID
          </label>
          <input
            type="text"
            placeholder="Search by ID..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              updateFilter(e.target.value, minAmount, maxAmount, city, riskLevel, action);
            }}
            onFocus={(e) => Object.assign(e.target.style, inputFocusStyle)}
            onBlur={(e) => Object.assign(e.target.style, inputStyle)}
            style={inputStyle}
          />
        </div>

        {/* Min Amount */}
        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          <label style={{ fontSize: "13px", fontWeight: "600", color: "var(--muted)" }}>
            Min Amount (₹)
          </label>
          <input
            type="number"
            placeholder="0"
            value={minAmount}
            onChange={(e) => {
              setMinAmount(e.target.value);
              updateFilter(search, e.target.value, maxAmount, city, riskLevel, action);
            }}
            onFocus={(e) => Object.assign(e.target.style, inputFocusStyle)}
            onBlur={(e) => Object.assign(e.target.style, inputStyle)}
            style={inputStyle}
          />
        </div>

        {/* Max Amount */}
        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          <label style={{ fontSize: "13px", fontWeight: "600", color: "var(--muted)" }}>
            Max Amount (₹)
          </label>
          <input
            type="number"
            placeholder="∞"
            value={maxAmount}
            onChange={(e) => {
              setMaxAmount(e.target.value);
              updateFilter(search, minAmount, e.target.value, city, riskLevel, action);
            }}
            onFocus={(e) => Object.assign(e.target.style, inputFocusStyle)}
            onBlur={(e) => Object.assign(e.target.style, inputStyle)}
            style={inputStyle}
          />
        </div>

        {/* City Dropdown */}
        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          <label style={{ fontSize: "13px", fontWeight: "600", color: "var(--muted)" }}>
            City
          </label>
          <select
            value={city}
            onChange={(e) => {
              setCity(e.target.value);
              updateFilter(search, minAmount, maxAmount, e.target.value, riskLevel, action);
            }}
            onFocus={(e) => Object.assign(e.target.style, inputFocusStyle)}
            onBlur={(e) => Object.assign(e.target.style, inputStyle)}
            style={inputStyle}
          >
            <option value="">All Cities</option>
            {cities.map(c => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>

        {/* Risk Level Dropdown */}
        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          <label style={{ fontSize: "13px", fontWeight: "600", color: "var(--muted)" }}>
            Risk Level
          </label>
          <select
            value={riskLevel}
            onChange={(e) => {
              setRiskLevel(e.target.value);
              updateFilter(search, minAmount, maxAmount, city, e.target.value, action);
            }}
            onFocus={(e) => Object.assign(e.target.style, inputFocusStyle)}
            onBlur={(e) => Object.assign(e.target.style, inputStyle)}
            style={inputStyle}
          >
            <option value="">All Risk Levels</option>
            {riskLevels.map(rl => (
              <option key={rl} value={rl}>{rl}</option>
            ))}
          </select>
        </div>

        {/* Action Dropdown */}
        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          <label style={{ fontSize: "13px", fontWeight: "600", color: "var(--muted)" }}>
            Action
          </label>
          <select
            value={action}
            onChange={(e) => {
              setAction(e.target.value);
              updateFilter(search, minAmount, maxAmount, city, riskLevel, e.target.value);
            }}
            onFocus={(e) => Object.assign(e.target.style, inputFocusStyle)}
            onBlur={(e) => Object.assign(e.target.style, inputStyle)}
            style={inputStyle}
          >
            <option value="">All Actions</option>
            {actions.map(a => (
              <option key={a} value={a}>{a}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Action buttons */}
      <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
        <button
          className="btn btn-primary"
          onClick={handleSearch}
          style={{
            fontSize: "14px",
            padding: "12px 28px",
            fontWeight: "600",
            letterSpacing: "0.5px",
          }}
        >
          🔎 Search
        </button>
        <button
          className="btn btn-ghost"
          onClick={handleReset}
          style={{
            fontSize: "14px",
            padding: "12px 28px",
            fontWeight: "600",
            letterSpacing: "0.5px",
          }}
        >
          ↻ Reset
        </button>
        <div style={{
          marginLeft: "auto",
          fontSize: "12px",
          color: "var(--muted)",
          fontStyle: "italic",
        }}>
          ✨ Filters apply automatically (Press Enter or click Search)
        </div>
      </div>
    </div>
  );
}
