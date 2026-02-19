import React, { useState, useEffect, useRef, useCallback } from "react";
import { BrowserRouter, Routes, Route, Link, useLocation } from "react-router-dom";
import StatsBar    from "./components/StatsBar.jsx";
import RiskGauge   from "./components/RiskGauge.jsx";
import ShapPanel   from "./components/ShapPanel.jsx";
import AlertBanner from "./components/AlertBanner.jsx";
import Dashboard   from "./components/Dashboard.jsx";
import TransactionDetail from "./components/TransactionDetail.jsx";
import SearchFilter from "./components/SearchFilter.jsx";
import NewTransactionModal from "./components/NewTransactionModal.jsx";
import TransactionsPage from "./pages/TransactionsPage";
import AnalyticsPage    from "./pages/AnalyticsPage";

const API = "http://localhost:8000";
const WS  = "ws://localhost:8000/ws/stream";

export default function App() {
  const [transactions, setTransactions] = useState([]);
  const [filteredTransactions, setFilteredTransactions] = useState([]);
  const [selected,     setSelected]     = useState(null);
  const [stats,        setStats]        = useState(null);
  const [wsStatus,     setWsStatus]     = useState("connecting");
  const [streaming,    setStreaming]     = useState(true);
  const [showNewTxnModal, setShowNewTxnModal] = useState(false);
  const [filterCriteria, setFilterCriteria] = useState({
    search: "",
    minAmount: "",
    maxAmount: "",
    city: "",
    riskLevel: "",
    action: "",
  });
  const wsRef = useRef(null);
  const pingRef = useRef(null);

  // ── Apply filters whenever transactions or filter criteria change ──────────────
  useEffect(() => {
    const filtered = transactions.filter(txn => {
      const { search, minAmount, maxAmount, city, riskLevel, action } = filterCriteria;
      // Transaction ID search (partial match, case-insensitive)
      if (search && !txn.transaction_id?.toLowerCase().includes(search.toLowerCase())) {
        return false;
      }
      // Amount range
      if (minAmount && Number(txn.amount) < Number(minAmount)) return false;
      if (maxAmount && Number(txn.amount) > Number(maxAmount)) return false;
      // City
      if (city && txn.transaction_city !== city) return false;
      // Risk level
      if (riskLevel && txn.risk_level !== riskLevel) return false;
      // Action
      if (action && txn.action !== action) return false;
      return true;
    });
    setFilteredTransactions(filtered);
  }, [transactions, filterCriteria]);

  // ── Handle filter changes from SearchFilter component ────────────────────────
  const handleFilter = (criteria) => {
    setFilterCriteria(criteria);
    // Reset selection if selected transaction no longer matches filters
    if (selected && !transactions.some(t => t.transaction_id === selected.transaction_id)) {
      setSelected(null);
    }
  };

  // ── WebSocket ────────────────────────────────────────────────────────────
  const connectWS = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    const ws = new WebSocket(WS);

    ws.onopen = () => {
      setWsStatus("connected");
      // Keepalive ping every 20s
      pingRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ action: "ping" }));
        }
      }, 20000);
    };

    ws.onmessage = (e) => {
      try {
        const txn = JSON.parse(e.data);
        if (txn.action === "pong") return;
        // Add to transactions list (source of truth)
        setTransactions(prev => [txn, ...prev].slice(0, 100));
        // Also ensure the live filtered view shows the newest transaction at the top
        setFilteredTransactions(prev => {
          // remove any existing instance of this txn, then prepend
          const without = prev.filter(t => t.transaction_id !== txn.transaction_id);
          return [txn, ...without].slice(0, 100);
        });
        setSelected(txn);   // auto-select incoming transaction to show OTP/BLOCK
      } catch {}
    };

    ws.onclose = () => {
      setWsStatus("disconnected");
      clearInterval(pingRef.current);
      // Reconnect after 2s
      setTimeout(connectWS, 2000);
    };

    ws.onerror = () => ws.close();
    wsRef.current = ws;
  }, []);

  useEffect(() => {
    connectWS();
    fetchStats();
    const statsTimer = setInterval(fetchStats, 5000);
    // Handle browser navigation (back/forward)
    const onPop = () => {
      const path = window.location.pathname || "/";
      if (path.startsWith("/txn/")) {
        const id = path.split("/txn/")[1];
        // try to find in current transactions
        const found = transactions.find(t => t.transaction_id === id);
        if (found) setSelected(found);
        else fetchTransactionById(id).then(t => t && setSelected(t));
      } else {
        setSelected(null);
      }
    };
    window.addEventListener("popstate", onPop);
    return () => {
      clearInterval(statsTimer);
      clearInterval(pingRef.current);
      wsRef.current?.close();
      window.removeEventListener("popstate", onPop);
    };
  }, [connectWS]);

  async function fetchTransactionById(id) {
    try {
      const res = await fetch(`${API}/api/transactions?limit=200`);
      const data = await res.json();
      return data.transactions.find(t => t.transaction_id === id) || null;
    } catch { return null; }
  }

  // ── Stats ────────────────────────────────────────────────────────────────
  async function fetchStats() {
    try {
      const res  = await fetch(`${API}/api/stats`);
      const data = await res.json();
      setStats(data.stats);
    } catch {}
  }

  // ── Stream control ───────────────────────────────────────────────────────
  async function toggleStream() {
    const action = streaming ? "stop" : "start";
    await fetch(`${API}/api/stream/control`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ action, interval: 3.0 }),
    });
    setStreaming(!streaming);
  }

  // ── Inject fraud demo ────────────────────────────────────────────────────
  async function injectFraud() {
    await fetch(`${API}/api/transaction/fraud`, { method: "POST" });
  }

  // ── Simulate one transaction ─────────────────────────────────────────────
  async function simulateOne() {
    await fetch(`${API}/api/transaction/simulate`, { method: "POST" });
  }

  const statusColor = {
    connected:    "var(--green)",
    disconnected: "var(--red)",
    connecting:   "var(--yellow)",
  };

  // Open a transaction detail page
  function openTransaction(txn) {
    setSelected(txn);
    try {
      window.history.pushState({}, "", `/txn/${txn.transaction_id}`);
    } catch {}
  }

  function closeDetail() {
    setSelected(null);
    try { window.history.pushState({}, "", "/"); } catch {}
  }

  const sharedProps = {
    transactions, selected, setSelected, stats,
    wsStatus, streaming,
    onToggleStream: toggleStream,
    onInjectFraud:  injectFraud,
    onSimulate:     simulateOne,
  };

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={
          <div style={{
            minHeight: "100vh",
            background: "var(--bg)",
            display: "flex",
            flexDirection: "column",
          }}>
            {/* ── Header ─────────────────────────────────────────────────────── */}
            <header style={{
              background: "var(--surface)",
              borderBottom: "1px solid var(--border)",
              padding: "0 24px",
              height: "60px",
              display: "flex",
              alignItems: "center",
              gap: "16px",
              flexShrink: 0,
              position: "sticky",
              top: 0,
              zIndex: 100,
            }}>
              {/* Logo */}
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <div style={{
                  width: "36px", height: "36px",
                  background: "linear-gradient(135deg, #3b82f6, #8b5cf6)",
                  borderRadius: "10px",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: "18px", fontWeight: "900", color: "#fff",
                }}>A</div>
                <div>
                  <div style={{ fontWeight: "800", fontSize: "18px", letterSpacing: "-0.02em" }}>
                    ArgusAI
                  </div>
                  <div style={{ fontSize: "14px", color: "var(--muted)", marginTop: "-2px" }}>
                    Fraud Detection & Risk Management
                  </div>
                </div>
              </div>

              {/* WS status */}
              <div style={{
                display: "flex", alignItems: "center", gap: "6px",
                background: "var(--surface2)",
                padding: "4px 12px", borderRadius: "999px",
                fontSize: "12px", color: statusColor[wsStatus],
                border: `1px solid ${statusColor[wsStatus]}44`,
              }}>
                <span style={{
                  width: "6px", height: "6px", borderRadius: "50%",
                  background: statusColor[wsStatus],
                  animation: wsStatus === "connected" ? "blink 1.5s ease infinite" : "none",
                  display: "inline-block",
                }} />
                {wsStatus === "connected" ? "Live" : wsStatus}
              </div>

              <nav style={{ display: "flex", gap: "4px", marginLeft: "12px" }}>
                <Link to="/" style={{ padding: "6px 14px", borderRadius: "6px", textDecoration: "none", fontSize: "13px", color: "#fff", background: "var(--surface2)", border: "1px solid var(--border)" }}>
                  📊 Dashboard
                </Link>
                <Link to="/transactions" style={{ padding: "6px 14px", borderRadius: "6px", textDecoration: "none", fontSize: "13px", color: "var(--muted)", background: "transparent", border: "1px solid transparent" }}>
                  💳 Transactions
                </Link>
                <Link to="/analytics" style={{ padding: "6px 14px", borderRadius: "6px", textDecoration: "none", fontSize: "13px", color: "var(--muted)", background: "transparent", border: "1px solid transparent" }}>
                  📈 Analytics
                </Link>
              </nav>

              {/* Control buttons */}
              <button className="btn btn-primary" onClick={() => setShowNewTxnModal(true)} title="Initiate a new transaction">
                💳 New Transaction
              </button>
              <button className="btn btn-ghost" onClick={simulateOne} title="Simulate one random transaction">
                ⚡ Simulate
              </button>
              <button className="btn btn-danger" onClick={injectFraud} title="Inject a fraud transaction">
                🚨 Inject Fraud
              </button>
              <button
                className={`btn ${streaming ? "btn-ghost" : "btn-success"}`}
                onClick={toggleStream}
              >
                {streaming ? "⏸ Pause" : "▶ Resume"} Stream
              </button>
            </header>

            {/* ── Main content ───────────────────────────────────────────────── */}
            <main style={{ flex: 1, padding: "20px 24px", display: "flex", flexDirection: "column" }}>

              {/* Stats bar */}
              <StatsBar stats={stats} />

              {/* Search / Filter bar */}
              <SearchFilter
                transactions={transactions}
                filterCriteria={filterCriteria}
                onFilter={handleFilter}
              />

              {/* Alert banner (shows for OTP / BLOCK) */}
              {selected && (selected.action === "OTP" || selected.action === "BLOCK") && (
                <AlertBanner
                  transaction={selected}
                  onVerified={() => setSelected(null)}
                />
              )}

              {/* Main 3-column layout */}
              <div style={{
                display: "grid",
                gridTemplateColumns: "220px 1fr 280px",
                gap: "16px",
                flex: 1,
                minHeight: 0,
              }}>

                {/* LEFT — Risk Gauge */}
                <div className="card" style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "flex-start",
                  gap: "20px",
                }}>
                  <div style={{ fontWeight: "700", fontSize: "13px", color: "var(--muted)" }}>
                    CURRENT RISK
                  </div>
                  <RiskGauge
                    score={selected?.risk_score || 0}
                    action={selected?.action}
                  />

                  {/* Transaction quick-info */}
                  {selected && (
                    <div style={{ width: "100%", display: "flex", flexDirection: "column", gap: "8px" }}>
                      {[
                        { label: "Amount",   value: `₹${Number(selected.amount || 0).toLocaleString("en-IN")}` },
                        { label: "City",     value: selected.transaction_city || "—" },
                        { label: "Method",   value: selected.payment_type || "—" },
                        { label: "Device",   value: selected.device_type || "—" },
                        { label: "Fraud %",  value: `${selected.fraud_prob || 0}%` },
                        { label: "Anomaly",  value: selected.is_anomaly ? "⚠️ Yes" : "✅ No" },
                      ].map((item, i) => (
                        <div key={i} style={{
                          display: "flex",
                          justifyContent: "space-between",
                          padding: "5px 8px",
                          background: "var(--surface2)",
                          borderRadius: "6px",
                          fontSize: "11px",
                        }}>
                          <span style={{ color: "var(--muted)" }}>{item.label}</span>
                          <span style={{ fontWeight: "600" }}>{item.value}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* CENTER — Live feed or detail */}
                <div>
                  {window.location.pathname.startsWith("/txn/") ? (
                    <TransactionDetail transaction={selected} onBack={closeDetail} />
                  ) : (
                    <Dashboard
                      transactions={filteredTransactions}
                      onSelect={openTransaction}
                      selected={selected}
                    />
                  )}
                </div>

                {/* RIGHT — SHAP */}
                <ShapPanel explanations={selected?.shap_explanation} />
              </div>

              {/* ── Footer ──────────────────────────────────────────────────── */}
              <div style={{
                marginTop: "16px",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                fontSize: "11px",
                color: "var(--muted)",
              }}>
                <span>ArgusAI v1.0 — Hybrid XGBoost + Autoencoder Fraud Engine</span>
                <span>
                  Click any row to inspect &nbsp;|&nbsp; 🚨 Inject Fraud for live demo
                </span>
              </div>
            </main>

            {/* New Transaction Modal */}
            <NewTransactionModal
              isOpen={showNewTxnModal}
              onClose={() => setShowNewTxnModal(false)}
              cities={[...new Set(transactions.map(t => t.transaction_city).filter(Boolean))].sort()}
            />
          </div>
        } />
        <Route path="/transactions" element={<TransactionsPage {...sharedProps} />} />
        <Route path="/analytics" element={<AnalyticsPage {...sharedProps} />} />
      </Routes>
    </BrowserRouter>
  );
}