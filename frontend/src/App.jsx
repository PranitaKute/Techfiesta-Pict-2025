import React, { useState, useEffect, useRef, useCallback } from "react";
import { BrowserRouter, Routes, Route, Link, useLocation, useParams, useNavigate } from "react-router-dom";
import StatsBar          from "./components/StatsBar.jsx";
import RiskGauge         from "./components/RiskGauge.jsx";
import ShapPanel         from "./components/ShapPanel.jsx";
import AlertBanner       from "./components/AlertBanner.jsx";
import Dashboard         from "./components/Dashboard.jsx";
import TransactionDetail from "./components/TransactionDetail.jsx";
import SearchFilter      from "./components/SearchFilter.jsx";
import NewTransactionModal from "./components/NewTransactionModal.jsx";
import TransactionsPage  from "./pages/TransactionsPage";
import AnalyticsPage     from "./pages/AnalyticsPage";
import Razorpaytab       from "./components/Razorpaytab.jsx";
import LoginPage         from "./pages/LoginPage.jsx";

const API = "http://localhost:8000";
const WS  = "ws://localhost:8000/ws/stream";

const navItems = [
  { path: "/",             label: "Dashboard",    icon: "📊" },
  { path: "/transactions", label: "Transactions", icon: "💳" },
  { path: "/analytics",    label: "Analytics",    icon: "📈" },
  { path: "/razorpay",     label: "Razorpay",     icon: "⚡" },
];

// Matches Layout.jsx nav exactly — same active style for all tabs
function NavLinks() {
  const location = useLocation();
  return (
    <nav style={{ display: "flex", gap: "4px", marginLeft: "12px" }}>
      {navItems.map(({ path, label, icon }) => {
        const active = location.pathname === path;
        return (
          <Link key={path} to={path} style={{
            padding: "5px 11px", borderRadius: "6px",
            textDecoration: "none", fontSize: "12px",
            fontWeight: active ? "600" : "400",
            background: active ? "var(--surface2)" : "transparent",
            color:      active ? "#fff"            : "var(--muted)",
            border:     active ? "1px solid var(--border)" : "1px solid transparent",
            display: "flex", alignItems: "center", gap: "5px",
          }}>
            <span>{icon}</span>
            <span>{label}</span>
          </Link>
        );
      })}
    </nav>
  );
}

// Transaction Detail Page Component
function TransactionDetailPage({ transactions }) {
  const { id } = useParams();
  const navigate = useNavigate();
  const transaction = transactions.find(t => t.transaction_id === id);

  if (!transaction) {
    return (
      <div style={{ minHeight: "100vh", background: "var(--bg)", display: "flex", flexDirection: "column" }}>
        <header style={{
          background: "var(--surface)", borderBottom: "1px solid var(--border)",
          padding: "0 24px", height: "60px",
          display: "flex", alignItems: "center", gap: "16px",
          flexShrink: 0, position: "sticky", top: 0, zIndex: 100,
        }}>
          <button onClick={() => navigate("/transactions")} className="btn btn-ghost" style={{ marginRight: "auto" }}>← Back</button>
        </header>
        <main style={{ flex: 1, padding: "40px 24px", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
          <div style={{ fontSize: "32px", marginBottom: "16px" }}>🔍</div>
          <div style={{ color: "var(--muted)", fontSize: "16px" }}>Transaction not found</div>
        </main>
      </div>
    );
  }

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg)", display: "flex", flexDirection: "column" }}>
      <header style={{
        background: "var(--surface)", borderBottom: "1px solid var(--border)",
        padding: "0 24px", height: "60px",
        display: "flex", alignItems: "center", gap: "16px",
        flexShrink: 0, position: "sticky", top: 0, zIndex: 100,
      }}>
        <button onClick={() => navigate("/transactions")} className="btn btn-ghost">← Back to Transactions</button>
        <div style={{ flex: 1 }} />
        <span style={{ fontSize: "12px", color: "var(--muted)" }}>Transaction ID: {transaction.transaction_id}</span>
      </header>
      <main style={{ flex: 1, padding: "20px 24px", display: "flex", flexDirection: "column" }}>
        <TransactionDetail transaction={transaction} onBack={() => navigate("/transactions")} />
      </main>
    </div>
  );
}

export default function App() {
  const [transactions,         setTransactions]         = useState([]);
  const [filteredTransactions, setFilteredTransactions] = useState([]);
  const [selected,             setSelected]             = useState(null);
  const [stats,                setStats]                = useState(null);
  const [wsStatus,             setWsStatus]             = useState("connecting");
  const [streaming,            setStreaming]             = useState(true);
  const [showNewTxnModal,      setShowNewTxnModal]      = useState(false);
  const [filterCriteria,       setFilterCriteria]       = useState({
    search: "", minAmount: "", maxAmount: "", city: "", riskLevel: "", action: "",
  });

  const wsRef   = useRef(null);
  const pingRef = useRef(null);

  useEffect(() => {
    const filtered = transactions.filter(txn => {
      const { search, minAmount, maxAmount, city, riskLevel, action } = filterCriteria;
      if (search    && !txn.transaction_id?.toLowerCase().includes(search.toLowerCase())) return false;
      if (minAmount && Number(txn.amount) < Number(minAmount)) return false;
      if (maxAmount && Number(txn.amount) > Number(maxAmount)) return false;
      if (city      && txn.transaction_city !== city)          return false;
      if (riskLevel && txn.risk_level !== riskLevel)           return false;
      if (action    && txn.action !== action)                  return false;
      return true;
    });
    setFilteredTransactions(filtered);
  }, [transactions, filterCriteria]);

  const handleFilter = (criteria) => setFilterCriteria(criteria);

  const connectWS = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    const ws = new WebSocket(WS);
    ws.onopen = () => {
      setWsStatus("connected");
      pingRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ action: "ping" }));
      }, 20000);
    };
    ws.onmessage = (e) => {
      try {
        const txn = JSON.parse(e.data);
        if (txn.action === "pong") return;
        setTransactions(prev => [txn, ...prev].slice(0, 100));
        setFilteredTransactions(prev => {
          const without = prev.filter(t => t.transaction_id !== txn.transaction_id);
          return [txn, ...without].slice(0, 100);
        });
        setSelected(txn);
      } catch {}
    };
    ws.onclose = () => {
      setWsStatus("disconnected");
      clearInterval(pingRef.current);
      setTimeout(connectWS, 2000);
    };
    ws.onerror = () => ws.close();
    wsRef.current = ws;
  }, []);

  useEffect(() => {
    connectWS();
    fetchStats();
    const statsTimer = setInterval(fetchStats, 5000);
    return () => {
      clearInterval(statsTimer);
      clearInterval(pingRef.current);
      wsRef.current?.close();
    };
  }, [connectWS]);

  async function fetchStats() {
    try {
      const res  = await fetch(`${API}/api/stats`);
      const data = await res.json();
      setStats(data.stats);
    } catch {}
  }

  async function toggleStream() {
    const action = streaming ? "stop" : "start";
    await fetch(`${API}/api/stream/control`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, interval: 3.0 }),
    });
    setStreaming(!streaming);
  }

  async function injectFraud() {
    await fetch(`${API}/api/transaction/fraud`, { method: "POST" });
  }

  async function simulateOne() {
    await fetch(`${API}/api/transaction/simulate`, { method: "POST" });
  }

  function openTransaction(txn) {
    setSelected(txn);
    try { window.history.pushState({}, "", `/txn/${txn.transaction_id}`); } catch {}
  }

  function closeDetail() {
    setSelected(null);
    try { window.history.pushState({}, "", "/"); } catch {}
  }

  const statusColor = {
    connected: "var(--green)", disconnected: "var(--red)", connecting: "var(--yellow)",
  };

  const sharedProps = {
    transactions, selected, setSelected, stats,
    wsStatus, streaming,
    onToggleStream:   toggleStream,
    onInjectFraud:    injectFraud,
    onSimulate:       simulateOne,
    onNewTransaction: () => setShowNewTxnModal(true),
  };

  // Shared header used by Dashboard and Razorpay routes (no <Layout>)
  const Header = () => (
    <header style={{
      background: "var(--surface)", borderBottom: "1px solid var(--border)",
      padding: "0 24px", height: "60px",
      display: "flex", alignItems: "center", gap: "16px",
      flexShrink: 0, position: "sticky", top: 0, zIndex: 100,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
        <div style={{
          width: "36px", height: "36px",
          background: "linear-gradient(135deg, #3b82f6, #8b5cf6)",
          borderRadius: "10px", display: "flex", alignItems: "center",
          justifyContent: "center", fontSize: "18px", fontWeight: "900", color: "#fff",
        }}>A</div>
        <div>
          <div style={{ fontWeight: "800", fontSize: "20px", letterSpacing: "-0.02em" }}>ArgusAI</div>
          <div style={{ fontSize: "12px", color: "var(--muted)", marginTop: "-2px" }}>
            Fraud Detection & Risk Management
          </div>
        </div>
      </div>

      <div style={{
        display: "flex", alignItems: "center", gap: "6px",
        background: "var(--surface2)", padding: "4px 12px", borderRadius: "999px",
        fontSize: "11px", color: statusColor[wsStatus],
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

      <NavLinks />
      <div style={{ flex: 1 }} />
      <Link to="/login" className="btn btn-ghost" style={{ marginRight: 8 }}>🔐 Sign In</Link>
      <button className="btn btn-ghost"   onClick={simulateOne}>⚡ Simulate</button>
      <button className="btn btn-danger"  onClick={injectFraud}>🚨 Inject Fraud</button>
      <button className={`btn ${streaming ? "btn-ghost" : "btn-success"}`} onClick={toggleStream}>
        {streaming ? "⏸ Pause" : "▶ Resume"} Stream
      </button>
    </header>
  );

  return (
    <BrowserRouter>
      <Routes>

        {/* Dashboard */}
        <Route path="/" element={
          <div style={{ minHeight: "100vh", background: "var(--bg)", display: "flex", flexDirection: "column" }}>
            <Header />
            <main style={{ flex: 1, padding: "20px 24px", display: "flex", flexDirection: "column" }}>
              <StatsBar stats={stats} />
              <SearchFilter transactions={transactions} filterCriteria={filterCriteria} onFilter={handleFilter} />
              {selected && (selected.action === "OTP" || selected.action === "BLOCK") && (
                <AlertBanner transaction={selected} onVerified={() => setSelected(null)} />
              )}
              <div style={{ display: "grid", gridTemplateColumns: "220px 1fr 280px", gap: "16px", flex: 1, minHeight: 0 }}>
                <div className="card" style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "flex-start", gap: "20px" }}>
                  <div style={{ fontWeight: "700", fontSize: "13px", color: "var(--muted)" }}>CURRENT RISK</div>
                  <RiskGauge score={selected?.risk_score || 0} action={selected?.action} />
                  {selected && (
                    <div style={{ width: "100%", display: "flex", flexDirection: "column", gap: "8px" }}>
                      {[
                        { label: "Amount",  value: `₹${Number(selected.amount || 0).toLocaleString("en-IN")}` },
                        { label: "City",    value: selected.transaction_city || "—" },
                        { label: "Method",  value: selected.payment_type || "—" },
                        { label: "Device",  value: selected.device_type || "—" },
                        { label: "Fraud %", value: `${selected.fraud_prob || 0}%` },
                        { label: "Anomaly", value: selected.is_anomaly ? "⚠️ Yes" : "✅ No" },
                      ].map((item, i) => (
                        <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "5px 8px", background: "var(--surface2)", borderRadius: "6px", fontSize: "11px" }}>
                          <span style={{ color: "var(--muted)" }}>{item.label}</span>
                          <span style={{ fontWeight: "600" }}>{item.value}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                <div>
                  {window.location.pathname.startsWith("/txn/") ? (
                    <TransactionDetail transaction={selected} onBack={closeDetail} />
                  ) : (
                    <Dashboard transactions={filteredTransactions} onSelect={openTransaction} selected={selected} />
                  )}
                </div>
                <ShapPanel explanations={selected?.shap_explanation} />
              </div>
              <div style={{ marginTop: "16px", display: "flex", alignItems: "center", justifyContent: "space-between", fontSize: "11px", color: "var(--muted)" }}>
                <span>ArgusAI v1.0 — Hybrid XGBoost + Autoencoder Fraud Engine</span>
                <span>Click any row to inspect &nbsp;|&nbsp; 🚨 Inject Fraud for live demo</span>
              </div>
            </main>
          </div>
        } />

        {/* Pre-screen payment page (user-facing) is served separately at /user.html */}

        {/* Transactions & Analytics use <Layout> — no extra header */}
        <Route path="/transactions" element={<TransactionsPage {...sharedProps} />} />
        <Route path="/transactions/:id" element={<TransactionDetailPage {...sharedProps} />} />
        <Route path="/analytics"    element={<AnalyticsPage    {...sharedProps} />} />
        <Route path="/login"        element={<LoginPage />} />

        {/* Razorpay — standalone header */}
        <Route path="/razorpay" element={
          <div style={{ minHeight: "100vh", background: "var(--bg)", display: "flex", flexDirection: "column" }}>
            <Header />
            <Razorpaytab />
          </div>
        } />

      </Routes>

      {/* Modal outside routes — always mounted */}
      <NewTransactionModal
        isOpen={showNewTxnModal}
        onClose={() => setShowNewTxnModal(false)}
        cities={["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad"]}
      />
    </BrowserRouter>
  );
}