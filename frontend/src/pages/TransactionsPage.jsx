import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Layout    from '../components/Layout'
import Dashboard from '../components/Dashboard'
import ShapPanel from '../components/ShapPanel'
import RiskGauge from '../components/RiskGauge'
import SearchFilter from '../components/SearchFilter'

export default function TransactionsPage({ transactions, selected, setSelected, wsStatus, streaming, onToggleStream, onInjectFraud, onSimulate }) {
  const navigate = useNavigate()
  const [filterCriteria, setFilterCriteria] = useState({
    search: "",
    minAmount: "",
    maxAmount: "",
    city: "",
    riskLevel: "",
    action: "",
  });

  const filtered = (transactions || []).filter(t => {
    const { search, minAmount, maxAmount, city, riskLevel, action } = filterCriteria;
    // Transaction ID search (partial match, case-insensitive)
    if (search && !t.transaction_id?.toLowerCase().includes(search.toLowerCase())) {
      return false;
    }
    // Amount range
    if (minAmount && Number(t.amount) < Number(minAmount)) return false;
    if (maxAmount && Number(t.amount) > Number(maxAmount)) return false;
    // City
    if (city && t.transaction_city !== city) return false;
    // Risk level (assuming risk_level is a field, or map from action)
    if (riskLevel && t.risk_level !== riskLevel) return false;
    // Action
    if (action && t.action !== action) return false;
    return true;
  });

  const handleFilter = (criteria) => {
    setFilterCriteria(criteria);
  };

  const handleTransactionClick = (txn) => {
    setSelected(txn);
    navigate(`/transactions/${txn.transaction_id}`);
  };

  return (
    <Layout
      wsStatus={wsStatus}
      streaming={streaming}
      onToggleStream={onToggleStream}
      onInjectFraud={onInjectFraud}
      onSimulate={onSimulate}
    >
      <div style={{ marginBottom: '16px' }}>
        <h1 style={{ fontSize: '20px', fontWeight: '800', marginBottom: '4px' }}>💳 All Transactions</h1>
        <p style={{ fontSize: '12px', color: 'var(--muted)' }}>{filtered.length} transactions shown</p>
      </div>

      {/* Search / Filter bar */}
      <SearchFilter
        transactions={transactions}
        filterCriteria={filterCriteria}
        onFilter={handleFilter}
      />

      {/* Table + SHAP */}
      <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr 280px', gap: '16px', flex: 1, minHeight: 0 }}>
        <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '20px' }}>
          <div style={{ fontWeight: '700', fontSize: '13px', color: 'var(--muted)' }}>CURRENT RISK</div>
          <RiskGauge score={selected?.risk_score || 0} action={selected?.action} />
          {selected && (
            <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {[
                { label: 'Amount', value: `₹${Number(selected.amount || 0).toLocaleString('en-IN')}` },
                { label: 'City', value: selected.transaction_city || '—' },
                { label: 'Method', value: selected.payment_type || '—' },
                { label: 'Device', value: selected.device_type || '—' },
                { label: 'Fraud %', value: `${selected.fraud_prob || 0}%` },
                { label: 'Anomaly', value: selected.is_anomaly ? '⚠️ Yes' : '✅ No' },
              ].map((item, i) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '5px 8px', background: 'var(--surface2)', borderRadius: '6px', fontSize: '11px' }}>
                  <span style={{ color: 'var(--muted)' }}>{item.label}</span>
                  <span style={{ fontWeight: '600' }}>{item.value}</span>
                </div>
              ))}
            </div>
          )}
        </div>
        <Dashboard transactions={filtered} onSelect={handleTransactionClick} selected={selected} />
        <ShapPanel explanations={selected?.shap_explanation} />
      </div>
    </Layout>
  )
}