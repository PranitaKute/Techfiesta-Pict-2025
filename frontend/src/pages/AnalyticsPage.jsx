import Layout   from '../components/Layout'
import StatsBar from '../components/StatsBar'

export default function AnalyticsPage({ transactions, stats, wsStatus, streaming, onToggleStream, onInjectFraud, onSimulate }) {
  const data    = transactions || []
  const total   = data.length || 1
  const blocked = data.filter(t => t.action === 'BLOCK').length
  const otps    = data.filter(t => t.action === 'OTP').length
  const allowed = data.filter(t => t.action === 'ALLOW').length

  const avgScore = data.length
    ? (data.reduce((s, t) => s + (t.risk_score || 0), 0) / data.length).toFixed(1)
    : 0

  const byType = data.reduce((acc, t) => {
    const k = t.payment_type || 'Unknown'
    acc[k] = (acc[k] || 0) + 1
    return acc
  }, {})

  const byCity = data.reduce((acc, t) => {
    const k = t.transaction_city || 'Unknown'
    acc[k] = (acc[k] || 0) + 1
    return acc
  }, {})

  const barColor = (pct) => pct > 50 ? '#ef4444' : pct > 25 ? '#f59e0b' : '#22c55e'

  return (
    <Layout
      wsStatus={wsStatus}
      streaming={streaming}
      onToggleStream={onToggleStream}
      onInjectFraud={onInjectFraud}
      onSimulate={onSimulate}
    >
      <div style={{ marginBottom: '20px' }}>
        <h1 style={{ fontSize: '20px', fontWeight: '800', marginBottom: '4px' }}>📈 Analytics</h1>
        <p style={{ fontSize: '12px', color: 'var(--muted)' }}>Computed from {data.length} live transactions this session</p>
      </div>

      <StatsBar stats={stats} />

      {/* Summary cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginBottom: '20px' }}>
        {[
          { label: 'Avg Risk Score', value: avgScore,                              color: '#f59e0b' },
          { label: 'Block Rate',     value: `${((blocked/total)*100).toFixed(1)}%`, color: '#ef4444' },
          { label: 'OTP Rate',       value: `${((otps/total)*100).toFixed(1)}%`,    color: '#a855f7' },
          { label: 'Allow Rate',     value: `${((allowed/total)*100).toFixed(1)}%`, color: '#22c55e' },
        ].map((c, i) => (
          <div key={i} className="card" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '28px', fontWeight: '800', color: c.color }}>{c.value}</div>
            <div style={{ fontSize: '12px', color: 'var(--muted)', marginTop: '4px' }}>{c.label}</div>
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>

        {/* By payment type */}
        <div className="card">
          <div style={{ fontWeight: '700', fontSize: '13px', marginBottom: '16px' }}>💳 By Payment Type</div>
          {Object.entries(byType).sort((a,b) => b[1]-a[1]).map(([type, count]) => {
            const pct = ((count / total) * 100).toFixed(1)
            return (
              <div key={type} style={{ marginBottom: '12px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                  <span>{type}</span>
                  <span style={{ color: 'var(--muted)' }}>{count} ({pct}%)</span>
                </div>
                <div style={{ height: '6px', background: 'var(--surface2)', borderRadius: '3px', overflow: 'hidden' }}>
                  <div style={{ width: `${pct}%`, height: '100%', background: '#3b82f6', borderRadius: '3px', transition: 'width 0.5s' }} />
                </div>
              </div>
            )
          })}
        </div>

        {/* By city */}
        <div className="card">
          <div style={{ fontWeight: '700', fontSize: '13px', marginBottom: '16px' }}>🌆 By City</div>
          {Object.entries(byCity).sort((a,b) => b[1]-a[1]).slice(0, 8).map(([city, count]) => {
            const pct = ((count / total) * 100).toFixed(1)
            return (
              <div key={city} style={{ marginBottom: '12px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                  <span>{city}</span>
                  <span style={{ color: 'var(--muted)' }}>{count} ({pct}%)</span>
                </div>
                <div style={{ height: '6px', background: 'var(--surface2)', borderRadius: '3px', overflow: 'hidden' }}>
                  <div style={{ width: `${pct}%`, height: '100%', background: barColor(Number(pct)), borderRadius: '3px', transition: 'width 0.5s' }} />
                </div>
              </div>
            )
          })}
        </div>

        {/* Decision breakdown */}
        <div className="card">
          <div style={{ fontWeight: '700', fontSize: '13px', marginBottom: '16px' }}>⚖️ Decision Breakdown</div>
          {[
            { label: 'ALLOW', count: allowed, color: '#22c55e' },
            { label: 'BLOCK', count: blocked, color: '#ef4444' },
            { label: 'OTP',   count: otps,    color: '#a855f7' },
          ].map(row => {
            const pct = ((row.count / total) * 100).toFixed(1)
            return (
              <div key={row.label} style={{ marginBottom: '16px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '6px' }}>
                  <span style={{ color: row.color, fontWeight: '700' }}>● {row.label}</span>
                  <span style={{ color: 'var(--muted)' }}>{row.count} ({pct}%)</span>
                </div>
                <div style={{ height: '8px', background: 'var(--surface2)', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{ width: `${pct}%`, height: '100%', background: row.color, borderRadius: '4px', transition: 'width 0.5s' }} />
                </div>
              </div>
            )
          })}
        </div>

        {/* Recent blocked */}
        <div className="card">
          <div style={{ fontWeight: '700', fontSize: '13px', marginBottom: '16px' }}>🚨 Recent Blocked Transactions</div>
          {data.filter(t => t.action === 'BLOCK').slice(0, 5).map((t, i) => (
            <div key={i} style={{
              display: 'flex', justifyContent: 'space-between', padding: '8px',
              background: 'var(--surface2)', borderRadius: '6px', marginBottom: '6px',
              fontSize: '12px', borderLeft: '3px solid #ef4444'
            }}>
              <span style={{ color: '#ef4444', fontFamily: 'monospace' }}>{t.transaction_id}</span>
              <span>₹{Number(t.amount || 0).toLocaleString('en-IN')}</span>
              <span style={{ color: 'var(--muted)' }}>Score: {t.risk_score}</span>
            </div>
          ))}
          {data.filter(t => t.action === 'BLOCK').length === 0 && (
            <div style={{ color: 'var(--muted)', fontSize: '12px', textAlign: 'center', padding: '20px' }}>
              ✅ No blocked transactions yet
            </div>
          )}
        </div>

      </div>
    </Layout>
  )
}