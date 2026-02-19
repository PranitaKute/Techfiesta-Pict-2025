import { Link, useLocation } from 'react-router-dom'
const navItems = [
  { path: '/transactions', label: 'Transactions', icon: '💳' },
  { path: '/analytics',    label: 'Analytics',    icon: '📈' },
]

export default function Layout({ children, streaming, onToggleStream, onInjectFraud, onSimulate, wsStatus }) {
  const location = useLocation()

  const statusColor = {
    connected:    'var(--green)',
    disconnected: 'var(--red)',
    connecting:   'var(--yellow)',
  }

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <header style={{
        background: 'var(--surface)', borderBottom: '1px solid var(--border)',
        padding: '0 24px', height: '60px', display: 'flex', alignItems: 'center',
        gap: '16px', flexShrink: 0, position: 'sticky', top: 0, zIndex: 100,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '36px', height: '36px',
            background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
            borderRadius: '10px', display: 'flex', alignItems: 'center',
            justifyContent: 'center', fontSize: '18px', fontWeight: '900', color: '#fff',
          }}>A</div>
          <div>
            <div style={{ fontWeight: '800', fontSize: '16px', letterSpacing: '-0.02em' }}>ArgusAI</div>
            <div style={{ fontSize: '10px', color: 'var(--muted)', marginTop: '-2px' }}>Fraud Detection & Risk Management</div>
          </div>
        </div>

        {/* WS status badge */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: '6px',
          background: 'var(--surface2)', padding: '4px 12px', borderRadius: '999px',
          fontSize: '11px', color: statusColor[wsStatus],
          border: `1px solid ${statusColor[wsStatus]}44`,
        }}>
          <span style={{
            width: '6px', height: '6px', borderRadius: '50%',
            background: statusColor[wsStatus],
            animation: wsStatus === 'connected' ? 'blink 1.5s ease infinite' : 'none',
            display: 'inline-block',
          }} />
          {wsStatus === 'connected' ? 'Live' : wsStatus}
        </div>

        {/* Nav links in header */}
        <nav style={{ display: 'flex', gap: '4px', marginLeft: '12px' }}>
          {navItems.map(item => {
            const active = location.pathname === item.path
            return (
              <Link key={item.path} to={item.path} style={{
                padding: '6px 14px', borderRadius: '6px', textDecoration: 'none',
                fontSize: '13px', fontWeight: active ? '600' : '400',
                background: active ? 'var(--surface2)' : 'transparent',
                color: active ? '#fff' : 'var(--muted)',
                border: active ? '1px solid var(--border)' : '1px solid transparent',
              }}>
                {item.icon} {item.label}
              </Link>
            )
          })}
        </nav>

        <div style={{ flex: 1 }} />

        {/* Action buttons — same as your original */}
        <button className="btn btn-ghost" onClick={onSimulate}>⚡ Simulate</button>
        <button className="btn btn-danger" onClick={onInjectFraud}>🚨 Inject Fraud</button>
        <button className={`btn ${streaming ? 'btn-ghost' : 'btn-success'}`} onClick={onToggleStream}>
          {streaming ? '⏸ Pause' : '▶ Resume'} Stream
        </button>
      </header>

      {/* Page content */}
      <main style={{ flex: 1, padding: '20px 24px', display: 'flex', flexDirection: 'column' }}>
        {children}
      </main>

      <div style={{ padding: '12px 24px', display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--muted)' }}>
        <span>ArgusAI v1.0 — Hybrid XGBoost + Autoencoder Fraud Engine</span>
        <span>Click any row to inspect &nbsp;|&nbsp; 🚨 Inject Fraud for live demo</span>
      </div>
    </div>
  )
}