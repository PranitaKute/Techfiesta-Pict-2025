import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

const API = "http://localhost:8000";

// Animated background grid dots — pure CSS, no libs needed
const GridBackground = () => (
  <div style={{
    position: "fixed", inset: 0, zIndex: 0, overflow: "hidden",
    background: "var(--bg)",
  }}>
    {/* Radial glow behind the card */}
    <div style={{
      position: "absolute",
      top: "50%", left: "50%",
      transform: "translate(-50%, -50%)",
      width: "600px", height: "600px",
      background: "radial-gradient(circle, #3b82f611 0%, #8b5cf608 40%, transparent 70%)",
      borderRadius: "50%",
    }} />
    {/* Subtle grid */}
    <div style={{
      position: "absolute", inset: 0,
      backgroundImage: `
        linear-gradient(rgba(59,130,246,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(59,130,246,0.04) 1px, transparent 1px)
      `,
      backgroundSize: "40px 40px",
    }} />
    {/* Floating particles */}
    {[...Array(6)].map((_, i) => (
      <div key={i} style={{
        position: "absolute",
        width: `${4 + i * 2}px`, height: `${4 + i * 2}px`,
        borderRadius: "50%",
        background: i % 2 === 0 ? "#3b82f633" : "#8b5cf633",
        top: `${10 + i * 14}%`,
        left: `${5 + i * 16}%`,
        animation: `float${i % 3} ${3 + i}s ease-in-out infinite`,
      }} />
    ))}
    <style>{`
      @keyframes float0 { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-12px)} }
      @keyframes float1 { 0%,100%{transform:translateY(0)} 50%{transform:translateY(10px)} }
      @keyframes float2 { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
      @keyframes slideUp {
        from { opacity: 0; transform: translateY(24px); }
        to   { opacity: 1; transform: translateY(0); }
      }
      @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
      }
      @keyframes shake {
        0%,100% { transform: translateX(0); }
        20%     { transform: translateX(-8px); }
        40%     { transform: translateX(8px); }
        60%     { transform: translateX(-6px); }
        80%     { transform: translateX(6px); }
      }
      .auth-card {
        animation: slideUp 0.45s cubic-bezier(0.22, 1, 0.36, 1) both;
      }
      .auth-input {
        width: 100%;
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 11px 14px;
        color: #fff;
        font-size: 13px;
        outline: none;
        transition: border-color 0.2s, box-shadow 0.2s;
        box-sizing: border-box;
        font-family: inherit;
      }
      .auth-input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px #3b82f622;
      }
      .auth-input::placeholder { color: var(--muted); }
      .auth-btn-primary {
        width: 100%;
        padding: 12px;
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        border: none;
        border-radius: 8px;
        color: #fff;
        font-size: 13px;
        font-weight: 700;
        cursor: pointer;
        transition: opacity 0.2s, transform 0.1s;
        font-family: inherit;
        letter-spacing: 0.02em;
      }
      .auth-btn-primary:hover:not(:disabled) { opacity: 0.88; transform: translateY(-1px); }
      .auth-btn-primary:active { transform: translateY(0); }
      .auth-btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
      .tab-btn {
        flex: 1; padding: "10px"; border: none;
        background: transparent; cursor: pointer;
        font-family: inherit; font-size: 13px;
        transition: color 0.2s;
      }
      .error-shake { animation: shake 0.4s ease; }
    `}</style>
  </div>
);

export default function LoginPage() {
  const navigate = useNavigate();
  const [mode,     setMode]     = useState("login");   // "login" | "register"
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirm,  setConfirm]  = useState("");
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState("");
  const [success,  setSuccess]  = useState("");
  const [shake,    setShake]    = useState(false);

  // Clear errors on mode switch
  useEffect(() => {
    setError(""); setSuccess(""); setUsername(""); setPassword(""); setConfirm("");
  }, [mode]);

  const triggerShake = () => {
    setShake(true);
    setTimeout(() => setShake(false), 500);
  };

  async function handleSubmit(e) {
    e.preventDefault();
    setError(""); setSuccess("");

    // Client-side validation
    if (!username.trim() || !password.trim()) {
      setError("Please fill in all fields."); triggerShake(); return;
    }
    if (mode === "register") {
      if (password !== confirm) {
        setError("Passwords don't match."); triggerShake(); return;
      }
      if (password.length < 6) {
        setError("Password must be at least 6 characters."); triggerShake(); return;
      }
    }

    setLoading(true);
    try {
      const endpoint = mode === "login" ? "/api/login" : "/api/register";
      const res  = await fetch(`${API}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: username.trim(), password }),
      });
      const data = await res.json();

      if (!data.ok) {
        const msgs = {
          user_exists:        "Username already taken. Try another.",
          not_found:          "Account not found. Please register first.",
          invalid_credentials:"Wrong password. Please try again.",
        };
        setError(msgs[data.error] || "Something went wrong. Please try again.");
        triggerShake();
      } else {
        // Success
        if (mode === "register") {
          setSuccess("Account created! Signing you in…");
          // Auto-switch to login after a beat
          setTimeout(() => {
            setMode("login");
            setSuccess("");
          }, 1200);
        } else {
          // Save user to sessionStorage so rest of app knows who's logged in
          sessionStorage.setItem("argus_user", JSON.stringify(data.user));
          setSuccess(`Welcome back, ${data.user.username}! Redirecting…`);
          setTimeout(() => navigate("/pay"), 900);
        }
      }
    } catch {
      setError("Cannot reach server. Is the backend running?");
      triggerShake();
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", position: "relative" }}>
      <GridBackground />

      <div className={`auth-card ${shake ? "error-shake" : ""}`} style={{
        position: "relative", zIndex: 1,
        width: "100%", maxWidth: "400px",
        margin: "0 16px",
      }}>

        {/* Logo */}
        <div style={{ textAlign: "center", marginBottom: "32px" }}>
          <div style={{
            width: "56px", height: "56px", margin: "0 auto 14px",
            background: "linear-gradient(135deg, #3b82f6, #8b5cf6)",
            borderRadius: "16px",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: "28px", fontWeight: "900", color: "#fff",
            boxShadow: "0 8px 32px #3b82f640",
          }}>A</div>
          <div style={{ fontWeight: "800", fontSize: "22px", letterSpacing: "-0.02em" }}>ArgusAI</div>
          <div style={{ fontSize: "12px", color: "var(--muted)", marginTop: "2px" }}>
            Fraud Detection & Risk Management
          </div>
        </div>

        {/* Card */}
        <div className="card" style={{ padding: "28px", border: "1px solid var(--border)" }}>

          {/* Tab switcher */}
          <div style={{
            display: "flex", marginBottom: "24px",
            background: "var(--surface2)", borderRadius: "8px", padding: "3px",
          }}>
            {["login", "register"].map(m => (
              <button
                key={m}
                onClick={() => setMode(m)}
                style={{
                  flex: 1, padding: "8px", border: "none", cursor: "pointer",
                  borderRadius: "6px", fontSize: "12px", fontWeight: "600",
                  fontFamily: "inherit", textTransform: "capitalize",
                  transition: "all 0.2s",
                  background: mode === m ? "var(--surface)" : "transparent",
                  color:      mode === m ? "#fff"           : "var(--muted)",
                  boxShadow:  mode === m ? "0 1px 4px #0006" : "none",
                }}
              >
                {m === "login" ? "🔐 Sign In" : "✨ Register"}
              </button>
            ))}
          </div>

          <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "14px" }}>

            {/* Username */}
            <div>
              <label style={{ display: "block", fontSize: "11px", fontWeight: "600", color: "var(--muted)", marginBottom: "6px", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                Username
              </label>
              <input
                className="auth-input"
                type="text"
                placeholder="e.g. john_doe"
                value={username}
                onChange={e => setUsername(e.target.value)}
                autoFocus
                autoComplete="username"
                disabled={loading}
              />
            </div>

            {/* Password */}
            <div>
              <label style={{ display: "block", fontSize: "11px", fontWeight: "600", color: "var(--muted)", marginBottom: "6px", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                Password
              </label>
              <input
                className="auth-input"
                type="password"
                placeholder={mode === "register" ? "Min. 6 characters" : "Enter your password"}
                value={password}
                onChange={e => setPassword(e.target.value)}
                autoComplete={mode === "login" ? "current-password" : "new-password"}
                disabled={loading}
              />
            </div>

            {/* Confirm password (register only) */}
            {mode === "register" && (
              <div style={{ animation: "slideUp 0.3s ease" }}>
                <label style={{ display: "block", fontSize: "11px", fontWeight: "600", color: "var(--muted)", marginBottom: "6px", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  Confirm Password
                </label>
                <input
                  className="auth-input"
                  type="password"
                  placeholder="Repeat your password"
                  value={confirm}
                  onChange={e => setConfirm(e.target.value)}
                  autoComplete="new-password"
                  disabled={loading}
                  style={{
                    borderColor: confirm && confirm !== password ? "#ef4444" : undefined,
                  }}
                />
                {confirm && confirm !== password && (
                  <div style={{ fontSize: "11px", color: "#ef4444", marginTop: "4px" }}>
                    Passwords don't match
                  </div>
                )}
              </div>
            )}

            {/* Error message */}
            {error && (
              <div style={{
                padding: "10px 12px",
                background: "#ef444418", border: "1px solid #ef444444",
                borderRadius: "6px", color: "#ef4444",
                fontSize: "12px", display: "flex", alignItems: "center", gap: "6px",
              }}>
                <span>⚠️</span> {error}
              </div>
            )}

            {/* Success message */}
            {success && (
              <div style={{
                padding: "10px 12px",
                background: "#22c55e18", border: "1px solid #22c55e44",
                borderRadius: "6px", color: "#22c55e",
                fontSize: "12px", display: "flex", alignItems: "center", gap: "6px",
              }}>
                <span>✅</span> {success}
              </div>
            )}

            {/* Submit button */}
            <button
              type="submit"
              className="auth-btn-primary"
              disabled={loading || (mode === "register" && confirm && confirm !== password)}
              style={{ marginTop: "4px" }}
            >
              {loading
                ? (mode === "login" ? "Signing in…" : "Creating account…")
                : (mode === "login" ? "Sign In →" : "Create Account →")
              }
            </button>

          </form>

          {/* Footer hint */}
          <div style={{ marginTop: "20px", textAlign: "center", fontSize: "11px", color: "var(--muted)" }}>
            {mode === "login"
              ? <>Don't have an account? <button onClick={() => setMode("register")} style={{ background: "none", border: "none", color: "#3b82f6", cursor: "pointer", fontSize: "11px", fontFamily: "inherit", fontWeight: "600" }}>Register here</button></>
              : <>Already have an account? <button onClick={() => setMode("login")} style={{ background: "none", border: "none", color: "#3b82f6", cursor: "pointer", fontSize: "11px", fontFamily: "inherit", fontWeight: "600" }}>Sign in</button></>
            }
          </div>
        </div>

        {/* Back to dashboard */}
        <div style={{ textAlign: "center", marginTop: "16px" }}>
          <button
            onClick={() => navigate("/")}
            style={{ background: "none", border: "none", color: "var(--muted)", cursor: "pointer", fontSize: "11px", fontFamily: "inherit" }}
          >
            ← Back to Dashboard
          </button>
        </div>

      </div>
    </div>
  );
}