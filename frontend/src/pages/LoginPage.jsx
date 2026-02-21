import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

const API = "http://localhost:8000";

export default function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  async function handleLogin(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const res = await fetch(`${API}/api/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      const data = await res.json();
      if (data.ok) {
        localStorage.setItem("argus_user", JSON.stringify(data.user));
        navigate("/");
      } else {
        setError(data.error || "Login failed");
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleRegister(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const res = await fetch(`${API}/api/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      const data = await res.json();
      if (data.ok) {
        localStorage.setItem("argus_user", JSON.stringify(data.user));
        navigate("/");
      } else {
        setError(data.error || "Register failed");
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 420, margin: "60px auto", padding: 24, border: "1px solid var(--border)", borderRadius: 12 }}>
      <h2 style={{ marginBottom: 12 }}>Sign in to ArgusAI</h2>
      <form onSubmit={handleLogin} style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <input placeholder="Username" value={username} onChange={e => setUsername(e.target.value)} style={{ padding: 10 }} />
        <input placeholder="Password" type="password" value={password} onChange={e => setPassword(e.target.value)} style={{ padding: 10 }} />
        {error && <div style={{ color: "#ef4444" }}>{error}</div>}
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn btn-primary" type="submit" disabled={loading} style={{ flex: 1 }}>{loading ? "Signing in..." : "Sign in"}</button>
          <button className="btn btn-ghost" onClick={handleRegister} disabled={loading} style={{ flex: 1 }}>Register</button>
        </div>
      </form>
    </div>
  );
}
