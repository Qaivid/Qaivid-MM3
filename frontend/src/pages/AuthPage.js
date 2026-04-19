import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Film, LogIn, UserPlus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import { useAuth } from "@/hooks/useAuth";

export default function AuthPage() {
  const navigate = useNavigate();
  const { user, login, register } = useAuth();
  const [mode, setMode] = useState("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // If already authenticated (e.g. user hit "back" after logging in),
  // bounce them straight to their landing page.
  useEffect(() => {
    if (user && typeof user === "object") {
      navigate(user.role === "admin" ? "/admin" : "/app", { replace: true });
    }
  }, [user, navigate]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      let result;
      if (mode === "login") {
        result = await login(email, password);
      } else {
        result = await register(email, password, name);
      }
      navigate(result?.role === "admin" ? "/admin" : "/app", { replace: true });
    } catch (err) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : detail?.msg || err.message || "Failed");
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-[#0A0A0C] flex items-center justify-center px-4">
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_30%_40%,rgba(212,175,55,0.06)_0%,transparent_55%)]" />
      </div>
      <div className="relative w-full max-w-md">
        {/* Logo */}
        <div className="flex items-center justify-center gap-2.5 mb-8">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#D4AF37] to-[#C85A17] flex items-center justify-center">
            <Film className="w-5 h-5 text-[#0A0A0C]" />
          </div>
          <span className="font-heading text-2xl font-semibold text-[#F3F4F6] tracking-tight">Qaivid</span>
          <Badge className="bg-[#D4AF37]/10 text-[#D4AF37] border border-[#D4AF37]/20 text-[9px]">2.0</Badge>
        </div>

        <div className="bg-[#111116] border border-white/10 rounded-xl p-8" data-testid="auth-form">
          {/* Tabs */}
          <div className="flex gap-1 mb-6 bg-[#0A0A0C] rounded-lg p-1">
            <button
              onClick={() => { setMode("login"); setError(""); }}
              className={`flex-1 py-2 text-sm font-medium rounded-md transition-all ${mode === "login" ? "bg-[#D4AF37] text-[#0A0A0C]" : "text-[#71717A] hover:text-white"}`}
              data-testid="login-tab"
            >
              <LogIn className="w-4 h-4 inline mr-1.5" />Sign In
            </button>
            <button
              onClick={() => { setMode("register"); setError(""); }}
              className={`flex-1 py-2 text-sm font-medium rounded-md transition-all ${mode === "register" ? "bg-[#D4AF37] text-[#0A0A0C]" : "text-[#71717A] hover:text-white"}`}
              data-testid="register-tab"
            >
              <UserPlus className="w-4 h-4 inline mr-1.5" />Sign Up
            </button>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {mode === "register" && (
              <div>
                <label className="text-[10px] tracking-widest uppercase text-[#71717A] font-semibold mb-1.5 block">Name</label>
                <Input
                  data-testid="name-input"
                  value={name} onChange={e => setName(e.target.value)}
                  placeholder="Your name"
                  className="bg-[#0A0A0C] border-white/10 text-white placeholder:text-zinc-600 h-11"
                />
              </div>
            )}
            <div>
              <label className="text-[10px] tracking-widest uppercase text-[#71717A] font-semibold mb-1.5 block">Email</label>
              <Input
                data-testid="email-input"
                type="email" value={email} onChange={e => setEmail(e.target.value)}
                placeholder="you@example.com" required
                className="bg-[#0A0A0C] border-white/10 text-white placeholder:text-zinc-600 h-11"
              />
            </div>
            <div>
              <label className="text-[10px] tracking-widest uppercase text-[#71717A] font-semibold mb-1.5 block">Password</label>
              <Input
                data-testid="password-input"
                type="password" value={password} onChange={e => setPassword(e.target.value)}
                placeholder="Min 6 characters" required minLength={6}
                className="bg-[#0A0A0C] border-white/10 text-white placeholder:text-zinc-600 h-11"
              />
            </div>

            {error && (
              <div className="bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2 text-sm text-red-400" data-testid="auth-error">
                {error}
              </div>
            )}

            <Button
              type="submit" disabled={loading}
              className="w-full bg-gradient-to-r from-[#D4AF37] to-[#C85A17] hover:from-[#F1C40F] hover:to-[#D4AF37] text-[#0A0A0C] font-semibold h-11"
              data-testid="auth-submit-btn"
            >
              {loading ? "..." : mode === "login" ? "Sign In" : "Create Account"}
            </Button>
          </form>
        </div>

        <p className="text-center text-[#4a4a55] text-xs mt-6">
          From Content to Cinema, Powered by AI
        </p>
      </div>
    </div>
  );
}
