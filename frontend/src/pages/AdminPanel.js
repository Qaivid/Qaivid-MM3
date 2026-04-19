import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, Users, Film, BarChart3, Cpu, Shield, Trash2, RefreshCw, Check, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import { useAuth } from "@/hooks/useAuth";
import api from "@/lib/api";

const TABS = [
  { key: "overview",   label: "Overview",   icon: BarChart3 },
  { key: "providers",  label: "Providers",  icon: Cpu },
  { key: "users",      label: "Users",      icon: Users },
  { key: "projects",   label: "Projects",   icon: Film },
];

const QUALITY_COLOR = {
  highest: "text-[#D4AF37]",
  high:    "text-[#10B981]",
  good:    "text-[#8B5CF6]",
};

const SPEED_COLOR = {
  fast:   "text-[#10B981]",
  medium: "text-[#D4AF37]",
  slow:   "text-[#EF4444]",
};

function ProviderCard({ p, active, onSelect }) {
  const isActive = active === p.id;
  return (
    <button
      onClick={() => onSelect(p.id)}
      className={`w-full text-left p-4 rounded-xl border transition-all ${
        isActive
          ? "border-[#D4AF37] bg-[#D4AF37]/5"
          : "border-white/[0.06] bg-[#0A0A0C] hover:border-white/20"
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-sm font-semibold text-[#F3F4F6]">{p.label}</span>
            <span className="text-[10px] text-[#4a4a55]">by {p.vendor}</span>
          </div>
          <p className="text-xs text-[#71717A] mb-2">{p.notes}</p>
          <div className="flex items-center gap-3 text-[10px]">
            <span className={`font-medium ${QUALITY_COLOR[p.quality] || "text-[#71717A]"}`}>
              {p.quality} quality
            </span>
            <span className={`font-medium ${SPEED_COLOR[p.speed] || "text-[#71717A]"}`}>
              {p.speed}
            </span>
            {p.cost_per_image_usd && (
              <span className="text-[#4a4a55]">${p.cost_per_image_usd.toFixed(3)}/image</span>
            )}
          </div>
        </div>
        <div className={`flex-shrink-0 w-5 h-5 rounded-full border flex items-center justify-center mt-0.5 ${
          isActive ? "border-[#D4AF37] bg-[#D4AF37]" : "border-white/20"
        }`}>
          {isActive && <Check className="w-3 h-3 text-[#0A0A0C]" />}
        </div>
      </div>
    </button>
  );
}

function ProvidersTab() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(null);
  const [settings, setSettings] = useState({});
  const [imageProviders, setImageProviders] = useState([]);
  const [videoProviders, setVideoProviders] = useState([]);

  useEffect(() => {
    load();
  }, []);

  const load = async () => {
    setLoading(true);
    try {
      const data = await api.getAdminSettings();
      setSettings(data.settings || {});
      setImageProviders(data.image_providers || []);
      setVideoProviders(data.video_providers || []);
    } catch (e) {
      toast.error("Failed to load provider settings");
    }
    setLoading(false);
  };

  const save = async (key, value) => {
    setSaving(key);
    try {
      const res = await api.updateAdminSettings({ [key]: value });
      setSettings(res.settings || {});
      toast.success("Provider updated");
    } catch (e) {
      toast.error(e?.response?.data?.detail || "Failed to save");
    }
    setSaving(null);
  };

  if (loading) return (
    <div className="py-12 text-center text-[#4a4a55] text-sm">Loading provider settings…</div>
  );

  const sections = [
    {
      key: "image_provider_references",
      label: "Reference Image Generator",
      subtitle: "Generates character & location portraits. Runs once per project — quality matters most here.",
      providers: imageProviders,
    },
    {
      key: "image_provider_stills",
      label: "Shot Still Generator",
      subtitle: "Generates all 61 shot stills. Runs at scale — speed and cost matter here.",
      providers: imageProviders,
    },
    {
      key: "video_provider",
      label: "Video Generator",
      subtitle: "Animates shot stills into 2–15s video clips via AtlasCloud.",
      providers: videoProviders,
    },
  ];

  return (
    <div className="space-y-8" data-testid="admin-providers">
      {sections.map(section => (
        <div key={section.key}>
          <div className="mb-3">
            <h3 className="text-sm font-semibold text-[#F3F4F6]">{section.label}</h3>
            <p className="text-xs text-[#4a4a55] mt-0.5">{section.subtitle}</p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {section.providers.map(p => (
              <div key={p.id} className="relative">
                <ProviderCard
                  p={p}
                  active={settings[section.key]}
                  onSelect={(id) => save(section.key, id)}
                />
                {saving === section.key && settings[section.key] === p.id && (
                  <div className="absolute inset-0 bg-[#0A0A0C]/60 rounded-xl flex items-center justify-center">
                    <span className="text-[10px] text-[#D4AF37]">Saving…</span>
                  </div>
                )}
              </div>
            ))}
          </div>
          {section.key === "image_provider_stills" && (
            <div className="mt-2 flex items-start gap-2 p-3 bg-[#111116] rounded-lg border border-white/[0.04]">
              <AlertCircle className="w-3.5 h-3.5 text-[#D4AF37] flex-shrink-0 mt-0.5" />
              <p className="text-[11px] text-[#71717A]">
                Flux Schnell is <strong className="text-[#F3F4F6]">93% cheaper</strong> than GPT Image 1 for shot stills (~$0.18 vs $2.44 per project) with comparable quality for non-portrait shots. Add a <strong className="text-[#F3F4F6]">FAL_API_KEY</strong> in platform secrets to unlock Flux providers.
              </p>
            </div>
          )}
          {section.key === "video_provider" && (
            <div className="mt-2 flex items-start gap-2 p-3 bg-[#111116] rounded-lg border border-white/[0.04]">
              <AlertCircle className="w-3.5 h-3.5 text-[#8B5CF6] flex-shrink-0 mt-0.5" />
              <p className="text-[11px] text-[#71717A]">
                All video providers use your <strong className="text-[#F3F4F6]">ATLAS_CLOUD_API_KEY</strong>. Kling models produce higher-fidelity motion than WAN 2.6 but take longer to render.
              </p>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export default function AdminPanel() {
  const navigate = useNavigate();
  const { user, isAdmin } = useAuth();
  const [tab, setTab] = useState("overview");
  const [stats, setStats] = useState(null);
  const [users, setUsers] = useState([]);
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!isAdmin) { navigate("/app"); return; }
    loadData();
  }, [isAdmin, navigate]);

  const loadData = async () => {
    setLoading(true);
    try {
      const [s, u, p] = await Promise.all([
        api.getAdminStats(),
        api.getAdminUsers(),
        api.getAdminProjects(),
      ]);
      setStats(s);
      setUsers(u);
      setProjects(p);
    } catch (e) { toast.error("Failed to load admin data"); }
    setLoading(false);
  };

  const handleResetCredits = async (userId) => {
    try {
      await api.resetUserCredits(userId);
      toast.success("Credits reset");
      loadData();
    } catch { toast.error("Failed"); }
  };

  const handleDeleteUser = async (userId) => {
    if (!window.confirm("Delete this user and all their projects?")) return;
    try {
      await api.deleteAdminUser(userId);
      toast.success("User deleted");
      loadData();
    } catch { toast.error("Failed"); }
  };

  const handleUpdatePlan = async (userId, plan) => {
    try {
      await api.updateAdminUser(userId, { plan });
      toast.success(`Plan updated to ${plan}`);
      loadData();
    } catch { toast.error("Failed"); }
  };

  if (loading && tab !== "providers") {
    return (
      <div className="min-h-screen bg-[#0A0A0C] flex items-center justify-center">
        <p className="text-[#4a4a55]">Loading admin panel...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0A0A0C]">
      <header className="border-b border-[#1F1F24] bg-[#0A0A0C]/90 backdrop-blur-md sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" onClick={() => navigate("/app")} className="text-[#4a4a55] hover:text-white w-8 h-8" data-testid="admin-back">
              <ArrowLeft className="w-4 h-4" />
            </Button>
            <Shield className="w-5 h-5 text-[#D4AF37]" />
            <h1 className="font-heading text-lg font-semibold text-[#F3F4F6]" data-testid="admin-title">Admin Panel</h1>
          </div>
          <span className="text-xs text-[#4a4a55]">{user?.email}</span>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-6">
        <div className="flex gap-1 mb-6 bg-[#111116] rounded-lg p-1 w-fit" data-testid="admin-tabs">
          {TABS.map(t => (
            <button key={t.key} onClick={() => setTab(t.key)}
              className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md transition-all ${tab === t.key ? "bg-[#D4AF37] text-[#0A0A0C]" : "text-[#71717A] hover:text-white"}`}
              data-testid={`admin-tab-${t.key}`}>
              <t.icon className="w-4 h-4" /> {t.label}
            </button>
          ))}
        </div>

        {tab === "providers" && <ProvidersTab />}

        {tab === "overview" && stats && (
          <div className="space-y-6" data-testid="admin-overview">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[
                { label: "Total Users", value: stats.users, color: "text-[#D4AF37]" },
                { label: "Total Projects", value: stats.projects, color: "text-[#8B5CF6]" },
                { label: "Pro Users", value: stats.plans?.pro || 0, color: "text-[#10B981]" },
                { label: "Studio Users", value: stats.plans?.studio || 0, color: "text-[#C85A17]" },
              ].map(s => (
                <div key={s.label} className="bg-[#111116] border border-white/[0.06] rounded-xl p-5">
                  <p className="text-[10px] uppercase tracking-widest text-[#4a4a55] mb-1">{s.label}</p>
                  <p className={`text-3xl font-heading font-semibold ${s.color}`}>{s.value}</p>
                </div>
              ))}
            </div>
            <div className="bg-[#111116] border border-white/[0.06] rounded-xl p-5">
              <h3 className="text-sm font-semibold text-[#F3F4F6] mb-3">Plan Distribution</h3>
              <div className="flex gap-3">
                {Object.entries(stats.plans || {}).map(([plan, count]) => (
                  <div key={plan} className="flex-1 bg-[#0A0A0C] rounded-lg p-3 text-center">
                    <p className="text-xs text-[#71717A] capitalize">{plan}</p>
                    <p className="text-xl font-semibold text-[#F3F4F6]">{count}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {tab === "users" && (
          <div className="space-y-3" data-testid="admin-users">
            {users.map(u => (
              <div key={u.id} className="bg-[#111116] border border-white/[0.06] rounded-xl p-4 flex items-center gap-4">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm font-medium text-[#F3F4F6] truncate">{u.name || u.email}</span>
                    <Badge className={`text-[9px] ${u.role === "admin" ? "bg-[#D4AF37]/10 text-[#D4AF37]" : "bg-[#27272A] text-[#71717A]"}`}>{u.role}</Badge>
                    <Badge className="bg-[#8B5CF6]/10 text-[#8B5CF6] text-[9px]">{u.plan}</Badge>
                  </div>
                  <p className="text-xs text-[#4a4a55]">{u.email} {'\u2022'} {u.project_count} projects {'\u2022'} {Math.round(u.credit_balance || 0)} credits</p>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <select
                    value={u.plan} onChange={e => handleUpdatePlan(u.id, e.target.value)}
                    className="bg-[#0A0A0C] border border-white/10 rounded-lg text-xs text-[#A1A1AA] px-2 py-1.5"
                    data-testid={`plan-select-${u.id}`}
                  >
                    <option value="free">Free</option>
                    <option value="starter">Starter</option>
                    <option value="pro">Pro</option>
                    <option value="studio">Studio</option>
                  </select>
                  <Button variant="ghost" size="sm" onClick={() => handleResetCredits(u.id)}
                    className="text-[#D4AF37] hover:text-white text-xs h-8" data-testid={`reset-credits-${u.id}`}>
                    <RefreshCw className="w-3 h-3 mr-1" /> Reset
                  </Button>
                  {u.role !== "admin" && (
                    <Button variant="ghost" size="sm" onClick={() => handleDeleteUser(u.id)}
                      className="text-[#EF4444] hover:text-white text-xs h-8" data-testid={`delete-user-${u.id}`}>
                      <Trash2 className="w-3 h-3" />
                    </Button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {tab === "projects" && (
          <div className="space-y-3" data-testid="admin-projects">
            {projects.map(p => (
              <div key={p.id} className="bg-[#111116] border border-white/[0.06] rounded-xl p-4 flex items-center gap-4">
                <div className="flex-1 min-w-0">
                  <h3 className="text-sm font-medium text-[#F3F4F6] truncate">{p.name}</h3>
                  <p className="text-xs text-[#4a4a55]">{p.owner_email || "Unknown"} {'\u2022'} {p.input_mode} {'\u2022'} {p.status}</p>
                </div>
                <span className="text-[9px] text-[#4a4a55] shrink-0">{new Date(p.created_at).toLocaleDateString()}</span>
              </div>
            ))}
            {projects.length === 0 && <p className="text-[#4a4a55] text-center py-8">No projects yet</p>}
          </div>
        )}
      </div>
    </div>
  );
}
