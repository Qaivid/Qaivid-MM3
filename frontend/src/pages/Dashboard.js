import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Plus, Film, Trash2, ChevronRight, Video, Sparkles, Layers, Eye, Clapperboard, Image, Upload, Wand2, ArrowRight, Camera, Clock, Music, Users, Brain, Mail, Settings, Shield, LogOut, CreditCard } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { toast } from "sonner";
import api from "@/lib/api";
import { useAuth } from "@/hooks/useAuth";

const STATUS_CONFIG = {
  draft: { label: "New", color: "text-[#71717A]", bg: "bg-[#71717A]/10", step: 0 },
  input_added: { label: "Content Ready", color: "text-[#3B82F6]", bg: "bg-[#3B82F6]/10", step: 1 },
  interpreting: { label: "Analyzing...", color: "text-[#F59E0B]", bg: "bg-[#F59E0B]/10", step: 1 },
  interpreted: { label: "Storyboard Ready", color: "text-[#8B5CF6]", bg: "bg-[#8B5CF6]/10", step: 2 },
  scenes_built: { label: "Scenes Locked", color: "text-[#D4AF37]", bg: "bg-[#D4AF37]/10", step: 3 },
  shots_built: { label: "Shots Planned", color: "text-[#C85A17]", bg: "bg-[#C85A17]/10", step: 4 },
  prompts_ready: { label: "Ready to Generate", color: "text-[#10B981]", bg: "bg-[#10B981]/10", step: 5 },
  complete: { label: "Complete", color: "text-[#10B981]", bg: "bg-[#10B981]/10", step: 6 },
};
const TOTAL_STEPS = 6;

const CONTENT_TYPES = [
  { label: "Music Videos" }, { label: "Lyric Videos" }, { label: "Short Films" },
  { label: "Ghazals" }, { label: "Qawwalis" }, { label: "Spoken Word" },
  { label: "Brand Films" }, { label: "Documentaries" }, { label: "YouTube Shorts" },
  { label: "Poetry Films" },
];

const TYPE_LABEL = { song: "Music Video", poem: "Poem Film", ghazal: "Ghazal Film", qawwali: "Qawwali", script: "Short Film", story: "Narration", ad: "Brand Film", documentary: "Documentary" };

export default function Dashboard() {
  const navigate = useNavigate();
  const { user, logout, isAdmin } = useAuth();
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [newProject, setNewProject] = useState({ name: "", description: "", input_mode: "song" });

  useEffect(() => { loadProjects(); }, []);
  const loadProjects = async () => {
    try { setProjects(await api.listProjects()); } catch { /* ok */ }
    finally { setLoading(false); }
  };
  const handleCreate = async () => {
    if (!newProject.name.trim()) { toast.error("Name required"); return; }
    try {
      const p = await api.createProject(newProject);
      setShowCreate(false); setNewProject({ name: "", description: "", input_mode: "song" });
      navigate(`/project/${p.id}`);
    } catch { toast.error("Failed"); }
  };
  const handleDelete = async (e, id) => {
    e.stopPropagation();
    if (!window.confirm("Delete this project?")) return;
    try { await api.deleteProject(id); setProjects(p => p.filter(x => x.id !== id)); toast.success("Deleted"); }
    catch { toast.error("Failed"); }
  };

  return (
    <div className="min-h-screen bg-[#0A0A0C]">
      {/* ─── Navbar ─── */}
      <nav className="fixed top-0 left-0 right-0 z-50 border-b border-white/[0.06] bg-[#0A0A0C]/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#D4AF37] to-[#C85A17] flex items-center justify-center">
              <Film className="w-4 h-4 text-[#0A0A0C]" />
            </div>
            <span className="font-heading text-xl font-semibold text-[#F3F4F6] tracking-tight" data-testid="app-title">Qaivid</span>
            <Badge className="bg-[#D4AF37]/10 text-[#D4AF37] border border-[#D4AF37]/20 text-[9px] tracking-wider">2.0</Badge>
          </div>
          <div className="hidden md:flex items-center gap-6 text-sm">
            <a href="#projects" className="text-[#A1A1AA] hover:text-white transition-colors">Projects</a>
            <a href="#how-it-works" className="text-[#4a4a55] hover:text-[#A1A1AA] transition-colors">How It Works</a>
            <a href="#features" className="text-[#4a4a55] hover:text-[#A1A1AA] transition-colors">Features</a>
          </div>
          <div className="flex items-center gap-2">
            {/* Credits */}
            {user && (
              <span className="text-xs text-[#D4AF37] bg-[#D4AF37]/10 px-2.5 py-1 rounded-lg border border-[#D4AF37]/20 font-mono" data-testid="credit-balance">
                <CreditCard className="w-3 h-3 inline mr-1" />{Math.round(user.credit_balance || 0)}
              </span>
            )}
            {/* Admin */}
            {isAdmin && (
              <button onClick={() => navigate("/admin")}
                className="w-9 h-9 rounded-lg border border-white/[0.06] bg-white/[0.02] flex items-center justify-center text-[#71717A] hover:text-[#D4AF37] hover:border-[#D4AF37]/30 transition-all"
                data-testid="admin-btn" title="Admin Panel">
                <Shield className="w-4 h-4" />
              </button>
            )}
            {/* Logout */}
            <button onClick={async () => { await logout(); navigate("/auth"); }}
              className="w-9 h-9 rounded-lg border border-white/[0.06] bg-white/[0.02] flex items-center justify-center text-[#71717A] hover:text-red-400 hover:border-red-400/30 transition-all"
              data-testid="logout-btn" title="Sign Out">
              <LogOut className="w-4 h-4" />
            </button>
            <Dialog open={showCreate} onOpenChange={setShowCreate}>
            <DialogTrigger asChild>
              <Button data-testid="create-project-btn" className="bg-[#D4AF37] text-[#0A0A0C] hover:bg-[#F1C40F] font-semibold gap-2 h-9 text-sm">
                <Plus className="w-4 h-4" /> New Project
              </Button>
            </DialogTrigger>
            <DialogContent className="bg-[#141417] border border-[#27272A] text-[#F3F4F6] sm:max-w-lg">
              <DialogHeader>
                <DialogTitle className="font-heading text-2xl text-[#F3F4F6]">New Video Project</DialogTitle>
                <p className="text-sm text-[#71717A] mt-1">Qaivid will build the entire cinematic pipeline from your source content.</p>
              </DialogHeader>
              <div className="space-y-4 mt-3">
                <div>
                  <label className="text-[10px] tracking-widest uppercase text-[#71717A] font-semibold mb-1.5 block">Project Title</label>
                  <Input data-testid="project-name-input" value={newProject.name} onChange={e => setNewProject(p => ({...p, name: e.target.value}))}
                    placeholder='e.g. "Birha" — Punjabi Music Video' className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] placeholder:text-[#3f3f46] h-11" />
                </div>
                <div>
                  <label className="text-[10px] tracking-widest uppercase text-[#71717A] font-semibold mb-1.5 block">Brief</label>
                  <Textarea data-testid="project-description-input" value={newProject.description} onChange={e => setNewProject(p => ({...p, description: e.target.value}))}
                    placeholder="Describe the video concept..." className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] placeholder:text-[#3f3f46] min-h-[60px]" rows={2} />
                </div>
                <div>
                  <label className="text-[10px] tracking-widest uppercase text-[#71717A] font-semibold mb-1.5 block">Source Type</label>
                  <Select value={newProject.input_mode} onValueChange={v => setNewProject(p => ({...p, input_mode: v}))}>
                    <SelectTrigger data-testid="input-mode-select" className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] h-11"><SelectValue /></SelectTrigger>
                    <SelectContent className="bg-[#1E1E24] border-[#27272A]">
                      {[{v:"song",l:"Song / Music Video"},{v:"poem",l:"Poem / Spoken Word"},{v:"ghazal",l:"Ghazal"},{v:"qawwali",l:"Qawwali / Devotional"},{v:"script",l:"Script / Short Film"},{v:"story",l:"Story / Narration"},{v:"ad",l:"Ad / Brand Film"},{v:"documentary",l:"Documentary"}].map(m => (
                        <SelectItem key={m.v} value={m.v} className="text-[#F3F4F6] focus:bg-[#27272E] focus:text-white">{m.l}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <Button data-testid="confirm-create-btn" onClick={handleCreate} className="w-full bg-[#D4AF37] text-[#0A0A0C] hover:bg-[#F1C40F] font-semibold h-11">Create Project</Button>
              </div>
            </DialogContent>
          </Dialog>
          </div>
        </div>
      </nav>

      {/* ─── Hero ─── */}
      <section className="relative pt-32 pb-16 overflow-hidden">
        <div className="absolute inset-0">
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_30%_40%,rgba(212,175,55,0.08)_0%,transparent_55%)]" />
          <div className="absolute top-0 right-0 w-2/3 h-full bg-[radial-gradient(ellipse_at_70%_30%,rgba(200,90,23,0.06)_0%,transparent_50%)]" />
        </div>
        <div className="relative max-w-7xl mx-auto px-6">
          <div className="max-w-3xl">
            <Badge className="mb-6 bg-[#D4AF37]/10 text-[#D4AF37] border border-[#D4AF37]/20 hover:bg-[#D4AF37]/15" data-testid="badge-hero">
              <Sparkles className="w-3 h-3 mr-1.5" /> End-to-End AI Video Production
            </Badge>
            <h1 className="font-heading text-6xl font-semibold text-[#F3F4F6] leading-[1.05] mb-6 tracking-tight">
              From lyrics and stories<br/>to finished cinematic video.
            </h1>
            <p className="text-[#71717A] text-xl leading-relaxed mb-8 max-w-2xl">
              Qaivid understands the meaning, culture, and emotion in your content — then builds intelligent storyboards, shot plans, and generation-ready prompts for any AI video model.
            </p>
            <div className="flex gap-3 mb-12">
              <Button onClick={() => setShowCreate(true)} size="lg" className="bg-gradient-to-r from-[#D4AF37] to-[#C85A17] hover:from-[#F1C40F] hover:to-[#D4AF37] text-[#0A0A0C] font-semibold shadow-lg shadow-[#D4AF37]/15 h-12 px-8" data-testid="hero-cta">
                <Clapperboard className="w-4 h-4 mr-2" /> Start a Project
              </Button>
              <Button variant="outline" size="lg" className="border-[#27272A] text-[#A1A1AA] hover:text-white hover:bg-[#141417] h-12 px-6" onClick={() => document.getElementById('how-it-works')?.scrollIntoView({behavior:'smooth'})} data-testid="hero-secondary">
                See How It Works
              </Button>
            </div>
          </div>
          {/* Content type pills */}
          <div className="flex flex-wrap gap-2">
            {CONTENT_TYPES.map(({ label }) => (
              <span key={label} className="px-4 py-1.5 rounded-full bg-white/[0.03] border border-white/[0.06] text-sm text-[#71717A] hover:border-[#D4AF37]/30 hover:text-[#D4AF37] transition-all cursor-default">
                {label}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* ─── How It Works ─── */}
      <section id="how-it-works" className="relative py-24 px-6">
        <div className="absolute inset-0"><div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-[radial-gradient(ellipse_at_center,rgba(212,175,55,0.04)_0%,transparent_70%)]" /></div>
        <div className="relative max-w-6xl mx-auto">
          <h2 className="font-heading text-4xl font-semibold text-center text-[#F3F4F6] mb-3">From Content to Cinema</h2>
          <p className="text-center text-[#71717A] mb-16 max-w-2xl mx-auto">Qaivid handles every stage of video production — just bring your lyrics, script, or story</p>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-10">
            {[
              { icon: Upload, step: "1", title: "Add Your Source Content", desc: "Paste lyrics, a script, poem, story, or SRT subtitles. Qaivid auto-detects language, structure, and content type." },
              { icon: Brain, step: "2", title: "Context Intelligence Runs", desc: "One deep AI call interprets meaning, culture, emotion, and narrative structure. Culture packs add South Asian, Urdu, Punjabi intelligence automatically." },
              { icon: Clapperboard, step: "3", title: "Storyboard & Shot Plan Built", desc: "Scenes are designed from narrative structure. Shots are planned with cinematography, lighting, camera, and pacing — all deterministic, no extra AI calls." },
              { icon: Video, step: "4", title: "Generation-Ready Prompts", desc: "Prompts adapted for 11 models (Kling, Runway, Midjourney, Flux...) with character and environment references injected. Export to JSON, CSV, or storyboard." },
            ].map(({ icon: Icon, step, title, desc }) => (
              <div key={step} className="text-center group">
                <div className="w-16 h-16 rounded-2xl bg-[#D4AF37]/10 border border-[#D4AF37]/15 flex items-center justify-center mx-auto mb-5 group-hover:border-[#D4AF37]/30 group-hover:bg-[#D4AF37]/15 transition-all">
                  <Icon className="w-7 h-7 text-[#D4AF37]" />
                </div>
                <div className="text-[10px] text-[#D4AF37] font-bold mb-2 tracking-[0.2em] uppercase">Step {step}</div>
                <h3 className="font-semibold mb-2 text-[#F3F4F6]">{title}</h3>
                <p className="text-sm text-[#71717A] leading-relaxed">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── Features ─── */}
      <section id="features" className="relative py-24 px-6">
        <div className="absolute inset-0"><div className="absolute top-0 left-1/3 w-[500px] h-[500px] bg-[radial-gradient(ellipse_at_center,rgba(200,90,23,0.03)_0%,transparent_70%)]" /></div>
        <div className="relative max-w-6xl mx-auto">
          <h2 className="font-heading text-4xl font-semibold text-center text-[#F3F4F6] mb-3">Everything Included</h2>
          <p className="text-center text-[#71717A] mb-16 max-w-2xl mx-auto">Every creative and technical decision handled — so you focus on the vision, not the pipeline</p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              { icon: Brain, title: "Context Intelligence Engine", desc: "Deep meaning analysis understands metaphor, culture, emotion, and narrative mode. 7 culture packs cover Punjabi, Urdu, Hindi, and diaspora content automatically." },
              { icon: Users, title: "Character & Environment Profiles", desc: "Define your cast and locations once. Qaivid injects character appearance and environment details into every shot prompt for visual consistency." },
              { icon: Clock, title: "Content-Aware Pacing", desc: "12 pacing profiles (song, ghazal, qawwali, ad, documentary...) control shot duration, density, and rhythm — matched to the content type automatically." },
              { icon: Camera, title: "Cinematography-Aware Shots", desc: "Every shot gets camera type, height, movement, lighting, and motion constraints. One shot = one intention. No multi-phase prompts." },
              { icon: Layers, title: "Continuity Tracking", desc: "Subject tracking, motif recurrence, location continuity, emotional flow, and temporal logic are monitored across every scene and shot." },
              { icon: Image, title: "11 Generation Model Adapters", desc: "Prompts adapted for Kling, Runway, Midjourney, DALL-E, Flux, SDXL, Wan 2.6, Veo, Pika — each with model-specific constraints and risk notes." },
            ].map(({ icon: Icon, title, desc }) => (
              <div key={title} className="rounded-xl bg-white/[0.02] border border-white/[0.06] p-6 hover:border-[#D4AF37]/20 hover:bg-[#D4AF37]/[0.02] transition-all group">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-[#D4AF37]/15 to-[#C85A17]/10 border border-[#D4AF37]/15 flex items-center justify-center mb-4 group-hover:from-[#D4AF37]/25 group-hover:to-[#C85A17]/15 transition-all">
                  <Icon className="w-6 h-6 text-[#D4AF37]" />
                </div>
                <h3 className="font-semibold mb-2 text-[#F3F4F6]">{title}</h3>
                <p className="text-sm text-[#71717A] leading-relaxed">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── Projects ─── */}
      <section id="projects" className="relative py-16 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between mb-8">
            <div>
              <h2 className="font-heading text-3xl font-semibold text-[#F3F4F6]">My Projects</h2>
              <p className="text-[#4a4a55] mt-1">Your AI video production projects</p>
            </div>
            <Button onClick={() => setShowCreate(true)} className="bg-[#D4AF37] text-[#0A0A0C] hover:bg-[#F1C40F] font-semibold gap-2" data-testid="projects-create-btn">
              <Plus className="w-4 h-4" /> New Project
            </Button>
          </div>
          {loading ? (
            <div className="py-16 text-center text-[#4a4a55]">Loading projects...</div>
          ) : projects.length === 0 ? (
            <div className="rounded-xl bg-white/[0.02] border border-white/[0.06] py-16 text-center" data-testid="empty-state">
              <Film className="w-12 h-12 text-[#27272A] mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-[#A1A1AA] mb-2">No projects yet</h3>
              <p className="text-[#4a4a55] mb-6 text-sm">Upload your first track or script to start producing</p>
              <Button onClick={() => setShowCreate(true)} className="bg-[#D4AF37] text-[#0A0A0C] hover:bg-[#F1C40F] font-semibold gap-2" data-testid="empty-create-btn">
                <Plus className="w-4 h-4" /> Start Your First Project
              </Button>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
              {projects.map((project, idx) => {
                const sc = STATUS_CONFIG[project.status] || STATUS_CONFIG.draft;
                const progress = (sc.step / TOTAL_STEPS) * 100;
                return (
                  <div key={project.id} data-testid={`project-card-${idx}`} onClick={() => navigate(`/project/${project.id}`)}
                    className="group cursor-pointer bg-[#141417] border border-[#1F1F24] rounded-xl overflow-hidden transition-all duration-200 hover:border-[#D4AF37]/20 hover:translate-y-[-1px] animate-fade-up"
                    style={{ animationDelay: `${idx * 50}ms` }}>
                    <div className="h-0.5 bg-[#1E1E24]"><div className="h-full bg-gradient-to-r from-[#D4AF37] to-[#C85A17] transition-all" style={{width:`${Math.max(progress,4)}%`}} /></div>
                    <div className="p-5">
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex-1 min-w-0">
                          <h3 className="font-heading text-lg font-semibold text-[#F3F4F6] truncate group-hover:text-[#D4AF37] transition-colors">{project.name}</h3>
                          <p className="text-xs text-[#4a4a55] mt-0.5 truncate">{project.description || "No brief"}</p>
                        </div>
                        <button data-testid={`delete-project-${idx}`} onClick={e => handleDelete(e, project.id)}
                          className="ml-3 p-1 rounded-lg text-[#27272A] hover:text-[#EF4444] hover:bg-[#27272E] transition-all opacity-0 group-hover:opacity-100">
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="text-[9px] tracking-widest uppercase font-semibold text-[#4a4a55] bg-[#0A0A0C] px-2 py-0.5 rounded border border-[#1F1F24]">
                            {TYPE_LABEL[project.input_mode] || project.input_mode}
                          </span>
                          <span className={`text-[9px] tracking-wider uppercase font-semibold px-2 py-0.5 rounded ${sc.bg} ${sc.color}`}>{sc.label}</span>
                        </div>
                        <ChevronRight className="w-4 h-4 text-[#27272A] group-hover:text-[#D4AF37] transition-colors" />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </section>

      {/* ─── CTA ─── */}
      <section className="relative py-24 px-6">
        <div className="absolute inset-0"><div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[300px] bg-[radial-gradient(ellipse_at_center,rgba(212,175,55,0.06)_0%,transparent_70%)]" /></div>
        <div className="relative max-w-3xl mx-auto text-center">
          <h2 className="font-heading text-4xl font-semibold text-[#F3F4F6] mb-4">Your Next Video Starts Here</h2>
          <p className="text-[#71717A] mb-8 text-lg">Paste your lyrics or script and let Qaivid direct, plan, and prepare every shot — end to end.</p>
          <Button onClick={() => setShowCreate(true)} size="lg" className="bg-gradient-to-r from-[#D4AF37] via-[#C85A17] to-[#D4AF37] hover:from-[#F1C40F] hover:to-[#F1C40F] text-[#0A0A0C] font-semibold shadow-lg shadow-[#D4AF37]/15 h-12 px-8" data-testid="cta-start">
            Start a New Project <ArrowRight className="w-4 h-4 ml-1.5" />
          </Button>
        </div>
      </section>

      {/* ─── Footer ─── */}
      <footer className="border-t border-white/[0.04] bg-[#08080B]" data-testid="footer">
        <div className="max-w-7xl mx-auto px-6 py-12">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center gap-2 mb-3">
                <div className="w-7 h-7 rounded-md bg-gradient-to-br from-[#D4AF37] to-[#C85A17] flex items-center justify-center"><Film className="w-3.5 h-3.5 text-[#0A0A0C]" /></div>
                <span className="font-heading text-lg font-semibold text-[#F3F4F6]">Qaivid</span>
              </div>
              <p className="text-sm text-[#4a4a55]">The AI video production system that understands meaning before it creates. Context-first. Culture-aware. Cinema-ready.</p>
            </div>
            <div>
              <h3 className="font-semibold text-[10px] uppercase tracking-[0.15em] text-[#71717A] mb-3">Product</h3>
              <ul className="space-y-2 text-sm text-[#4a4a55]">
                <li className="hover:text-[#A1A1AA] transition-colors cursor-pointer">How It Works</li>
                <li className="hover:text-[#A1A1AA] transition-colors cursor-pointer">Features</li>
                <li className="hover:text-[#A1A1AA] transition-colors cursor-pointer">Pricing</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold text-[10px] uppercase tracking-[0.15em] text-[#71717A] mb-3">Capabilities</h3>
              <ul className="space-y-2 text-sm text-[#4a4a55]">
                <li>Context Intelligence</li>
                <li>7 Culture Packs</li>
                <li>11 Model Adapters</li>
                <li>Continuity Tracking</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold text-[10px] uppercase tracking-[0.15em] text-[#71717A] mb-3">Support</h3>
              <ul className="space-y-2 text-sm text-[#4a4a55]">
                <li className="flex items-center gap-1.5"><Mail className="w-3 h-3" /> support@qaivid.com</li>
                <li className="hover:text-[#A1A1AA] transition-colors cursor-pointer">Contact</li>
                <li className="hover:text-[#A1A1AA] transition-colors cursor-pointer">Privacy Policy</li>
                <li className="hover:text-[#A1A1AA] transition-colors cursor-pointer">Terms of Service</li>
              </ul>
            </div>
          </div>
          <div className="border-t border-white/[0.04] mt-8 pt-6 flex items-center justify-between">
            <p className="text-xs text-[#3f3f46]">&copy; {new Date().getFullYear()} Qaivid. All rights reserved.</p>
            <p className="text-xs text-[#3f3f46]">From Content to Cinema, Powered by AI</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
