import { Link } from "react-router-dom";
import { useAuth } from "@/hooks/useAuth";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Sparkles,
  ArrowRight,
  Film,
  Upload,
  Wand2,
  Layers,
  Camera,
  Users,
  Globe,
  Music,
  Clock,
  FileDown,
  Check,
  X,
} from "lucide-react";

const FORMATS = [
  { icon: "🎵", label: "Music Videos" },
  { icon: "🎬", label: "Short Films" },
  { icon: "✨", label: "Lyric Videos" },
  { icon: "📱", label: "YouTube Shorts" },
  { icon: "💼", label: "Product Ads" },
  { icon: "🎙️", label: "Documentaries" },
  { icon: "🎧", label: "Podcast Visuals" },
  { icon: "📖", label: "Book Trailers" },
  { icon: "🌀", label: "AI Art Films" },
];

const STEPS = [
  {
    icon: Upload,
    step: "01",
    title: "Bring your source",
    desc: "Upload audio, paste lyrics, or drop in a script. Qaivid Core auto-detects language, structure and intent before any visuals are imagined.",
  },
  {
    icon: Wand2,
    step: "02",
    title: "AI interprets your vision",
    desc: "A full director's brief is generated — tone, palette, characters, environments, narrative arc — adapted to your culture and language.",
  },
  {
    icon: Layers,
    step: "03",
    title: "Scene & shot planning",
    desc: "Your work is broken down into scenes, then individual shots, with framing, motion and timing — all synced to your audio.",
  },
  {
    icon: Film,
    step: "04",
    title: "Render & assemble",
    desc: "Locked character references keep every shot consistent. Each shot is rendered to video and stitched into a final cut, ready to export.",
  },
];

const FEATURES = [
  {
    icon: Wand2,
    title: "AI Director's Brief",
    desc: "A single source of creative truth: visual style, narrative arc, characters, environments and props — generated in seconds, editable end-to-end.",
  },
  {
    icon: Users,
    title: "Character Consistency",
    desc: "Lock a face, costume and look once. Every shot inherits the same identity through reference imagery — no more drifting characters.",
  },
  {
    icon: Globe,
    title: "Culture & Language Aware",
    desc: "Native handling of Punjabi, Urdu, Hindi, English and more. Culture packs shape visual references authentically — not as Western defaults.",
  },
  {
    icon: Clock,
    title: "Audio-Synced Storyboarding",
    desc: "Verses, choruses, beats and silences are mapped to visual timing automatically. Every shot lands on the right moment of your audio.",
  },
  {
    icon: Camera,
    title: "Scene → Shot → Prompt Pipeline",
    desc: "A real production hierarchy, not a single prompt. Edit any layer — brief, scene, shot, or prompt — and re-render only what changed.",
  },
  {
    icon: FileDown,
    title: "Export-Ready Output",
    desc: "Reference packs, storyboard exports and final-cut video. Production-grade artefacts you can hand to a team or publish directly.",
  },
];

const COMPARE = [
  ["One prompt, one clip", "Full pipeline: brief → scenes → shots → render"],
  ["Inconsistent characters between shots", "Locked character & environment references"],
  ["English-first, Western defaults", "Multilingual, culture-aware by design"],
  ["Black-box outputs", "Editable at every stage — brief, scene, shot, prompt"],
  ["Demo-quality clips", "Built for finished, exportable films"],
];

function NavBar({ user }) {
  return (
    <nav className="sticky top-0 z-50 backdrop-blur-xl bg-[#0A0A0C]/70 border-b border-white/[0.06]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2.5 group">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-[#D4AF37] to-[#C85A17] flex items-center justify-center shadow-lg shadow-[#D4AF37]/20">
            <Film className="w-4.5 h-4.5 text-black" />
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className="text-white font-semibold tracking-tight">Qaivid</span>
            <span className="text-[10px] uppercase tracking-[0.18em] text-[#D4AF37]/80 font-medium">Core 2.0</span>
          </div>
        </Link>
        <div className="hidden md:flex items-center gap-7 text-sm text-zinc-400">
          <a href="#how" className="hover:text-white transition-colors">How it works</a>
          <a href="#features" className="hover:text-white transition-colors">Features</a>
          <a href="#why" className="hover:text-white transition-colors">Why Qaivid</a>
        </div>
        <div className="flex items-center gap-2">
          {user ? (
            <Link to="/app">
              <Button size="sm" className="bg-gradient-to-r from-[#D4AF37] to-[#C85A17] text-black hover:opacity-90 font-medium">
                Open Dashboard <ArrowRight className="w-3.5 h-3.5 ml-1" />
              </Button>
            </Link>
          ) : (
            <>
              <Link to="/auth" className="text-sm text-zinc-300 hover:text-white px-3 py-1.5">Sign in</Link>
              <Link to="/auth">
                <Button size="sm" className="bg-gradient-to-r from-[#D4AF37] to-[#C85A17] text-black hover:opacity-90 font-medium">
                  Get Started
                </Button>
              </Link>
            </>
          )}
        </div>
      </div>
    </nav>
  );
}

export default function LandingPage() {
  const { user } = useAuth();
  const ctaTo = user ? "/app" : "/auth";
  const ctaLabel = user ? "Open Dashboard" : "Start Creating";

  return (
    <div className="min-h-screen bg-[#0A0A0C] text-zinc-100 overflow-x-hidden">
      <NavBar user={user} />

      {/* HERO */}
      <section className="relative pt-16 pb-24 px-4">
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_30%_20%,rgba(139,92,246,0.18)_0%,transparent_55%)]" />
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_75%_30%,rgba(212,175,55,0.10)_0%,transparent_55%)]" />
          <div className="absolute bottom-0 left-0 w-full h-1/2 bg-[radial-gradient(ellipse_at_50%_90%,rgba(200,90,23,0.12)_0%,transparent_60%)]" />
        </div>

        <div className="relative max-w-5xl mx-auto text-center">
          <Badge className="mb-6 bg-[#D4AF37]/10 text-[#D4AF37] border border-[#D4AF37]/30 hover:bg-[#D4AF37]/15">
            <Sparkles className="w-3 h-3 mr-1.5" /> End-to-End AI Video Production Studio
          </Badge>

          <h1 className="text-5xl sm:text-6xl md:text-7xl font-extrabold leading-[1.05] tracking-tight">
            <span className="block text-white">From Content</span>
            <span className="block text-white">to Cinema —</span>
            <span className="block bg-gradient-to-r from-[#D4AF37] via-[#F59E0B] to-[#C85A17] bg-clip-text text-transparent">
              Powered by AI
            </span>
          </h1>

          <p className="mt-7 max-w-2xl mx-auto text-lg text-zinc-400 leading-relaxed">
            A complete production pipeline that turns a song, script or story into a finished, character-consistent film.
            Director's brief, storyboard, shots and final cut — all generated, all coherent, all yours.
          </p>

          <div className="mt-9 flex flex-wrap items-center justify-center gap-3">
            <Link to={ctaTo}>
              <Button size="lg" className="bg-gradient-to-r from-[#D4AF37] to-[#C85A17] text-black hover:opacity-90 font-semibold shadow-lg shadow-[#D4AF37]/20 px-7">
                <Sparkles className="w-4 h-4 mr-2" /> {ctaLabel}
              </Button>
            </Link>
            <a href="#how">
              <Button size="lg" variant="outline" className="border-white/15 bg-white/[0.02] text-zinc-200 hover:bg-white/[0.05] hover:text-white px-7">
                See how it works
              </Button>
            </a>
          </div>

          <p className="mt-5 text-sm text-zinc-500">
            Built for filmmakers, musicians, marketers and creators who care about craft.
          </p>
        </div>
      </section>

      {/* FORMATS STRIP */}
      <section className="relative py-12 px-4 border-y border-white/[0.05] bg-white/[0.015]">
        <div className="max-w-5xl mx-auto">
          <p className="text-center text-xs text-zinc-500 mb-5 font-medium tracking-[0.18em] uppercase">
            One platform · every format
          </p>
          <div className="flex flex-wrap justify-center gap-2.5">
            {FORMATS.map(({ icon, label }) => (
              <div
                key={label}
                className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-white/[0.03] border border-white/[0.08] text-sm text-zinc-300"
              >
                <span>{icon}</span>
                <span>{label}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* HOW IT WORKS */}
      <section id="how" className="relative py-24 px-4">
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[500px] bg-[radial-gradient(ellipse_at_center,rgba(139,92,246,0.06)_0%,transparent_70%)]" />
        </div>
        <div className="relative max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <p className="text-xs uppercase tracking-[0.18em] text-[#D4AF37] font-medium mb-3">How it works</p>
            <h2 className="text-3xl sm:text-4xl font-bold text-white tracking-tight">From a single line of text to a finished film</h2>
            <p className="mt-4 max-w-2xl mx-auto text-zinc-400">
              Qaivid Core handles every stage of production — you bring the idea, it builds the pipeline.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {STEPS.map(({ icon: Icon, step, title, desc }) => (
              <div
                key={step}
                className="relative rounded-2xl bg-white/[0.025] border border-white/[0.07] p-6 hover:border-[#D4AF37]/30 hover:bg-white/[0.035] transition-all"
              >
                <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-[#D4AF37]/20 to-[#C85A17]/10 border border-[#D4AF37]/20 flex items-center justify-center mb-5">
                  <Icon className="w-5 h-5 text-[#D4AF37]" />
                </div>
                <div className="text-[11px] tracking-[0.18em] text-[#D4AF37]/80 font-semibold mb-1.5">STEP {step}</div>
                <h3 className="font-semibold text-white mb-2">{title}</h3>
                <p className="text-sm text-zinc-400 leading-relaxed">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CORE FEATURES */}
      <section id="features" className="relative py-24 px-4 border-t border-white/[0.05]">
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-0 left-1/4 w-[500px] h-[500px] bg-[radial-gradient(ellipse_at_center,rgba(200,90,23,0.05)_0%,transparent_70%)]" />
          <div className="absolute bottom-0 right-1/4 w-[450px] h-[450px] bg-[radial-gradient(ellipse_at_center,rgba(139,92,246,0.05)_0%,transparent_70%)]" />
        </div>
        <div className="relative max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <p className="text-xs uppercase tracking-[0.18em] text-[#D4AF37] font-medium mb-3">Core features</p>
            <h2 className="text-3xl sm:text-4xl font-bold text-white tracking-tight">Everything a production needs — coherent by design</h2>
            <p className="mt-4 max-w-2xl mx-auto text-zinc-400">
              Qaivid Core handles the creative and technical decisions so you stay focused on the vision.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {FEATURES.map(({ icon: Icon, title, desc }) => (
              <div
                key={title}
                className="group rounded-2xl bg-white/[0.025] border border-white/[0.07] p-6 hover:border-[#D4AF37]/30 hover:bg-white/[0.04] transition-all"
              >
                <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-[#D4AF37]/20 to-[#C85A17]/10 border border-[#D4AF37]/20 flex items-center justify-center mb-5 group-hover:from-[#D4AF37]/30 group-hover:to-[#C85A17]/20 transition-all">
                  <Icon className="w-5 h-5 text-[#D4AF37]" />
                </div>
                <h3 className="font-semibold text-white mb-2">{title}</h3>
                <p className="text-sm text-zinc-400 leading-relaxed">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* WHY QAIVID IS DIFFERENT */}
      <section id="why" className="relative py-24 px-4 border-t border-white/[0.05]">
        <div className="relative max-w-5xl mx-auto">
          <div className="text-center mb-14">
            <p className="text-xs uppercase tracking-[0.18em] text-[#D4AF37] font-medium mb-3">Why Qaivid Core</p>
            <h2 className="text-3xl sm:text-4xl font-bold text-white tracking-tight">A production studio, not a prompt box</h2>
            <p className="mt-4 max-w-2xl mx-auto text-zinc-400">
              Most AI video tools generate a clip. Qaivid Core builds the entire production around your idea.
            </p>
          </div>

          <div className="rounded-2xl border border-white/[0.07] overflow-hidden bg-white/[0.02]">
            <div className="grid grid-cols-2 text-xs uppercase tracking-[0.16em] font-semibold">
              <div className="px-6 py-4 text-zinc-500 border-r border-white/[0.07] bg-white/[0.015]">
                Generic AI video tools
              </div>
              <div className="px-6 py-4 text-[#D4AF37]">
                Qaivid Core
              </div>
            </div>
            {COMPARE.map(([left, right], i) => (
              <div
                key={i}
                className={`grid grid-cols-1 md:grid-cols-2 ${i !== COMPARE.length - 1 ? "border-b border-white/[0.05]" : ""}`}
              >
                <div className="px-6 py-5 text-sm text-zinc-400 border-r border-white/[0.05] flex items-start gap-3">
                  <X className="w-4 h-4 text-zinc-600 mt-0.5 shrink-0" />
                  <span>{left}</span>
                </div>
                <div className="px-6 py-5 text-sm text-zinc-100 flex items-start gap-3">
                  <Check className="w-4 h-4 text-[#D4AF37] mt-0.5 shrink-0" />
                  <span>{right}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* FINAL CTA */}
      <section className="relative py-28 px-4">
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[400px] bg-[radial-gradient(ellipse_at_center,rgba(212,175,55,0.10)_0%,transparent_70%)]" />
        </div>
        <div className="relative max-w-3xl mx-auto text-center">
          <h2 className="text-4xl sm:text-5xl font-bold text-white tracking-tight">
            Your next film starts with a single line of text.
          </h2>
          <p className="mt-5 text-lg text-zinc-400">
            Bring the idea. Qaivid Core handles the pipeline.
          </p>
          <div className="mt-9 flex justify-center">
            <Link to={ctaTo}>
              <Button size="lg" className="bg-gradient-to-r from-[#D4AF37] via-[#F59E0B] to-[#C85A17] text-black hover:opacity-90 font-semibold shadow-lg shadow-[#D4AF37]/25 px-8">
                {ctaLabel} <ArrowRight className="w-4 h-4 ml-2" />
              </Button>
            </Link>
          </div>
          <p className="mt-4 text-sm text-zinc-500">No credit card needed.</p>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="border-t border-white/[0.06] px-4 py-10">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-md bg-gradient-to-br from-[#D4AF37] to-[#C85A17] flex items-center justify-center">
              <Film className="w-3.5 h-3.5 text-black" />
            </div>
            <div className="flex items-baseline gap-1.5">
              <span className="text-white text-sm font-semibold tracking-tight">Qaivid</span>
              <span className="text-[10px] uppercase tracking-[0.18em] text-[#D4AF37]/80">Core 2.0</span>
            </div>
            <span className="text-zinc-600 text-sm hidden sm:inline mx-2">·</span>
            <span className="text-zinc-500 text-sm hidden sm:inline italic">From Content to Cinema</span>
          </div>
          <div className="flex items-center gap-5 text-sm text-zinc-400">
            <Link to="/auth" className="hover:text-white transition-colors">Sign in</Link>
            <Link to="/auth" className="hover:text-white transition-colors">Get Started</Link>
            {user?.role === "admin" && (
              <Link to="/admin" className="hover:text-white transition-colors">Admin</Link>
            )}
          </div>
        </div>
        <div className="max-w-7xl mx-auto mt-6 pt-6 border-t border-white/[0.04] text-center text-xs text-zinc-600">
          © {new Date().getFullYear()} Qaivid. All rights reserved.
        </div>
      </footer>
    </div>
  );
}
