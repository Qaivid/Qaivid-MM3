import { useState, useEffect, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { ArrowLeft, FileText, Brain, Users, Layers, Clapperboard, Sparkles, Link2, AlertTriangle, Download, Film, Video, Eye, Wand2, Image, Clock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import api from "@/lib/api";
import InputEditor from "@/components/InputEditor";
import UnderstandingPanel from "@/components/UnderstandingPanel";
import ContextPanel from "@/components/ContextPanel";
import ScenePanel from "@/components/ScenePanel";
import ShotPanel from "@/components/ShotPanel";
import PromptPanel from "@/components/PromptPanel";
import ValidationPanel from "@/components/ValidationPanel";
import ReferencesPanel from "@/components/ReferencesPanel";
import ContinuityPanel from "@/components/ContinuityPanel";
import CreativeBriefPanel from "@/components/CreativeBriefPanel";
import ProductionPanel from "@/components/ProductionPanel";
import TimelinePanel from "@/components/TimelinePanel";

const PIPELINE = [
  { key: "input", label: "Source", icon: FileText, desc: "Content input", phase: "input" },
  { key: "intelligence", label: "Intelligence", icon: Brain, desc: "Context engine", phase: "input" },
  { key: "brief", label: "Creative Brief", icon: Wand2, desc: "Direction & cast", phase: "planning" },
  { key: "storyboard", label: "Storyboard", icon: Layers, desc: "Scene design", phase: "planning" },
  { key: "shots", label: "Shot Plan", icon: Clapperboard, desc: "Cinematography", phase: "planning" },
  { key: "timeline", label: "Timeline", icon: Clock, desc: "Audio sync", phase: "planning" },
  { key: "generation", label: "Prompts", icon: Sparkles, desc: "Model prompts", phase: "production" },
  { key: "production", label: "Production", icon: Video, desc: "Stills & video", phase: "production" },
  { key: "continuity", label: "QA", icon: Link2, desc: "Continuity check", phase: "production" },
];

const PHASE_LABELS = { input: "INPUT", planning: "PLANNING", production: "PRODUCTION" };

const STATUS_PROGRESS = {
  draft: 0, input_added: 1, interpreting: 1, interpreted: 2,
  scenes_built: 4, shots_built: 5, prompts_ready: 6, complete: 7,
};

export default function ProjectWorkspace() {
  const { projectId } = useParams();
  const navigate = useNavigate();
  const [project, setProject] = useState(null);
  const [sourceInput, setSourceInput] = useState(null);
  const [context, setContext] = useState(null);
  const [scenes, setScenes] = useState([]);
  const [shots, setShots] = useState([]);
  const [prompts, setPrompts] = useState([]);
  const [brief, setBrief] = useState(null);
  const [validation, setValidation] = useState(null);
  const [activeTab, setActiveTab] = useState("input");
  const [loading, setLoading] = useState({});
  const [initialLoad, setInitialLoad] = useState(true);

  const loadProject = useCallback(async () => {
    try {
      const proj = await api.getProject(projectId);
      setProject(proj);
      await Promise.all([
        api.getInput(projectId).then(setSourceInput).catch(() => {}),
        api.getContext(projectId).then(setContext).catch(() => {}),
        api.getBrief(projectId).then(setBrief).catch(() => {}),
        api.getScenes(projectId).then(setScenes).catch(() => {}),
        api.getShots(projectId).then(setShots).catch(() => {}),
        api.getPrompts(projectId).then(setPrompts).catch(() => {}),
      ]);
      const tm = { draft: "input", input_added: "input", interpreting: "intelligence", interpreted: "intelligence", scenes_built: "storyboard", shots_built: "shots", prompts_ready: "generation" };
      if (tm[proj.status]) setActiveTab(tm[proj.status]);
    } catch { toast.error("Failed to load"); navigate("/app"); }
    finally { setInitialLoad(false); }
  }, [projectId, navigate]);

  useEffect(() => { loadProject(); }, [loadProject]);
  const setStepLoading = (s, v) => setLoading(p => ({ ...p, [s]: v }));

  const handleInterpret = async () => {
    setStepLoading("interpret", true);
    try {
      const c = await api.interpret(projectId);
      setContext(c);
      setProject(p => ({...p, status: "interpreted"}));
      setActiveTab("intelligence");
      toast.success("Context intelligence complete — review your results below");
    }
    catch (e) { toast.error(e?.response?.data?.detail || "Failed"); }
    finally { setStepLoading("interpret", false); }
  };

  const handleBuildScenes = async () => {
    setStepLoading("scenes", true);
    try { const s = await api.buildScenes(projectId); setScenes(s); setProject(p => ({...p, status: "scenes_built"})); setActiveTab("storyboard"); toast.success(`${s.length} scenes designed`); }
    catch (e) { toast.error(e?.response?.data?.detail || "Failed"); }
    finally { setStepLoading("scenes", false); }
  };

  const handleBuildShots = async () => {
    setStepLoading("shots", true);
    try { const s = await api.buildShots(projectId); setShots(s); setProject(p => ({...p, status: "shots_built"})); setActiveTab("shots"); toast.success(`${s.length} shots planned`); }
    catch (e) { toast.error(e?.response?.data?.detail || "Failed"); }
    finally { setStepLoading("shots", false); }
  };

  const handleBuildPrompts = async (model = "generic") => {
    setStepLoading("prompts", true);
    try { const p = await api.buildPrompts(projectId, model); setPrompts(p); setProject(pr => ({...pr, status: "prompts_ready"})); setActiveTab("generation"); toast.success(`${p.length} prompts for ${model}`); }
    catch (e) { toast.error(e?.response?.data?.detail || "Failed"); }
    finally { setStepLoading("prompts", false); }
  };

  const handleValidate = async () => {
    try { setValidation(await api.validate(projectId)); } catch { toast.error("Failed"); }
  };

  const handleExport = async (fmt) => {
    try {
      const d = await api.exportProject(projectId, fmt);
      const blob = new Blob([typeof d === 'string' ? d : JSON.stringify(d, null, 2)], { type: 'text/plain' });
      const u = URL.createObjectURL(blob);
      const a = document.createElement('a'); a.href = u;
      a.download = `${project?.name || 'export'}_${fmt}.${fmt === 'json' ? 'json' : fmt === 'csv' ? 'csv' : 'txt'}`;
      a.click(); URL.revokeObjectURL(u);
      toast.success(`Exported ${fmt}`);
    } catch { toast.error("Export failed"); }
  };

  if (initialLoad) {
    return (
      <div className="min-h-screen bg-[#0A0A0C] flex items-center justify-center">
        <div className="text-center">
          <div className="w-12 h-12 rounded-xl bg-[#D4AF37]/10 flex items-center justify-center mx-auto mb-4"><Film className="w-6 h-6 text-[#D4AF37] loading-pulse" /></div>
          <p className="text-[#4a4a55] text-sm">Loading project...</p>
        </div>
      </div>
    );
  }

  const statusStep = STATUS_PROGRESS[project?.status] || 0;
  const typeLabel = {song:"Music Video",poem:"Poem Film",ghazal:"Ghazal Film",qawwali:"Qawwali Film",script:"Short Film",story:"Story Film",ad:"Brand Film",documentary:"Documentary"}[project?.input_mode] || project?.input_mode;

  let currentPhase = null;

  return (
    <div className="min-h-screen bg-[#0A0A0C] flex flex-col">
      {/* Header */}
      <header className="border-b border-[#1F1F24] bg-[#0A0A0C]/90 backdrop-blur-md shrink-0 z-30">
        <div className="px-4 h-14 flex items-center gap-3">
          <Button data-testid="back-to-dashboard" variant="ghost" size="icon" onClick={() => navigate("/app")}
            className="text-[#4a4a55] hover:text-white hover:bg-[#1E1E24] w-8 h-8 rounded-lg shrink-0">
            <ArrowLeft className="w-4 h-4" />
          </Button>
          <div className="w-7 h-7 rounded-md bg-gradient-to-br from-[#D4AF37] to-[#C85A17] flex items-center justify-center shrink-0">
            <Film className="w-3.5 h-3.5 text-[#0A0A0C]" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <h1 className="font-heading text-base font-semibold text-[#F3F4F6] truncate" data-testid="project-title">{project?.name}</h1>
              <span className="text-[9px] tracking-widest uppercase text-[#4a4a55] bg-[#1E1E24] px-2 py-0.5 rounded border border-[#27272A] shrink-0">{typeLabel}</span>
            </div>
          </div>
          {/* Mini pipeline dots */}
          <div className="hidden lg:flex items-center gap-1 mr-2">
            {[1,2,3,4,5,6].map(i => (
              <div key={i} className={`w-2 h-2 rounded-full transition-all ${statusStep >= i ? 'bg-[#D4AF37]' : 'bg-[#1E1E24]'}`} />
            ))}
          </div>
          {context && (
            <Button data-testid="validate-btn" variant="ghost" size="sm" onClick={handleValidate}
              className="text-[#4a4a55] hover:text-white text-xs h-8">
              <AlertTriangle className="w-3.5 h-3.5 mr-1" /> Validate
            </Button>
          )}
          {prompts.length > 0 && (
            <div className="flex gap-1">
              {["json","csv","prompts","storyboard"].map(f => (
                <Button key={f} data-testid={`export-${f}-btn`} variant="ghost" size="sm" onClick={() => handleExport(f)}
                  className="text-[#4a4a55] hover:text-white text-[10px] h-8 px-2">
                  <Download className="w-3 h-3 mr-1" />{f.toUpperCase()}
                </Button>
              ))}
            </div>
          )}
        </div>
      </header>

      {/* Body */}
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <div className="w-[190px] border-r border-[#1F1F24] bg-[#0C0C0E] flex flex-col shrink-0 overflow-y-auto">
          <div className="flex-1 py-2">
            {PIPELINE.map((step) => {
              const Icon = step.icon;
              const isActive = activeTab === step.key;
              const stepIdx = PIPELINE.findIndex(s => s.key === step.key);
              const isDone = statusStep > stepIdx;
              const showPhase = step.phase !== currentPhase;
              if (showPhase) currentPhase = step.phase;

              return (
                <div key={step.key}>
                  {showPhase && (
                    <div className="px-4 pt-4 pb-1.5">
                      <span className="text-[8px] tracking-[0.2em] uppercase text-[#3f3f46] font-bold">{PHASE_LABELS[step.phase]}</span>
                    </div>
                  )}
                  <button data-testid={`nav-${step.key}`} onClick={() => setActiveTab(step.key)}
                    className={`mx-2 mb-0.5 w-[calc(100%-16px)] flex items-center gap-2.5 px-3 py-2 rounded-lg text-left transition-all duration-150 ${
                      isActive ? "bg-[#D4AF37]/10 text-[#D4AF37]"
                      : isDone ? "text-[#A1A1AA] hover:bg-[#141417] hover:text-[#F3F4F6]"
                      : "text-[#3f3f46] hover:bg-[#141417] hover:text-[#71717A]"
                    }`}>
                    <div className={`w-7 h-7 rounded-md flex items-center justify-center shrink-0 transition-colors ${
                      isActive ? 'bg-[#D4AF37]/20' : isDone ? 'bg-[#141417]' : 'bg-[#0A0A0C]'
                    }`}>
                      <Icon className="w-3.5 h-3.5" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="text-[11px] font-semibold truncate leading-tight">{step.label}</p>
                      <p className="text-[9px] opacity-50 truncate leading-tight">{step.desc}</p>
                    </div>
                    {isDone && !isActive && <div className="w-1.5 h-1.5 rounded-full bg-[#D4AF37]/40 shrink-0" />}
                  </button>
                </div>
              );
            })}
          </div>
          {/* Project status footer */}
          <div className="px-4 py-3 border-t border-[#1F1F24]">
            <div className="flex items-center gap-2 mb-2">
              <div className="flex-1 h-1 bg-[#1E1E24] rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-[#D4AF37] to-[#10B981] transition-all duration-500 rounded-full"
                  style={{ width: `${Math.max((statusStep / 7) * 100, 5)}%` }} />
              </div>
              <span className="text-[9px] text-[#4a4a55] font-mono">{Math.round((statusStep / 7) * 100)}%</span>
            </div>
            <p className="text-[9px] text-[#D4AF37] font-medium">{project?.status?.replace(/_/g, ' ')}</p>
          </div>
        </div>

        {/* Main */}
        <div className="flex-1 overflow-hidden">
          {activeTab === "input" && (
            <InputEditor projectId={projectId} project={project} sourceInput={sourceInput}
              setSourceInput={setSourceInput} onInterpret={handleInterpret} interpreting={loading.interpret} />
          )}
          {activeTab === "intelligence" && (
            <ResizablePanelGroup direction="horizontal" className="h-full">
              <ResizablePanel defaultSize={42} minSize={28}>
                <UnderstandingPanel context={context} loading={loading.interpret} />
              </ResizablePanel>
              <ResizableHandle className="w-1 bg-[#1F1F24] hover:bg-[#D4AF37] transition-colors" />
              <ResizablePanel defaultSize={58} minSize={28}>
                <ContextPanel projectId={projectId} context={context} setContext={setContext}
                  onNext={() => setActiveTab("brief")} />
              </ResizablePanel>
            </ResizablePanelGroup>
          )}
          {activeTab === "brief" && (
            <CreativeBriefPanel projectId={projectId} brief={brief} setBrief={setBrief}
              onNext={() => setActiveTab("storyboard")} />
          )}
          {activeTab === "storyboard" && (
            <ScenePanel scenes={scenes} setScenes={setScenes} projectId={projectId}
              onBuildScenes={handleBuildScenes} buildingScenes={loading.scenes}
              onBuildShots={handleBuildShots} buildingShots={loading.shots} />
          )}
          {activeTab === "shots" && (
            <ShotPanel shots={shots} scenes={scenes} projectId={projectId}
              onBuildPrompts={handleBuildPrompts} buildingPrompts={loading.prompts} />
          )}
          {activeTab === "timeline" && (
            <TimelinePanel projectId={projectId} />
          )}
          {activeTab === "generation" && (
            <PromptPanel prompts={prompts} shots={shots} scenes={scenes} onExport={handleExport}
              onNext={() => setActiveTab("production")} />
          )}
          {activeTab === "production" && (
            <ProductionPanel projectId={projectId} />
          )}
          {activeTab === "continuity" && <ContinuityPanel projectId={projectId} />}
        </div>

        {validation && (
          <div className="w-72 border-l border-[#1F1F24] overflow-y-auto shrink-0">
            <ValidationPanel validation={validation} onClose={() => setValidation(null)} />
          </div>
        )}
      </div>
    </div>
  );
}
