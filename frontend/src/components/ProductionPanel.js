import { useState, useEffect, useRef, useCallback } from "react";
import { Image, Video, Layers, Play, Loader2, Download, Film, CheckCircle, XCircle, RefreshCw } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import api from "@/lib/api";

const BACKEND = process.env.REACT_APP_BACKEND_URL;

export default function ProductionPanel({ projectId }) {
  const [referencePrompts, setReferencePrompts] = useState([]);
  const [stillPrompts, setStillPrompts] = useState([]);
  const [renderPlan, setRenderPlan] = useState([]);
  const [timeline, setTimeline] = useState(null);
  const [assembly, setAssembly] = useState(null);
  const [loading, setLoading] = useState({});

  useEffect(() => {
    api.getReferencePrompts(projectId).then(setReferencePrompts).catch(() => {});
    api.getStillPrompts(projectId).then(setStillPrompts).catch(() => {});
    api.getRenderPlan(projectId).then(setRenderPlan).catch(() => {});
    api.getTimeline(projectId).then(setTimeline).catch(() => {});
    api.getAssembly(projectId).then(setAssembly).catch(() => {});
  }, [projectId]);

  // Prompt builders
  const handleBuildRefs = async () => {
    setLoading(p => ({...p, buildRefs: true}));
    try { setReferencePrompts(await api.buildReferencePrompts(projectId)); toast.success("Reference prompts built"); }
    catch (e) { toast.error(e?.response?.data?.detail || "Failed"); }
    finally { setLoading(p => ({...p, buildRefs: false})); }
  };
  const handleBuildStills = async () => {
    setLoading(p => ({...p, buildStills: true}));
    try { setStillPrompts(await api.buildStillPrompts(projectId)); toast.success("Still prompts built"); }
    catch (e) { toast.error(e?.response?.data?.detail || "Failed"); }
    finally { setLoading(p => ({...p, buildStills: false})); }
  };
  const handleBuildRender = async () => {
    setLoading(p => ({...p, buildRender: true}));
    try { setRenderPlan(await api.buildRenderPlan(projectId)); toast.success("Render plan created"); }
    catch (e) { toast.error(e?.response?.data?.detail || "Failed"); }
    finally { setLoading(p => ({...p, buildRender: false})); }
  };
  const handleBuildTimeline = async () => {
    setLoading(p => ({...p, buildTimeline: true}));
    try { setTimeline(await api.buildTimeline(projectId)); toast.success("Timeline assembled"); }
    catch (e) { toast.error(e?.response?.data?.detail || "Failed"); }
    finally { setLoading(p => ({...p, buildTimeline: false})); }
  };

  // Real generation
  const handleGenRefs = async () => {
    setLoading(p => ({...p, genRefs: true}));
    try {
      const r = await api.generateReferences(projectId);
      toast.success(`Generated ${r.generated} reference images`);
      setReferencePrompts(await api.getReferencePrompts(projectId));
    } catch (e) { toast.error(e?.response?.data?.detail || "Generation failed"); }
    finally { setLoading(p => ({...p, genRefs: false})); }
  };
  const handleGenStills = async () => {
    setLoading(p => ({...p, genStills: true}));
    try {
      const r = await api.generateStills(projectId);
      toast.success(`Generated ${r.generated} shot stills`);
      setStillPrompts(await api.getStillPrompts(projectId));
    } catch (e) { toast.error(e?.response?.data?.detail || "Generation failed"); }
    finally { setLoading(p => ({...p, genStills: false})); }
  };
  const handleRenderVideos = async () => {
    setLoading(p => ({...p, render: true}));
    try {
      const r = await api.renderVideos(projectId);
      toast.success(`Submitted ${r.submitted} shots for video generation`);
      setRenderPlan(await api.getRenderPlan(projectId));
      // Start polling
      startPolling();
    } catch (e) { toast.error(e?.response?.data?.detail || "Render failed"); }
    finally { setLoading(p => ({...p, render: false})); }
  };

  // Render status polling
  const [renderStatus, setRenderStatus] = useState(null);
  const pollRef = useRef(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  }, []);

  const pollStatus = useCallback(async () => {
    try {
      const status = await api.getRenderStatus(projectId);
      setRenderStatus(status);
      if (status.all_done) {
        stopPolling();
        setRenderPlan(await api.getRenderPlan(projectId));
        if (status.completed > 0) toast.success(`${status.completed} video clips ready`);
      }
    } catch { /* ignore */ }
  }, [projectId, stopPolling]);

  const startPolling = useCallback(() => {
    stopPolling();
    pollStatus();
    pollRef.current = setInterval(pollStatus, 8000);
  }, [pollStatus, stopPolling]);

  // Auto-start polling if there are submitted/processing renders
  useEffect(() => {
    const hasInProgress = renderPlan.some(r => r.status === "submitted" || r.status === "processing");
    if (hasInProgress) startPolling();
    return stopPolling;
  }, [renderPlan, startPolling, stopPolling]);
  const handleAssemble = async () => {
    setLoading(p => ({...p, assemble: true}));
    try {
      const a = await api.assembleVideo(projectId);
      setAssembly(a);
      toast.success("Final video assembled!");
    } catch (e) { toast.error(e?.response?.data?.detail || "Assembly failed"); }
    finally { setLoading(p => ({...p, assemble: false})); }
  };

  const completedRefs = referencePrompts.filter(r => r.status === "completed").length;
  const completedStills = stillPrompts.filter(s => s.status === "completed").length;
  const completedRenders = renderPlan.filter(r => r.status === "completed").length;

  return (
    <div className="h-full flex flex-col">
      <div className="px-6 py-4 border-b border-[#27272A] shrink-0">
        <p className="panel-overline mb-1">Post-Production</p>
        <h2 className="font-heading text-2xl text-[#F3F4F6]">Production Pipeline</h2>
        <p className="text-sm text-[#71717A] mt-1">Generate images, render video clips, and assemble the final video.</p>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-6 space-y-5">

          {/* ─── Stage 1: Reference Images ─── */}
          <div className="bg-[#141417] border border-[#27272A] rounded-xl p-5" data-testid="production-refs">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-[#D4AF37]/10 flex items-center justify-center"><Image className="w-5 h-5 text-[#D4AF37]" /></div>
                <div>
                  <h3 className="text-sm font-semibold text-[#F3F4F6]">Reference Images</h3>
                  <p className="text-[10px] text-[#4a4a55]">Character portraits & location references — GPT Image 1</p>
                </div>
              </div>
              <div className="flex gap-2">
                <Button onClick={handleBuildRefs} disabled={loading.buildRefs} size="sm" className="bg-[#1E1E24] border border-[#27272A] text-[#A1A1AA] hover:text-white text-xs h-8">
                  {loading.buildRefs ? <Loader2 className="w-3 h-3 animate-spin" /> : "Build Prompts"}
                </Button>
                {referencePrompts.length > 0 && (
                  <Button onClick={handleGenRefs} disabled={loading.genRefs} size="sm" className="bg-[#D4AF37] text-[#0A0A0C] hover:bg-[#F1C40F] text-xs h-8" data-testid="gen-refs-btn">
                    {loading.genRefs ? <><Loader2 className="w-3 h-3 animate-spin mr-1" />Generating...</> : <><Play className="w-3 h-3 mr-1" />Generate Images</>}
                  </Button>
                )}
              </div>
            </div>
            {referencePrompts.length > 0 && (
              <div className="space-y-2">
                {referencePrompts.map(ref => (
                  <div key={ref.id} className="bg-[#1E1E24] rounded-lg p-3 flex items-center gap-3">
                    {ref.image_url ? (
                      <img src={`${BACKEND}${ref.image_url}`} alt={ref.name} className="w-14 h-14 rounded-lg object-cover border border-[#27272A]" />
                    ) : (
                      <div className="w-14 h-14 rounded-lg bg-[#0A0A0C] border border-[#27272A] flex items-center justify-center"><Image className="w-5 h-5 text-[#27272A]" /></div>
                    )}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className={`text-[9px] uppercase tracking-wider font-semibold px-1.5 py-0.5 rounded ${ref.type === 'character' ? 'bg-[#D4AF37]/10 text-[#D4AF37]' : 'bg-[#C85A17]/10 text-[#C85A17]'}`}>{ref.type}</span>
                        <span className="text-xs text-[#F3F4F6]">{ref.name}</span>
                      </div>
                      <p className="text-[10px] text-[#4a4a55] truncate mt-0.5">{ref.prompt?.slice(0, 80)}...</p>
                    </div>
                    <StatusBadge status={ref.status} />
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* ─── Stage 2: Shot Stills ─── */}
          <div className="bg-[#141417] border border-[#27272A] rounded-xl p-5" data-testid="production-stills">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-[#8B5CF6]/10 flex items-center justify-center"><Image className="w-5 h-5 text-[#8B5CF6]" /></div>
                <div>
                  <h3 className="text-sm font-semibold text-[#F3F4F6]">Shot Stills</h3>
                  <p className="text-[10px] text-[#4a4a55]">Individual shot images — {completedStills}/{stillPrompts.length} generated</p>
                </div>
              </div>
              <div className="flex gap-2">
                <Button onClick={handleBuildStills} disabled={loading.buildStills} size="sm" className="bg-[#1E1E24] border border-[#27272A] text-[#A1A1AA] hover:text-white text-xs h-8">
                  {loading.buildStills ? <Loader2 className="w-3 h-3 animate-spin" /> : "Build Prompts"}
                </Button>
                {stillPrompts.length > 0 && (
                  <Button onClick={handleGenStills} disabled={loading.genStills} size="sm" className="bg-[#8B5CF6] text-white hover:bg-[#7C3AED] text-xs h-8" data-testid="gen-stills-btn">
                    {loading.genStills ? <><Loader2 className="w-3 h-3 animate-spin mr-1" />Generating...</> : <><Play className="w-3 h-3 mr-1" />Generate Stills</>}
                  </Button>
                )}
              </div>
            </div>
            {stillPrompts.length > 0 && (
              <div className="grid grid-cols-4 lg:grid-cols-6 gap-2">
                {stillPrompts.slice(0, 12).map(sp => (
                  <div key={sp.id} className="bg-[#1E1E24] rounded-lg overflow-hidden">
                    <div className="aspect-video bg-[#0A0A0C] flex items-center justify-center">
                      {sp.image_url ? (
                        <img src={`${BACKEND}${sp.image_url}`} alt={`Shot ${sp.shot_number}`} className="w-full h-full object-cover" />
                      ) : (
                        <Image className="w-4 h-4 text-[#27272A]" />
                      )}
                    </div>
                    <div className="p-1.5 flex items-center justify-between">
                      <span className="text-[9px] text-[#4a4a55]">Shot {sp.shot_number}</span>
                      <StatusDot status={sp.status} />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* ─── Stage 3: Video Render ─── */}
          <div className="bg-[#141417] border border-[#27272A] rounded-xl p-5" data-testid="production-render">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-[#C85A17]/10 flex items-center justify-center"><Video className="w-5 h-5 text-[#C85A17]" /></div>
                <div>
                  <h3 className="text-sm font-semibold text-[#F3F4F6]">Video Clips</h3>
                  <p className="text-[10px] text-[#4a4a55]">Wan 2.6 image-to-video — {completedRenders}/{renderPlan.length} rendered</p>
                </div>
              </div>
              <div className="flex gap-2">
                <Button onClick={handleBuildRender} disabled={loading.buildRender} size="sm" className="bg-[#1E1E24] border border-[#27272A] text-[#A1A1AA] hover:text-white text-xs h-8">
                  {loading.buildRender ? <Loader2 className="w-3 h-3 animate-spin" /> : "Build Plan"}
                </Button>
                {renderPlan.length > 0 && (
                  <>
                    <Button onClick={handleRenderVideos} disabled={loading.render} size="sm" className="bg-[#C85A17] text-white hover:bg-[#D4AF37] text-xs h-8" data-testid="render-btn">
                      {loading.render ? <><Loader2 className="w-3 h-3 animate-spin mr-1" />Submitting...</> : <><Play className="w-3 h-3 mr-1" />Render Videos</>}
                    </Button>
                    {renderStatus && renderStatus.processing > 0 && (
                      <Button onClick={pollStatus} size="sm" className="bg-[#1E1E24] border border-[#27272A] text-[#A1A1AA] hover:text-white text-xs h-8" data-testid="check-status-btn">
                        <RefreshCw className="w-3 h-3 mr-1" /> Check Status
                      </Button>
                    )}
                  </>
                )}
              </div>
            </div>
            {renderPlan.length > 0 && (
              <div className="space-y-2">
                <div className="flex gap-1 flex-wrap">
                  {renderPlan.map(rj => (
                    <div key={rj.id} className={`h-8 rounded-md flex items-center justify-center px-2 text-[9px] ${
                      rj.status === 'completed' ? 'bg-[#10B981]/20 text-[#10B981]' :
                      rj.status === 'failed' ? 'bg-[#EF4444]/20 text-[#EF4444]' :
                      (rj.status === 'submitted' || rj.status === 'processing') ? 'bg-[#C85A17]/20 text-[#C85A17] animate-pulse' :
                      'bg-[#1E1E24] text-[#4a4a55]'
                    }`}>
                      Shot {rj.shot_number} {'\u2022'} {rj.duration_sec}s
                    </div>
                  ))}
                </div>
                {renderStatus && (renderStatus.processing > 0 || renderStatus.completed > 0) && (
                  <div className="bg-[#1E1E24] rounded-lg px-3 py-2 flex items-center justify-between" data-testid="render-progress">
                    <div className="flex items-center gap-3 text-xs">
                      {renderStatus.processing > 0 && (
                        <span className="text-[#C85A17] flex items-center gap-1"><Loader2 className="w-3 h-3 animate-spin" /> {renderStatus.processing} rendering</span>
                      )}
                      {renderStatus.completed > 0 && (
                        <span className="text-[#10B981]">{renderStatus.completed} complete</span>
                      )}
                      {renderStatus.failed > 0 && (
                        <span className="text-[#EF4444]">{renderStatus.failed} failed</span>
                      )}
                    </div>
                    {renderStatus.all_done && <span className="text-[10px] text-[#10B981] font-semibold">All done</span>}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* ─── Stage 4: Assembly ─── */}
          <div className="bg-[#141417] border border-[#27272A] rounded-xl p-5" data-testid="production-assembly">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-[#10B981]/10 flex items-center justify-center"><Film className="w-5 h-5 text-[#10B981]" /></div>
                <div>
                  <h3 className="text-sm font-semibold text-[#F3F4F6]">Final Assembly</h3>
                  <p className="text-[10px] text-[#4a4a55]">FFmpeg stitch + audio sync → final MP4</p>
                </div>
              </div>
              <div className="flex gap-2">
                <Button onClick={handleBuildTimeline} disabled={loading.buildTimeline} size="sm" className="bg-[#1E1E24] border border-[#27272A] text-[#A1A1AA] hover:text-white text-xs h-8">
                  {loading.buildTimeline ? <Loader2 className="w-3 h-3 animate-spin" /> : "Build Timeline"}
                </Button>
                {timeline && (
                  <Button onClick={handleAssemble} disabled={loading.assemble} size="sm" className="bg-[#10B981] text-white hover:bg-[#059669] text-xs h-8" data-testid="assemble-btn">
                    {loading.assemble ? <><Loader2 className="w-3 h-3 animate-spin mr-1" />Assembling...</> : <><Film className="w-3 h-3 mr-1" />Assemble Video</>}
                  </Button>
                )}
              </div>
            </div>
            {/* Timeline visualization */}
            {timeline && (
              <div className="bg-[#1E1E24] rounded-lg p-3 mb-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-[#F3F4F6]">{timeline.total_clips} clips</span>
                  <span className="text-xs text-[#D4AF37] font-mono">{timeline.total_duration_sec?.toFixed(1)}s total</span>
                </div>
                <div className="flex gap-0.5">
                  {timeline.clips?.map((clip, ci) => (
                    <div key={ci} className={`h-5 rounded-sm flex-1 min-w-[6px] transition-all ${clip.status === 'ready' ? 'bg-[#10B981]' : 'bg-[#27272A]'}`}
                      title={`Shot ${clip.shot_number}: ${clip.duration_sec}s`} />
                  ))}
                </div>
              </div>
            )}
            {/* Final video player */}
            {assembly && (
              <div className="bg-[#0A0A0C] rounded-lg overflow-hidden border border-[#10B981]/20">
                <video controls className="w-full max-h-[300px]" src={`${BACKEND}${assembly.video_url}`} />
                <div className="p-3 flex items-center justify-between">
                  <div>
                    <p className="text-sm font-semibold text-[#10B981]">Video Ready</p>
                    <p className="text-[10px] text-[#4a4a55]">{assembly.total_clips} clips • {assembly.duration_sec?.toFixed(1)}s • {assembly.has_audio ? 'with audio' : 'no audio'}</p>
                  </div>
                  <a href={`${BACKEND}${assembly.video_url}`} download className="inline-flex items-center gap-1.5 bg-[#10B981] text-white px-4 py-2 rounded-lg text-xs font-semibold hover:bg-[#059669] transition-colors" data-testid="download-video-btn">
                    <Download className="w-3.5 h-3.5" /> Download MP4
                  </a>
                </div>
              </div>
            )}
          </div>
        </div>
      </ScrollArea>
    </div>
  );
}

function StatusBadge({ status }) {
  if (status === "completed") return <CheckCircle className="w-4 h-4 text-[#10B981] shrink-0" />;
  if (status === "failed") return <XCircle className="w-4 h-4 text-[#EF4444] shrink-0" />;
  return <span className="text-[10px] text-[#4a4a55]">{status}</span>;
}

function StatusDot({ status }) {
  const color = status === "completed" ? "bg-[#10B981]" : status === "failed" ? "bg-[#EF4444]" : "bg-[#27272A]";
  return <div className={`w-2 h-2 rounded-full ${color}`} />;
}
