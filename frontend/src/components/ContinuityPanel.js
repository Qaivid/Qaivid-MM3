import { useState } from "react";
import { Link2, Users, MapPin, Sparkles, AlertTriangle, Loader2 } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import api from "@/lib/api";

export default function ContinuityPanel({ projectId }) {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    setLoading(true);
    try {
      const r = await api.getContinuity(projectId);
      setReport(r);
    } catch (e) {
      toast.error(e?.response?.data?.detail || "Failed to analyze continuity");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-full flex flex-col">
      <div className="px-6 py-4 border-b border-[#27272A] flex items-center justify-between shrink-0">
        <div>
          <p className="panel-overline mb-1">Continuity</p>
          <h2 className="font-heading text-2xl text-[#F3F4F6]">Tracking</h2>
        </div>
        <Button onClick={handleAnalyze} disabled={loading} size="sm" className="bg-[#D4AF37] text-[#0A0A0C] hover:bg-[#F1C40F] text-xs" data-testid="analyze-continuity-btn">
          {loading ? <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" /> : <Link2 className="w-3.5 h-3.5 mr-1.5" />}
          {loading ? "Analyzing..." : "Analyze"}
        </Button>
      </div>

      <ScrollArea className="flex-1">
        {!report ? (
          <div className="p-6 text-center text-[#71717A]">
            <Link2 className="w-12 h-12 mx-auto mb-3 opacity-30" />
            <p className="text-sm">Click Analyze to check continuity across scenes and shots.</p>
            <p className="text-xs mt-1 text-[#4a4a55]">Requires scenes and shots to be built first.</p>
          </div>
        ) : (
          <div className="p-6 space-y-5">
            {/* Summary */}
            <div className="grid grid-cols-3 gap-3">
              <div className="bg-[#141417] border border-[#27272A] rounded-lg p-3 text-center">
                <p className="text-xl font-semibold text-[#D4AF37]">{report.total_subjects_tracked}</p>
                <p className="text-[10px] text-[#71717A] uppercase tracking-wider">Subjects</p>
              </div>
              <div className="bg-[#141417] border border-[#27272A] rounded-lg p-3 text-center">
                <p className="text-xl font-semibold text-[#3B82F6]">{report.total_motifs_tracked}</p>
                <p className="text-[10px] text-[#71717A] uppercase tracking-wider">Motifs</p>
              </div>
              <div className="bg-[#141417] border border-[#27272A] rounded-lg p-3 text-center">
                <p className={`text-xl font-semibold ${report.total_warnings > 0 ? 'text-[#F59E0B]' : 'text-[#10B981]'}`}>{report.total_warnings}</p>
                <p className="text-[10px] text-[#71717A] uppercase tracking-wider">Warnings</p>
              </div>
            </div>

            {/* Subjects */}
            {report.subjects?.length > 0 && (
              <div data-testid="continuity-subjects">
                <p className="panel-overline mb-2">Subject Tracking</p>
                <div className="space-y-2">
                  {report.subjects.map((sub, i) => (
                    <div key={i} className="bg-[#141417] border border-[#27272A] rounded-lg p-3 flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Users className="w-3.5 h-3.5 text-[#D4AF37]" />
                        <span className="text-sm text-[#F3F4F6]">{sub.name}</span>
                        <span className="text-[10px] bg-[#1E1E24] text-[#71717A] px-1.5 py-0.5 rounded">{sub.type}</span>
                      </div>
                      <span className="text-xs text-[#A1A1AA]">{sub.appearance_count} shots, {sub.scene_ids?.length || 0} scenes</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Motifs */}
            {report.motifs?.length > 0 && (
              <div data-testid="continuity-motifs">
                <p className="panel-overline mb-2">Motif Recurrence</p>
                <div className="flex flex-wrap gap-2">
                  {report.motifs.map((m, i) => (
                    <div key={i} className={`px-3 py-1.5 rounded-lg border text-xs ${
                      m.is_recurring ? 'bg-[#D4AF37]/10 border-[#D4AF37]/20 text-[#D4AF37]' : 'bg-[#1E1E24] border-[#27272A] text-[#A1A1AA]'
                    }`}>
                      <Sparkles className="w-3 h-3 inline mr-1" />
                      {m.motif} <span className="opacity-60">({m.occurrence_count}x)</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Locations */}
            {report.locations?.length > 0 && (
              <div data-testid="continuity-locations">
                <p className="panel-overline mb-2">Location Usage</p>
                <div className="space-y-1">
                  {report.locations.map((loc, i) => (
                    <div key={i} className="flex items-center justify-between text-xs py-1.5 border-b border-[#1F1F24] last:border-0">
                      <div className="flex items-center gap-1.5">
                        <MapPin className="w-3 h-3 text-[#C85A17]" />
                        <span className="text-[#F3F4F6]">{loc.location}</span>
                      </div>
                      <span className="text-[#71717A]">Scenes: {loc.scene_numbers?.join(', ')}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Emotional Progression */}
            {report.emotional_progression?.length > 0 && (
              <div data-testid="continuity-emotions">
                <p className="panel-overline mb-2">Emotional Flow</p>
                <div className="flex gap-1 flex-wrap">
                  {report.emotional_progression.map((ep, i) => (
                    <div key={i} className="bg-[#1E1E24] rounded px-2 py-1 text-[10px]">
                      <span className="text-[#71717A]">S{ep.scene_number}</span>
                      <span className="text-[#A1A1AA] ml-1">{ep.emotion}</span>
                      {ep.temporal !== "present" && (
                        <span className="text-[#3B82F6] ml-1">({ep.temporal})</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Warnings */}
            {report.warnings?.length > 0 && (
              <div data-testid="continuity-warnings">
                <p className="panel-overline mb-2 text-[#F59E0B]">Continuity Warnings</p>
                <div className="space-y-2">
                  {report.warnings.map((w, i) => (
                    <div key={i} className="bg-[#F59E0B]/5 border border-[#F59E0B]/20 rounded-lg p-3">
                      <div className="flex items-start gap-2">
                        <AlertTriangle className="w-3.5 h-3.5 text-[#F59E0B] mt-0.5 shrink-0" />
                        <div>
                          <p className="text-xs text-[#F3F4F6]">{w.message}</p>
                          {w.suggestion && <p className="text-[10px] text-[#71717A] mt-1">{w.suggestion}</p>}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </ScrollArea>
    </div>
  );
}
