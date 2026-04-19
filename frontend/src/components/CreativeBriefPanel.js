import { useState, useEffect } from "react";
import { Wand2, Users, MapPin, Palette, Sparkles, Loader2, Film } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { toast } from "sonner";
import api from "@/lib/api";

export default function CreativeBriefPanel({ projectId, brief, setBrief, onNext }) {
  const [vibePresets, setVibePresets] = useState([]);
  const [selectedVibe, setSelectedVibe] = useState("");
  const [generating, setGenerating] = useState(false);

  useEffect(() => {
    api.listVibePresets().then(setVibePresets).catch(() => {});
  }, []);

  const handleGenerate = async () => {
    setGenerating(true);
    try {
      const b = await api.generateBrief(projectId, selectedVibe === "auto" ? "" : selectedVibe);
      setBrief(b);
      toast.success("Creative brief generated");
    } catch (e) { toast.error(e?.response?.data?.detail || "Failed"); }
    finally { setGenerating(false); }
  };

  // Generating in progress
  if (!brief && generating) {
    return (
      <div className="h-full flex flex-col">
        <div className="px-6 py-4 border-b border-[#27272A] shrink-0">
          <p className="panel-overline mb-1">Creative Direction</p>
          <h2 className="font-heading text-2xl text-[#F3F4F6]">Creative Brief</h2>
        </div>
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="text-center">
            <div className="relative w-16 h-16 mx-auto mb-5">
              <div className="w-16 h-16 rounded-2xl bg-[#D4AF37]/10 border border-[#D4AF37]/15 flex items-center justify-center">
                <Wand2 className="w-8 h-8 text-[#D4AF37]" />
              </div>
              <div className="absolute -top-1 -right-1">
                <Loader2 className="w-5 h-5 text-[#D4AF37] animate-spin" />
              </div>
            </div>
            <p className="text-[#F3F4F6] font-medium mb-1">Generating creative brief…</p>
            <p className="text-xs text-[#4a4a55]">Characters, locations, and visual direction being built from your context. ~20 seconds.</p>
          </div>
        </div>
      </div>
    );
  }

  // No brief yet — show the generate action
  if (!brief) {
    return (
      <div className="h-full flex flex-col">
        <div className="px-6 py-4 border-b border-[#27272A] shrink-0">
          <p className="panel-overline mb-1">Creative Direction</p>
          <h2 className="font-heading text-2xl text-[#F3F4F6]">Creative Brief</h2>
          <p className="text-sm text-[#71717A] mt-1">Defines characters, locations, and visual style from Context Intelligence.</p>
        </div>
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="max-w-sm text-center">
            <div className="w-16 h-16 rounded-2xl bg-[#D4AF37]/10 border border-[#D4AF37]/15 flex items-center justify-center mx-auto mb-5">
              <Wand2 className="w-8 h-8 text-[#D4AF37]" />
            </div>
            <h3 className="text-lg font-semibold text-[#F3F4F6] mb-2">Generate Creative Brief</h3>
            <p className="text-sm text-[#71717A] mb-6">Powered by the Context Engine — defines characters, locations, visual aesthetic, and motifs for the entire production.</p>
            <div className="mb-4">
              <label className="text-[10px] tracking-widest uppercase text-[#71717A] font-semibold mb-1.5 block text-left">Vibe Preset (optional)</label>
              <Select value={selectedVibe} onValueChange={setSelectedVibe}>
                <SelectTrigger data-testid="vibe-select" className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6]">
                  <SelectValue placeholder="Auto — from Context Intelligence" />
                </SelectTrigger>
                <SelectContent className="bg-[#1E1E24] border-[#27272A] max-h-60">
                  <SelectItem value="auto" className="text-[#F3F4F6] focus:bg-[#27272E] focus:text-white">Auto — from Context Intelligence</SelectItem>
                  {vibePresets.map(v => (
                    <SelectItem key={v.id} value={v.id} className="text-[#F3F4F6] focus:bg-[#27272E] focus:text-white">{v.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Button onClick={handleGenerate} disabled={generating} className="bg-[#D4AF37] text-[#0A0A0C] hover:bg-[#F1C40F] font-semibold w-full" data-testid="generate-brief-btn">
              <Wand2 className="w-4 h-4 mr-2" /> Generate Creative Brief
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <div className="px-6 py-4 border-b border-[#27272A] flex items-center justify-between shrink-0">
        <div>
          <p className="panel-overline mb-1">Creative Direction</p>
          <h2 className="font-heading text-2xl text-[#F3F4F6]">{brief.title || "Creative Brief"}</h2>
          {brief.tagline && <p className="text-sm text-[#D4AF37] italic mt-0.5">{brief.tagline}</p>}
        </div>
        {onNext && (
          <Button onClick={onNext} size="sm" className="bg-[#D4AF37] text-[#0A0A0C] hover:bg-[#F1C40F] text-xs" data-testid="brief-next-btn">
            Continue to Storyboard
          </Button>
        )}
      </div>
      <ScrollArea className="flex-1">
        <div className="p-6 space-y-5">
          {/* Narrative Arc */}
          {brief.narrative_arc && (
            <div className="bg-[#141417] border border-[#27272A] rounded-lg p-4">
              <p className="panel-overline mb-2">Narrative Arc</p>
              <p className="text-sm text-[#F3F4F6] leading-relaxed font-heading italic">{brief.narrative_arc}</p>
            </div>
          )}
          {brief.emotional_journey && (
            <div className="bg-[#141417] border border-[#27272A] rounded-lg p-4">
              <p className="panel-overline mb-2">Emotional Journey</p>
              <p className="text-sm text-[#A1A1AA]">{brief.emotional_journey}</p>
            </div>
          )}

          {/* Visual Aesthetic */}
          {brief.visual_aesthetic && (
            <div className="bg-[#141417] border border-[#27272A] rounded-lg p-4" data-testid="brief-aesthetic">
              <p className="panel-overline mb-3"><Palette className="w-3 h-3 inline mr-1" />Visual Aesthetic</p>
              <div className="space-y-2 text-xs">
                {brief.visual_aesthetic.style && <div><span className="text-[#71717A]">Style:</span> <span className="text-[#F3F4F6]">{brief.visual_aesthetic.style}</span></div>}
                {brief.visual_aesthetic.lighting_mood && <div><span className="text-[#71717A]">Lighting:</span> <span className="text-[#F3F4F6]">{brief.visual_aesthetic.lighting_mood}</span></div>}
                {brief.visual_aesthetic.cinematography_style && <div><span className="text-[#71717A]">Camera:</span> <span className="text-[#F3F4F6]">{brief.visual_aesthetic.cinematography_style}</span></div>}
                {brief.visual_aesthetic.color_palette?.length > 0 && (
                  <div className="flex items-center gap-2 flex-wrap mt-2">
                    {brief.visual_aesthetic.color_palette.map((c, i) => (
                      <span key={i} className="px-2 py-0.5 rounded bg-[#1E1E24] text-[#D4AF37] text-[10px]">{c}</span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Characters */}
          {brief.characters?.length > 0 && (
            <div data-testid="brief-characters">
              <p className="panel-overline mb-3"><Users className="w-3 h-3 inline mr-1" />Characters</p>
              <div className="space-y-3">
                {brief.characters.map((char, i) => (
                  <div key={i} className="film-slate bg-[#141417] border border-[#27272A] rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <h4 className="text-sm font-semibold text-[#F3F4F6]">{char.name}</h4>
                      <span className="text-[9px] bg-[#D4AF37]/10 text-[#D4AF37] px-1.5 py-0.5 rounded uppercase">{char.role}</span>
                      {char.age && <span className="text-[10px] text-[#71717A]">{char.age}</span>}
                    </div>
                    {(char.physical_description || char.physicalDescription) && (
                      <p className="text-xs text-[#A1A1AA] mb-1"><span className="text-[#4a4a55]">Look: </span>{char.physical_description || char.physicalDescription}</p>
                    )}
                    {char.wardrobe && <p className="text-xs text-[#A1A1AA] mb-1"><span className="text-[#4a4a55]">Wardrobe: </span>{char.wardrobe}</p>}
                    {char.emotional_arc && <p className="text-xs text-[#71717A]"><span className="text-[#4a4a55]">Arc: </span>{char.emotional_arc}</p>}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Locations */}
          {brief.locations?.length > 0 && (
            <div data-testid="brief-locations">
              <p className="panel-overline mb-3"><MapPin className="w-3 h-3 inline mr-1" />Locations</p>
              <div className="space-y-3">
                {brief.locations.map((loc, i) => (
                  <div key={i} className="film-slate bg-[#141417] border border-[#27272A] rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <h4 className="text-sm font-semibold text-[#F3F4F6]">{loc.name}</h4>
                      {(loc.time_of_day || loc.timeOfDay) && <span className="text-[10px] text-[#C85A17]">{loc.time_of_day || loc.timeOfDay}</span>}
                      {loc.mood && <span className="text-[10px] text-[#71717A]">{loc.mood}</span>}
                    </div>
                    {loc.description && <p className="text-xs text-[#A1A1AA] mb-1">{loc.description}</p>}
                    {(loc.visual_details || loc.visualDetails) && <p className="text-xs text-[#71717A]">{loc.visual_details || loc.visualDetails}</p>}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Visual Motifs */}
          {brief.visual_motifs?.length > 0 && (
            <div className="bg-[#141417] border border-[#27272A] rounded-lg p-4">
              <p className="panel-overline mb-2"><Sparkles className="w-3 h-3 inline mr-1" />Visual Motifs</p>
              <div className="flex flex-wrap gap-2">
                {brief.visual_motifs.map((m, i) => (
                  <span key={i} className="px-3 py-1 rounded-lg bg-[#D4AF37]/10 border border-[#D4AF37]/20 text-xs text-[#D4AF37]">{m}</span>
                ))}
              </div>
            </div>
          )}

          {/* Production Notes */}
          {brief.production_notes && (
            <div className="bg-[#141417] border border-[#27272A] rounded-lg p-4">
              <p className="panel-overline mb-2"><Film className="w-3 h-3 inline mr-1" />Production Notes</p>
              <p className="text-sm text-[#A1A1AA] leading-relaxed">{brief.production_notes}</p>
            </div>
          )}

          {/* Vibe preset */}
          {brief.vibe_preset && (
            <div className="text-[10px] text-[#4a4a55]">Vibe preset applied: {brief.vibe_preset}</div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
