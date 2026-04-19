import { useState } from "react";
import { Lock, Unlock, Layers, Plus } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { toast } from "sonner";
import api from "@/lib/api";

const VIZ_MODE_COLORS = {
  direct: "bg-[#10B981]/20 text-[#10B981]",
  indirect: "bg-[#3B82F6]/20 text-[#3B82F6]",
  symbolic: "bg-[#D4AF37]/20 text-[#D4AF37]",
  absorbed: "bg-[#71717A]/20 text-[#A1A1AA]",
  performance_only: "bg-[#C85A17]/20 text-[#C85A17]",
};

export default function ContextPanel({ projectId, context, setContext, onNext }) {
  const [showOverride, setShowOverride] = useState(false);
  const [overrideField, setOverrideField] = useState("");
  const [overrideValue, setOverrideValue] = useState("");

  if (!context) {
    return (
      <div className="h-full flex items-center justify-center text-[#71717A]">
        No context packet available
      </div>
    );
  }

  const handleAddOverride = async () => {
    if (!overrideField || !overrideValue) return;
    try {
      await api.addOverride(projectId, {
        field_path: overrideField,
        override_value: overrideValue,
        locked: true,
      });
      // Refresh context
      const updated = await api.getContext(projectId);
      setContext(updated);
      setShowOverride(false);
      setOverrideField("");
      setOverrideValue("");
      toast.success("Assumption overridden and locked");
    } catch {
      toast.error("Failed to save override");
    }
  };

  const locked = context.locked_assumptions || {};

  return (
    <ScrollArea className="h-full">
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="panel-overline mb-1">World Building</p>
            <h2 className="font-heading text-2xl text-[#F3F4F6]">Culture & Setting</h2>
          </div>
          <div className="flex gap-2">
            <Dialog open={showOverride} onOpenChange={setShowOverride}>
              <DialogTrigger asChild>
                <Button
                  data-testid="add-override-btn"
                  variant="outline"
                  size="sm"
                  className="border-[#27272A] text-[#A1A1AA] hover:text-white hover:bg-[#27272E] text-xs"
                >
                  <Plus className="w-3 h-3 mr-1" />
                  Override
                </Button>
              </DialogTrigger>
              <DialogContent className="bg-[#141417] border border-[#27272A] text-[#F3F4F6]">
                <DialogHeader>
                  <DialogTitle className="font-heading text-xl text-[#F3F4F6]">Override Assumption</DialogTitle>
                </DialogHeader>
                <div className="space-y-3 mt-3">
                  <div>
                    <label className="panel-overline mb-1 block">Field Path</label>
                    <Input
                      data-testid="override-field-input"
                      value={overrideField}
                      onChange={(e) => setOverrideField(e.target.value)}
                      placeholder="e.g. speaker_model.gender"
                      className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] text-sm font-mono"
                    />
                    <p className="text-[10px] text-[#71717A] mt-1">
                      Examples: speaker_model.gender, world_assumptions.geography, narrative_mode
                    </p>
                  </div>
                  <div>
                    <label className="panel-overline mb-1 block">New Value</label>
                    <Input
                      data-testid="override-value-input"
                      value={overrideValue}
                      onChange={(e) => setOverrideValue(e.target.value)}
                      placeholder="e.g. female"
                      className="bg-[#1E1E24] border-[#27272A] text-[#F3F4F6] text-sm"
                    />
                  </div>
                  <Button
                    data-testid="save-override-btn"
                    onClick={handleAddOverride}
                    className="w-full bg-[#D4AF37] text-[#0A0A0C] hover:bg-[#F1C40F] hover:text-[#0A0A0C]"
                  >
                    <Lock className="w-3 h-3 mr-1.5" />
                    Lock Override
                  </Button>
                </div>
              </DialogContent>
            </Dialog>
            <Button
              data-testid="continue-to-brief-btn"
              onClick={onNext}
              className="bg-[#D4AF37] text-[#0A0A0C] hover:bg-[#F1C40F] hover:text-[#0A0A0C] font-semibold text-xs"
            >
              <Layers className="w-3.5 h-3.5 mr-1.5" />
              Continue to Creative Brief
            </Button>
          </div>
        </div>

        {/* Locked Assumptions */}
        {Object.keys(locked).length > 0 && (
          <div className="bg-[#D4AF37]/5 border border-[#D4AF37]/20 rounded-lg p-4" data-testid="locked-assumptions">
            <p className="panel-overline mb-2 text-[#D4AF37]">Locked Assumptions</p>
            {Object.entries(locked).map(([k, v]) => (
              <div key={k} className="flex items-center gap-2 text-xs py-1">
                <Lock className="w-3 h-3 text-[#D4AF37]" />
                <span className="text-[#A1A1AA] font-mono">{k}:</span>
                <span className="text-[#F3F4F6]">{String(v)}</span>
              </div>
            ))}
          </div>
        )}

        {/* World Assumptions */}
        <div className="bg-[#141417] border border-[#27272A] rounded-lg p-4" data-testid="world-assumptions">
          <p className="panel-overline mb-3">World Assumptions</p>
          {context.world_assumptions && Object.entries(context.world_assumptions).map(([k, v]) => (
            <div key={k} className="flex justify-between text-xs py-1.5 border-b border-[#1F1F24] last:border-0">
              <span className="text-[#71717A]">{k.replace(/_/g, ' ')}</span>
              <span className="text-[#F3F4F6] text-right max-w-[60%]">{v || "—"}</span>
            </div>
          ))}
        </div>

        {/* Cultural Setting */}
        {context.cultural_setting && Object.keys(context.cultural_setting).length > 0 && (
          <div className="bg-[#141417] border border-[#27272A] rounded-lg p-4" data-testid="cultural-setting">
            <p className="panel-overline mb-3">Cultural Setting</p>
            {Object.entries(context.cultural_setting).map(([k, v]) => {
              if (Array.isArray(v)) {
                return (
                  <div key={k} className="py-1.5 border-b border-[#1F1F24] last:border-0">
                    <span className="text-[#71717A] text-xs">{k.replace(/_/g, ' ')}</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {v.map((item, i) => (
                        <span key={i} className="text-[10px] bg-[#1E1E24] text-[#A1A1AA] px-2 py-0.5 rounded">
                          {item}
                        </span>
                      ))}
                    </div>
                  </div>
                );
              }
              return (
                <div key={k} className="flex justify-between text-xs py-1.5 border-b border-[#1F1F24] last:border-0">
                  <span className="text-[#71717A]">{k.replace(/_/g, ' ')}</span>
                  <span className="text-[#F3F4F6]">{String(v) || "—"}</span>
                </div>
              );
            })}
          </div>
        )}

        {/* Motif Map */}
        {context.motif_map && Object.keys(context.motif_map).length > 0 && (
          <div className="bg-[#141417] border border-[#27272A] rounded-lg p-4">
            <p className="panel-overline mb-3">Motif Map</p>
            {Object.entries(context.motif_map).map(([motif, indices]) => (
              <div key={motif} className="flex items-center gap-2 text-xs py-1.5 border-b border-[#1F1F24] last:border-0">
                <span className="text-[#D4AF37] font-medium">{motif}</span>
                <span className="text-[#71717A]">lines: {Array.isArray(indices) ? indices.join(', ') : indices}</span>
              </div>
            ))}
          </div>
        )}

        {/* Line Meanings */}
        {context.line_meanings?.length > 0 && (
          <div data-testid="line-meanings">
            <p className="panel-overline mb-3">Line-by-Line Interpretation</p>
            <div className="space-y-2">
              {context.line_meanings.map((lm, i) => (
                <div key={i} className="bg-[#141417] border border-[#27272A] rounded-lg p-3 text-xs">
                  <div className="flex items-start justify-between mb-2">
                    <p className="font-heading text-sm text-[#F3F4F6] italic flex-1">{lm.text}</p>
                    <span className={`ml-2 px-2 py-0.5 rounded text-[10px] uppercase font-semibold shrink-0 ${
                      VIZ_MODE_COLORS[lm.visualization_mode] || VIZ_MODE_COLORS.direct
                    }`}>
                      {lm.visualization_mode}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[#A1A1AA]">
                    <div><span className="text-[#71717A]">Literal:</span> {lm.literal_meaning}</div>
                    <div><span className="text-[#71717A]">Implied:</span> {lm.implied_meaning}</div>
                    <div><span className="text-[#71717A]">Emotional:</span> {lm.emotional_meaning}</div>
                    <div><span className="text-[#71717A]">Cultural:</span> {lm.cultural_meaning}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Restrictions */}
        {context.restrictions?.length > 0 && (
          <div className="bg-[#141417] border border-[#EF4444]/20 rounded-lg p-4">
            <p className="panel-overline mb-2 text-[#EF4444]">Restrictions</p>
            {context.restrictions.map((r, i) => (
              <p key={i} className="text-xs text-[#A1A1AA] py-1">{r}</p>
            ))}
          </div>
        )}
      </div>
    </ScrollArea>
  );
}
