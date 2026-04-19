import { ScrollArea } from "@/components/ui/scroll-area";

const FIELD_MAP = [
  { key: "input_type", label: "Content Type" },
  { key: "narrative_mode", label: "Narrative Mode" },
  { key: "core_theme", label: "Core Theme" },
  { key: "dramatic_premise", label: "Dramatic Premise" },
];

const CONFIDENCE_COLORS = {
  high: "text-[#10B981]",
  medium: "text-[#F59E0B]",
  low: "text-[#EF4444]",
};

function confLevel(score) {
  if (score >= 0.7) return "high";
  if (score >= 0.4) return "medium";
  return "low";
}

export default function UnderstandingPanel({ context, loading }) {
  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="loading-pulse text-[#D4AF37]">Interpreting meaning...</div>
      </div>
    );
  }

  if (!context) {
    return (
      <div className="h-full flex items-center justify-center px-8 text-center">
        <div>
          <p className="text-[#71717A] text-lg">No interpretation yet</p>
          <p className="text-[#4a4a55] text-sm mt-1">Add input and run interpretation</p>
        </div>
      </div>
    );
  }

  const confidence = context.confidence_scores || {};

  return (
    <ScrollArea className="h-full">
      <div className="p-6 space-y-6">
        <div>
          <p className="panel-overline mb-1">Context Intelligence</p>
          <h2 className="font-heading text-2xl text-[#F3F4F6]">Meaning Analysis</h2>
        </div>

        {/* Core Fields */}
        <div className="space-y-4" data-testid="understanding-fields">
          {FIELD_MAP.map(f => (
            <div key={f.key} className="bg-[#141417] border border-[#27272A] rounded-lg p-4">
              <p className="panel-overline mb-1">{f.label}</p>
              <p className="text-[#F3F4F6] text-sm leading-relaxed">{context[f.key] || "—"}</p>
            </div>
          ))}
        </div>

        {/* Narrative Spine */}
        {context.narrative_spine && (
          <div className="bg-[#141417] border border-[#27272A] rounded-lg p-4">
            <p className="panel-overline mb-2">Narrative Spine</p>
            <p className="text-[#F3F4F6] text-sm leading-relaxed font-heading italic">
              {context.narrative_spine}
            </p>
          </div>
        )}

        {/* Speaker / Addressee */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-[#141417] border border-[#27272A] rounded-lg p-4" data-testid="speaker-model">
            <p className="panel-overline mb-2">Speaker</p>
            {context.speaker_model && Object.entries(context.speaker_model).map(([k, v]) => (
              <div key={k} className="flex justify-between text-xs py-1 border-b border-[#1F1F24] last:border-0">
                <span className="text-[#71717A]">{k.replace(/_/g, ' ')}</span>
                <span className="text-[#F3F4F6]">{v || "—"}</span>
              </div>
            ))}
          </div>
          <div className="bg-[#141417] border border-[#27272A] rounded-lg p-4" data-testid="addressee-model">
            <p className="panel-overline mb-2">Addressee</p>
            {context.addressee_model && Object.entries(context.addressee_model).map(([k, v]) => (
              <div key={k} className="flex justify-between text-xs py-1 border-b border-[#1F1F24] last:border-0">
                <span className="text-[#71717A]">{k.replace(/_/g, ' ')}</span>
                <span className="text-[#F3F4F6]">{v || "—"}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Emotional Arc */}
        {context.emotional_arc?.length > 0 && (
          <div className="bg-[#141417] border border-[#27272A] rounded-lg p-4" data-testid="emotional-arc">
            <p className="panel-overline mb-3">Emotional Arc</p>
            <div className="flex gap-2 flex-wrap">
              {context.emotional_arc.map((ea, i) => (
                <div key={i} className="bg-[#1E1E24] rounded px-3 py-2 text-xs">
                  <span className="text-[#D4AF37] font-medium">{ea.phase}</span>
                  <span className="text-[#71717A] mx-1">&rarr;</span>
                  <span className="text-[#F3F4F6]">{ea.emotion}</span>
                  {ea.intensity && (
                    <span className={`ml-1 text-[10px] uppercase ${
                      ea.intensity === 'high' ? 'text-[#EF4444]' : ea.intensity === 'medium' ? 'text-[#F59E0B]' : 'text-[#71717A]'
                    }`}>
                      {ea.intensity}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Confidence Scores */}
        <div className="bg-[#141417] border border-[#27272A] rounded-lg p-4" data-testid="confidence-scores">
          <p className="panel-overline mb-3">Confidence</p>
          <div className="space-y-2">
            {Object.entries(confidence).map(([k, v]) => (
              <div key={k} className="flex items-center justify-between">
                <span className="text-xs text-[#71717A]">{k}</span>
                <div className="flex items-center gap-2">
                  <div className="w-24 h-1.5 bg-[#1E1E24] rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${
                        confLevel(v) === 'high' ? 'bg-[#10B981]' : confLevel(v) === 'medium' ? 'bg-[#F59E0B]' : 'bg-[#EF4444]'
                      }`}
                      style={{ width: `${(v || 0) * 100}%` }}
                    />
                  </div>
                  <span className={`text-xs font-mono ${CONFIDENCE_COLORS[confLevel(v)]}`}>
                    {((v || 0) * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Ambiguity Flags */}
        {context.ambiguity_flags?.length > 0 && (
          <div className="bg-[#141417] border border-[#F59E0B]/20 rounded-lg p-4">
            <p className="panel-overline mb-2 text-[#F59E0B]">Ambiguity Flags</p>
            {context.ambiguity_flags.map((a, i) => (
              <div key={i} className="text-xs py-1.5 border-b border-[#1F1F24] last:border-0">
                <span className="text-[#F59E0B] font-medium">{a.field}:</span>
                <span className="text-[#A1A1AA] ml-1">{a.reason}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </ScrollArea>
  );
}
